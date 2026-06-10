"""LangGraph subgraph for the verify phase (MVP plan §3.2).

Flow::

    notetake -> claimify -> rank --Send×K--> verify_one -> collect
        -> compose -> lint -> (repair <=2) -> finalize -> END

Claims are verified concurrently (Send API); inside one claim the hypotheses
run sequentially, which is cheap and enables early stopping: once NEGATION is
confirmed by a quality source, the remaining hypotheses get half the query
budget. The subgraph is a drop-in replacement for ODR's
``final_report_generation`` node: it consumes ``research_brief``/``notes`` and
produces ``final_report`` + ``messages``.
"""

import logging
import operator
import time
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, MessageLikeRepresentation
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from open_deep_research.state import override_reducer
from store.snapshots import get_snapshot_store
from store.traces import get_run_store, run_id_from_config
from verify import aggregate as aggregate_mod
from verify import claimify as claimify_mod
from verify import compose as compose_mod
from verify import hunter as hunter_mod
from verify import linter as linter_mod
from verify import skeptic as skeptic_mod
from verify.config import get_settings
from verify.schema import Claim, Finding, LintViolation, VerifiedReport
from verify.stance import get_stance_classifier

logger = logging.getLogger(__name__)


class VerifyState(TypedDict, total=False):
    """State of the verify subgraph."""

    # inputs (provided by the parent ODR graph)
    research_brief: str
    notes: Annotated[list[str], override_reducer]
    messages: Annotated[list[MessageLikeRepresentation], operator.add]
    # internal
    run_id: str
    findings: List[Finding]
    claims: List[Claim]
    unverified_claims: List[Claim]
    verified_claims: Annotated[List[Claim], operator.add]
    synthesis: str
    violations: List[LintViolation]
    repairs: int
    started_at: float
    # output
    final_report: str


class VerifyClaimState(TypedDict):
    """Payload of one Send-dispatched claim verification."""

    claim: Claim
    run_id: str
    research_brief: str


async def notetake_node(state: VerifyState, config: RunnableConfig) -> dict:
    """Extract locator-backed findings from the snapshots captured during explore."""
    from verify.findings import extract_findings

    settings = get_settings()
    run_id = run_id_from_config(config)
    run_store = get_run_store(run_id)
    run_store.trace(node="verify", kind="phase_start", payload={"brief": state.get("research_brief", "")[:500]})

    findings = await extract_findings(
        run_store=run_store,
        snapshot_store=get_snapshot_store(),
        topic=state.get("research_brief", ""),
        settings=settings,
    )
    if not findings:
        # Fallback: no snapshots (e.g. non-Tavily search API). Build locator-less
        # findings from compressed notes so the pipeline still runs; citations
        # will be weaker and claims are gated by the grounding check anyway.
        logger.warning("No snapshot findings for run %s; falling back to notes", run_id)
        findings = [
            Finding(
                text=note[:1500],
                verbatim_quote=note[:1500],
                snapshot_id="",
                url="",
                quote_verified=False,
            )
            for note in state.get("notes", [])[:20]
            if note.strip()
        ]
    return {"run_id": run_id, "findings": findings, "started_at": time.monotonic()}


async def claimify_node(state: VerifyState, config: RunnableConfig) -> dict:
    """Extract claims from findings and split off unstable extractions."""
    settings = get_settings()
    run_store = get_run_store(state["run_id"])
    stance = get_stance_classifier(settings, run_store)
    claims = await claimify_mod.claimify(
        findings=state.get("findings", []),
        brief=state.get("research_brief", ""),
        settings=settings,
        run_store=run_store,
        stance=stance,
    )
    unverified = [c for c in claims if c.status == "extraction_unstable"]
    stable = [c for c in claims if c.status != "extraction_unstable"]
    return {"claims": stable, "unverified_claims": unverified}


def rank_node(state: VerifyState) -> dict:
    """Keep the top-K decision-relevant claims within budget."""
    settings = get_settings()
    ranked = claimify_mod.rank_by_decision_relevance(
        state.get("claims", []), settings.budget.max_claims
    )
    return {"claims": ranked}


def fan_out_claims(state: VerifyState):
    """Dispatch one verify_one task per ranked claim (Send API)."""
    claims = state.get("claims", [])
    if not claims:
        return "collect"
    return [
        Send(
            "verify_one",
            VerifyClaimState(
                claim=claim,
                run_id=state["run_id"],
                research_brief=state.get("research_brief", ""),
            ),
        )
        for claim in claims
    ]


async def verify_one_node(state: VerifyClaimState, config: RunnableConfig) -> dict:
    """Full verification cycle for a single claim."""
    settings = get_settings()
    claim = state["claim"]
    run_store = get_run_store(state["run_id"])
    snapshot_store = get_snapshot_store()
    stance = get_stance_classifier(settings, run_store)

    protocol = await skeptic_mod.build_protocol(claim, settings, run_store)
    seen_urls: set[str] = set()
    stance_checks = 0
    executed_hypotheses = 0
    early_stop = False

    for hypothesis in protocol.hypotheses:
        max_queries = None
        if early_stop:
            # NEGATION confirmed by a quality source: halve remaining budget.
            max_queries = max(1, settings.budget.queries_per_hypothesis // 2)
        evidence, search_log = await hunter_mod.gather(
            hypothesis=hypothesis,
            claim_id=claim.id,
            exclude_domains=set(claim.origin_domains),
            seen_urls=seen_urls,
            snapshot_store=snapshot_store,
            run_store=run_store,
            settings=settings,
            config=config,
            max_queries=max_queries,
        )
        protocol.searches.extend(search_log)
        executed_hypotheses += 1

        for ev in evidence:
            if stance_checks >= settings.budget.max_stance_checks_per_claim:
                break
            result = await stance.classify(
                claim.text,
                ev.quote,
                domain=ev.source_domain,
                published=str(ev.published_at or "unknown"),
            )
            stance_checks += 1
            ev.stance = result.stance
            ev.stance_score = max(0.0, min(1.0, result.score))
            # Pin the minimal quote back to the snapshot for exact offsets.
            if result.relevant_quote and ev.locator and ev.locator.snapshot_id:
                located = snapshot_store.locate(ev.locator.snapshot_id, result.relevant_quote)
                if located:
                    ev.quote = result.relevant_quote
                    ev.locator.char_start, ev.locator.char_end = located
            if ev.stance != "mentions":
                claim.evidence.append(ev)
            if (
                hypothesis.kind == "NEGATION"
                and ev.stance == "refutes"
                and ev.source_tier in ("A", "B")
                and ev.stance_score >= 0.7
            ):
                early_stop = True

    protocol.coverage = (
        executed_hypotheses / len(protocol.hypotheses) if protocol.hypotheses else 0.0
    )
    claim.protocol = protocol

    from verify.sources import cluster_origins

    claim.evidence = cluster_origins(claim.evidence)
    sigma, flags = aggregate_mod.aggregate(claim, settings)
    claim.sigma = sigma
    claim.flags = flags
    claim.status = aggregate_mod.status_from(sigma, flags)

    run_store.trace(
        node="verify_one",
        kind="claim_done",
        payload={
            "claim_id": claim.id,
            "status": claim.status,
            "p_true": sigma.p_true,
            "flags": flags,
            "n_evidence": len(claim.evidence),
            "stance_checks": stance_checks,
        },
    )
    return {"verified_claims": [claim]}


def collect_node(state: VerifyState) -> dict:
    """Fan-in: order verified claims by decision relevance."""
    claims = sorted(
        state.get("verified_claims", []),
        key=lambda c: c.decision_relevance,
        reverse=True,
    )
    return {"claims": claims, "repairs": 0}


async def compose_node(state: VerifyState, config: RunnableConfig) -> dict:
    """Write the synthesis from verified claims only."""
    settings = get_settings()
    run_store = get_run_store(state["run_id"])
    synthesis = await compose_mod.compose_synthesis(
        brief=state.get("research_brief", ""),
        claims=state.get("claims", []),
        unverified=state.get("unverified_claims", []),
        settings=settings,
        run_store=run_store,
    )
    return {"synthesis": synthesis}


async def lint_node(state: VerifyState, config: RunnableConfig) -> dict:
    """Check invariant I1 + drift on the synthesis."""
    settings = get_settings()
    run_store = get_run_store(state["run_id"])
    stance = get_stance_classifier(settings, run_store)
    violations = await linter_mod.lint_report(
        state.get("synthesis", ""), state.get("claims", []), stance, settings
    )
    run_store.trace(
        node="lint",
        kind="check",
        payload={"n_violations": len(violations), "repairs_done": state.get("repairs", 0)},
    )
    return {"violations": violations}


def route_after_lint(state: VerifyState) -> str:
    """Repair (bounded) or finalize."""
    settings = get_settings()
    if state.get("violations") and state.get("repairs", 0) < settings.thresholds.max_repairs:
        return "repair"
    return "finalize"


async def repair_node(state: VerifyState, config: RunnableConfig) -> dict:
    """One bounded repair iteration of the synthesis."""
    settings = get_settings()
    run_store = get_run_store(state["run_id"])
    fixed = await compose_mod.repair_synthesis(
        state.get("synthesis", ""),
        state.get("violations", []),
        state.get("claims", []),
        settings,
        run_store,
    )
    return {"synthesis": fixed, "repairs": state.get("repairs", 0) + 1}


async def finalize_node(state: VerifyState, config: RunnableConfig) -> Command:
    """Assemble the report, persist artifacts, emit final_report + message."""
    run_store = get_run_store(state["run_id"])
    claims = state.get("claims", [])
    report_md = compose_mod.assemble_report(
        state.get("synthesis", ""), claims, state.get("violations", [])
    )
    report = VerifiedReport(
        run_id=state["run_id"],
        question=state.get("research_brief", ""),
        markdown=report_md,
        claims=claims,
        unverified_observations=state.get("unverified_claims", []),
        lint_violations=state.get("violations", []),
    )
    run_store.save_text("report.md", report_md)
    run_store.save_json("claim_graph.json", report.claim_graph())
    elapsed = time.monotonic() - state.get("started_at", time.monotonic())
    run_store.trace(
        node="verify",
        kind="phase_done",
        payload={
            "n_claims": len(claims),
            "n_disputed": sum(1 for c in claims if "disputed" in c.flags),
            "elapsed_s": round(elapsed, 1),
        },
    )
    return Command(
        goto=END,
        update={
            "final_report": report_md,
            "messages": [AIMessage(content=report_md)],
            "notes": {"type": "override", "value": []},
        },
    )


def build_verify_graph(checkpointer: object | None = None):
    """Compile the verify subgraph."""
    builder = StateGraph(VerifyState)
    builder.add_node("notetake", notetake_node)
    builder.add_node("claimify", claimify_node)
    builder.add_node("rank", rank_node)
    builder.add_node("verify_one", verify_one_node)
    builder.add_node("collect", collect_node)
    builder.add_node("compose", compose_node)
    builder.add_node("lint", lint_node)
    builder.add_node("repair", repair_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "notetake")
    builder.add_edge("notetake", "claimify")
    builder.add_edge("claimify", "rank")
    builder.add_conditional_edges("rank", fan_out_claims, ["verify_one", "collect"])
    builder.add_edge("verify_one", "collect")
    builder.add_edge("collect", "compose")
    builder.add_edge("compose", "lint")
    builder.add_conditional_edges("lint", route_after_lint, {"repair": "repair", "finalize": "finalize"})
    builder.add_edge("repair", "lint")
    return builder.compile(checkpointer=checkpointer)


verify_subgraph = build_verify_graph()
