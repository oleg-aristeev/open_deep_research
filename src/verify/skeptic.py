"""Skeptic-lite: build the hypothesis protocol for one claim (5 templates v0).

One LLM call per claim returning a typed protocol (ACH style): NEGATION,
OUTDATED, SCOPE, NUMERIC (statistical claims only), CIRCULAR. Queries must be
phrased as a critic would phrase them — finding evidence FOR the hypothesis,
not confirmation of the claim (MVP plan §4.4).
"""

import logging
import time
from datetime import datetime
from typing import List

from langchain.chat_models import init_chat_model

from store.traces import RunStore
from verify.config import VerifySettings, get_settings
from verify.prompts import skeptic_numeric_block, skeptic_prompt
from verify.schema import Claim, Hypothesis, Protocol, SkepticProtocol

logger = logging.getLogger(__name__)

_EXPECTED_KINDS = ["NEGATION", "OUTDATED", "SCOPE", "NUMERIC", "CIRCULAR"]


def _fallback_hypotheses(claim: Claim) -> List[Hypothesis]:
    """Deterministic template protocol if the LLM call fails (graceful degradation)."""
    text = claim.text
    hypotheses = [
        Hypothesis(
            kind="NEGATION",
            statement=f"The claim is false: {text}",
            queries=[f"{text} debunked", f"evidence against {text}"],
            expected_signal="refuting stance from tier>=B source",
        ),
        Hypothesis(
            kind="OUTDATED",
            statement=f"The claim is no longer current: {text}",
            queries=[f"{text} update {datetime.now().year}", f"{text} no longer true"],
            expected_signal="fresher source with a different value/state",
        ),
        Hypothesis(
            kind="SCOPE",
            statement=f"The claim holds only under conditions: {text}",
            queries=[f"{text} only when", f"{text} depends on"],
            expected_signal="qualifying stance narrowing the scope",
        ),
        Hypothesis(
            kind="CIRCULAR",
            statement=f"All reporting of the claim traces to one origin: {text}",
            queries=[f'"according to" {text}', f"{text} original source press release"],
            expected_signal="all evidence collapsing into one origin cluster",
        ),
    ]
    if claim.type == "statistical":
        hypotheses.insert(
            3,
            Hypothesis(
                kind="NUMERIC",
                statement=f"The number is wrong or methodology-dependent: {text}",
                queries=[f"{text} alternative estimate", f"{text} methodology criticized"],
                expected_signal="materially different estimate of the same quantity",
            ),
        )
    return hypotheses


async def build_protocol(
    claim: Claim,
    settings: VerifySettings | None = None,
    run_store: RunStore | None = None,
) -> Protocol:
    """Generate the skeptic protocol for one claim."""
    from open_deep_research.utils import get_today_str

    settings = settings or get_settings()
    model = init_chat_model(
        model=settings.models.skeptic_model,
        max_tokens=settings.models.max_tokens,
        tags=["langsmith:nostream"],
    ).with_structured_output(SkepticProtocol).with_retry(stop_after_attempt=2)

    numeric_block = skeptic_numeric_block if claim.type == "statistical" else ""
    prompt = skeptic_prompt.format(
        date=get_today_str(),
        year=datetime.now().year,
        claim_json=claim.model_dump_json(
            include={"id", "text", "type", "scope", "entities", "quantity"}
        ),
        numeric_block=numeric_block,
    )
    started = time.monotonic()
    try:
        result: SkepticProtocol = await model.ainvoke(prompt)
        hypotheses = result.hypotheses
    except Exception as e:
        logger.warning("Skeptic LLM call failed for %s (%s); using template fallback", claim.id, e)
        hypotheses = _fallback_hypotheses(claim)

    # Enforce the v0 contract: known kinds only, NUMERIC only for statistical,
    # dedupe kinds, cap at the budget.
    allowed = {k for k in _EXPECTED_KINDS if k != "NUMERIC" or claim.type == "statistical"}
    seen_kinds: set[str] = set()
    cleaned: List[Hypothesis] = []
    for hyp in hypotheses:
        if hyp.kind not in allowed or hyp.kind in seen_kinds or not hyp.queries:
            continue
        seen_kinds.add(hyp.kind)
        hyp.queries = hyp.queries[: settings.budget.queries_per_hypothesis + 1]
        cleaned.append(hyp)
    if not cleaned:
        cleaned = _fallback_hypotheses(claim)
    cleaned = cleaned[: settings.budget.max_hypotheses_per_claim]

    protocol = Protocol(claim_id=claim.id, hypotheses=cleaned)
    if run_store:
        run_store.trace(
            node="skeptic",
            kind="llm",
            payload={"claim_id": claim.id, "kinds": [h.kind for h in cleaned]},
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    return protocol
