"""Claimify: extract atomic, decontextualized, typed claims from findings.

One frontier-model call per findings batch (MVP plan §4.3), followed by a
code-level quality gate: every claim must be grounded in its source finding
(grounding score >= tau_extract), otherwise it is flagged
``extraction_unstable`` and goes to the report as an unverified observation
instead of entering verification.
"""

import json
import logging
import time
from typing import List

from langchain.chat_models import init_chat_model

from store.traces import RunStore
from verify.config import VerifySettings, get_settings
from verify.prompts import claimify_prompt
from verify.schema import (
    Claim,
    ClaimifyResult,
    ClaimScope,
    Finding,
    Quantity,
)
from verify.sources import extract_domain
from verify.stance import StanceClassifier, grounding_scores_batch

logger = logging.getLogger(__name__)

_MAX_FINDINGS_IN_PROMPT = 80


def _format_findings(findings: List[Finding]) -> str:
    lines = []
    for f in findings[:_MAX_FINDINGS_IN_PROMPT]:
        lines.append(
            json.dumps(
                {"id": f.id, "text": f.text, "quote": f.verbatim_quote, "url": f.url},
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


async def claimify(
    findings: List[Finding],
    brief: str,
    settings: VerifySettings | None = None,
    run_store: RunStore | None = None,
    stance: StanceClassifier | None = None,
) -> List[Claim]:
    """Extract claims from findings and gate them on extraction stability."""
    from open_deep_research.utils import get_today_str

    settings = settings or get_settings()
    if not findings:
        return []

    model = init_chat_model(
        model=settings.models.claimify_model,
        max_tokens=settings.models.max_tokens,
        tags=["langsmith:nostream"],
    ).with_structured_output(ClaimifyResult).with_retry(stop_after_attempt=3)

    prompt = claimify_prompt.format(
        date=get_today_str(),
        brief=brief,
        findings=_format_findings(findings),
    )
    started = time.monotonic()
    result: ClaimifyResult = await model.ainvoke(prompt)

    findings_by_id = {f.id: f for f in findings}
    claims: List[Claim] = []
    for extracted in result.claims:
        derived = [fid for fid in extracted.derived_from_finding_ids if fid in findings_by_id]
        origin_domains = sorted(
            {extract_domain(findings_by_id[fid].url) for fid in derived}
        )
        quantity = None
        if extracted.quantity_value is not None or extracted.quantity_unit:
            quantity = Quantity(
                value=extracted.quantity_value,
                unit=extracted.quantity_unit,
                comparison_base=extracted.comparison_base,
            )
        claims.append(
            Claim(
                text=extracted.text,
                type=extracted.type,
                scope=ClaimScope(
                    time=extracted.time,
                    geo=extracted.geo,
                    population=extracted.population,
                    conditions=extracted.conditions,
                ),
                entities=extracted.entities,
                quantity=quantity,
                decision_relevance=max(0.0, min(1.0, extracted.decision_relevance)),
                derived_from=derived,
                origin_domains=origin_domains,
                ambiguity_note=extracted.ambiguity_note,
            )
        )

    if run_store:
        run_store.trace(
            node="claimify",
            kind="llm",
            payload={"n_findings": len(findings), "n_claims": len(claims)},
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    # Quality gate: claim must be entailed by its source finding(s).
    if stance is not None and claims:
        pairs = []
        for claim in claims:
            source_text = " ".join(
                findings_by_id[fid].verbatim_quote for fid in claim.derived_from
            ) or claim.text
            pairs.append((claim.text, source_text))
        scores = await grounding_scores_batch(stance, pairs)
        for claim, score in zip(claims, scores):
            if not claim.derived_from or score < settings.thresholds.tau_extract:
                claim.status = "extraction_unstable"
                logger.info(
                    "Claim %s flagged extraction_unstable (grounding=%.2f)", claim.id, score
                )
    return claims


def rank_by_decision_relevance(claims: List[Claim], max_claims: int) -> List[Claim]:
    """Stable verification queue: most decision-relevant stable claims first."""
    stable = [c for c in claims if c.status != "extraction_unstable"]
    stable.sort(key=lambda c: c.decision_relevance, reverse=True)
    return stable[:max_claims]
