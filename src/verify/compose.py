"""Composer: verified-claims-only report synthesis + deterministic rendering.

The LLM writes the synthesis from verified claims ONLY (findings are never in
the prompt — the main defense against unverified facts leaking into the
report). Code renders the claims table, the Phi summary and the lint-failure
block; honesty beats polish (MVP plan §4.8).
"""

import json
import logging
import time
from typing import List

from langchain.chat_models import init_chat_model

from store.traces import RunStore
from verify.aggregate import verdict_from
from verify.config import VerifySettings, get_settings
from verify.prompts import composer_prompt, repair_prompt
from verify.schema import Claim, LintViolation

logger = logging.getLogger(__name__)

_FLAG_LABELS = {
    "disputed": "⚠️ disputed",
    "single_origin": "single origin",
    "stale": "stale",
    "not_enough_evidence": "not enough evidence",
}


def _claim_brief(claim: Claim) -> dict:
    """Compact claim view fed to the composer (the only permitted fact source)."""
    sigma = claim.sigma
    top_for = sorted(claim.evidence_for, key=lambda e: e.stance_score, reverse=True)[:3]
    top_against = sorted(claim.evidence_against, key=lambda e: e.stance_score, reverse=True)[:3]
    return {
        "id": claim.id,
        "text": claim.text,
        "type": claim.type,
        "verdict": verdict_from(sigma, claim.flags) if sigma else "unverified",
        "p_true": sigma.p_true if sigma else None,
        "verbal": sigma.verbal if sigma else None,
        "agreement": sigma.agreement if sigma else None,
        "flags": claim.flags,
        "sources_for": [e.source_domain for e in top_for],
        "sources_against": [
            {"domain": e.source_domain, "stance": e.stance, "quote": e.quote[:200]}
            for e in top_against
        ],
    }


def render_claims_table(claims: List[Claim]) -> str:
    """Deterministic markdown table of all verified claims."""
    lines = [
        "| ID | Утверждение | Вердикт | p (вербально) | Grade | Флаги | За | Против |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for claim in claims:
        sigma = claim.sigma
        verdict = verdict_from(sigma, claim.flags) if sigma else "unverified"
        p_str = f"{sigma.p_true:.2f} ({sigma.verbal})" if sigma else "—"
        grade = sigma.evidence_grade if sigma else "—"
        flags = ", ".join(_FLAG_LABELS.get(f, f) for f in claim.flags) or "—"
        sources_for = ", ".join(sorted({e.source_domain for e in claim.evidence_for})[:3]) or "—"
        sources_against = (
            ", ".join(sorted({e.source_domain for e in claim.evidence_against})[:3]) or "—"
        )
        text = claim.text.replace("|", "\\|")
        lines.append(
            f"| `{claim.id}` | {text} | **{verdict}** | {p_str} | {grade} | {flags} "
            f"| {sources_for} | {sources_against} |"
        )
    return "\n".join(lines)


def render_phi_summary(claims: List[Claim]) -> str:
    """Render the 'how we hunted for counter-evidence' block from protocols."""
    lines = []
    total_searches = 0
    for claim in claims:
        if not claim.protocol:
            continue
        kinds = [h.kind for h in claim.protocol.hypotheses]
        n_searches = len(claim.protocol.searches)
        total_searches += n_searches
        lines.append(
            f"- `{claim.id}`: гипотезы {', '.join(kinds)}; запросов: {n_searches}; "
            f"покрытие протокола: {claim.protocol.coverage:.0%}"
        )
    header = (
        f"Всего выполнено {total_searches} целевых поисков опровержений "
        f"по {len(lines)} утверждениям. «Не нашли опровержений» заявляется только "
        f"после выполнения протокола; отсутствие свидетельств повышает u, а не p."
    )
    return header + "\n\n" + "\n".join(lines)


async def compose_synthesis(
    brief: str,
    claims: List[Claim],
    unverified: List[Claim],
    settings: VerifySettings | None = None,
    run_store: RunStore | None = None,
) -> str:
    """LLM call: write the synthesis sections from verified claims only."""
    from open_deep_research.utils import get_today_str

    settings = settings or get_settings()
    model = init_chat_model(
        model=settings.models.composer_model,
        max_tokens=settings.models.composer_max_tokens,
        tags=["langsmith:nostream"],
    )
    claims_json = json.dumps([_claim_brief(c) for c in claims], ensure_ascii=False, indent=1)
    unverified_json = json.dumps(
        [{"text": c.text, "note": c.ambiguity_note or "extraction unstable"} for c in unverified],
        ensure_ascii=False,
    )
    prompt = composer_prompt.format(
        date=get_today_str(),
        brief=brief,
        claims_json=claims_json,
        unverified_json=unverified_json,
    )
    started = time.monotonic()
    response = await model.ainvoke(prompt)
    if run_store:
        run_store.trace(
            node="composer",
            kind="llm",
            payload={"n_claims": len(claims)},
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    return str(response.content)


async def repair_synthesis(
    markdown: str,
    violations: List[LintViolation],
    claims: List[Claim],
    settings: VerifySettings | None = None,
    run_store: RunStore | None = None,
) -> str:
    """One repair iteration: fix only the listed lint violations."""
    settings = settings or get_settings()
    model = init_chat_model(
        model=settings.models.composer_model,
        max_tokens=settings.models.composer_max_tokens,
        tags=["langsmith:nostream"],
    )
    violations_text = "\n".join(
        f"- [{v.kind}] {v.detail} Sentence: {v.sentence[:300]}" for v in violations
    )
    prompt = repair_prompt.format(
        violations=violations_text,
        valid_ids=", ".join(c.id for c in claims),
        report=markdown,
    )
    response = await model.ainvoke(prompt)
    if run_store:
        run_store.trace(node="composer_repair", kind="llm", payload={"n_violations": len(violations)})
    return str(response.content)


def assemble_report(
    synthesis_md: str,
    claims: List[Claim],
    remaining_violations: List[LintViolation],
) -> str:
    """Append the code-rendered sections to the LLM synthesis."""
    parts = [synthesis_md.strip()]
    parts.append("\n## Таблица утверждений\n\n" + render_claims_table(claims))
    parts.append("\n## Φ: как мы искали опровержения\n\n" + render_phi_summary(claims))
    if remaining_violations:
        failed = "\n".join(
            f"- [{v.kind}] {v.sentence[:200]} — {v.detail}" for v in remaining_violations
        )
        parts.append(
            "\n## ⚠️ Не прошло линт атрибуции\n\n"
            "Следующие предложения не прошли проверку привязки к утверждениям "
            "после исчерпания repair-итераций (честность важнее красоты):\n\n" + failed
        )
    return "\n".join(parts) + "\n"
