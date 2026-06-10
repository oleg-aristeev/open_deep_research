"""Attribution linter: enforce invariant I1 and catch meaning drift.

Contract (design doc §4.4): every fact-asserting sentence of the composed
report must reference an existing verified claim via ``[clm_xxxxxxxx]`` or be
explicitly marked ``(интерпретация)`` / ``(методика)``. Sentences that cite a
claim are additionally NLI-checked against the claim text to catch drift
introduced during synthesis.
"""

import re
from typing import Dict, List

from verify.config import VerifySettings, get_settings
from verify.schema import Claim, LintViolation
from verify.stance import StanceClassifier, grounding_scores_batch

CLAIM_MARKER_RE = re.compile(r"\[(clm_[0-9a-f]{8})\]")
INTERPRETATION_MARKERS = ("(интерпретация)", "(методика)", "(interpretation)", "(methodology)")

_MAX_DRIFT_CHECKS = 40


def split_sentences(text: str) -> List[str]:
    """Split markdown into lintable sentences (headings/blank lines skipped)."""
    sentences: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or set(stripped) <= {"-", "|", " ", ":"}:
            continue
        if stripped.startswith("|"):  # table rows are rendered by code, not linted
            continue
        stripped = re.sub(r"^[-*>]\s+", "", stripped)
        # Split on sentence ends not followed by a claim marker continuation.
        parts = re.split(r"(?<=[.!?])\s+(?=[А-ЯA-Z«\"(\[])", stripped)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def _is_assertive(sentence: str) -> bool:
    """Filter out fragments that carry no factual assertion."""
    bare = CLAIM_MARKER_RE.sub("", sentence).strip(" .!?")
    return len(bare) >= 25


async def lint_report(
    markdown: str,
    claims: List[Claim],
    stance: StanceClassifier | None = None,
    settings: VerifySettings | None = None,
) -> List[LintViolation]:
    """Check the composed (LLM-written) part of the report against the contract."""
    settings = settings or get_settings()
    claims_by_id: Dict[str, Claim] = {c.id: c for c in claims}
    violations: List[LintViolation] = []
    drift_pairs: List[tuple[str, str, str]] = []  # (sentence, claim_id, claim_text)

    for sentence in split_sentences(markdown):
        marker_ids = CLAIM_MARKER_RE.findall(sentence)
        has_interpretation = any(m in sentence.lower() for m in INTERPRETATION_MARKERS)

        if not marker_ids and not has_interpretation:
            if _is_assertive(sentence):
                violations.append(
                    LintViolation(
                        kind="unmarked_sentence",
                        sentence=sentence,
                        detail="Fact-asserting sentence without [clm_*] marker or (интерпретация)/(методика) tag.",
                    )
                )
            continue

        for claim_id in marker_ids:
            claim = claims_by_id.get(claim_id)
            if claim is None:
                violations.append(
                    LintViolation(
                        kind="unknown_claim_id",
                        sentence=sentence,
                        claim_id=claim_id,
                        detail="Referenced claim id does not exist.",
                    )
                )
            elif claim.status in ("pending", "extraction_unstable", "unverified"):
                violations.append(
                    LintViolation(
                        kind="unverified_claim_ref",
                        sentence=sentence,
                        claim_id=claim_id,
                        detail=f"Claim {claim_id} has status '{claim.status}' and may not be asserted.",
                    )
                )
            else:
                bare = CLAIM_MARKER_RE.sub("", sentence).strip()
                drift_pairs.append((bare, claim_id, claim.text))

    # Drift check: the sentence must be entailed by the claim(s) it cites.
    if stance is not None and drift_pairs:
        checked = drift_pairs[:_MAX_DRIFT_CHECKS]
        scores = await grounding_scores_batch(
            stance, [(sentence, claim_text) for sentence, _, claim_text in checked]
        )
        for (sentence, claim_id, _), score in zip(checked, scores):
            if score < settings.thresholds.tau_drift:
                violations.append(
                    LintViolation(
                        kind="drift",
                        sentence=sentence,
                        claim_id=claim_id,
                        detail=f"Sentence is not entailed by claim {claim_id} (grounding={score:.2f}).",
                    )
                )
    return violations
