"""Stance classification: (claim, evidence) -> supports/refutes/qualifies/mentions.

Two interchangeable backends behind one interface (MVP plan §4.5):

* ``LLMStanceClassifier`` — cheap LLM with a rubric and structured output.
  Fast to ship, expensive to scale. Default.
* ``MiniCheckStanceClassifier`` — Bespoke-MiniCheck-7B behind a vLLM
  OpenAI-compatible endpoint for binary grounding, with LLM fallback for
  refutes/qualifies discrimination and borderline scores.

The same interface also powers grounding checks (claim<->finding stability,
report-sentence<->claim drift, citation precision in evals).
"""

import asyncio
import logging
import time
from typing import Protocol, runtime_checkable

import httpx
from langchain.chat_models import init_chat_model

from store.traces import RunStore
from verify.config import VerifySettings, get_settings
from verify.prompts import CONTENT_FIREWALL, grounding_prompt, stance_prompt
from verify.schema import StanceResult

logger = logging.getLogger(__name__)

_MAX_EVIDENCE_CHARS = 6000


@runtime_checkable
class StanceClassifier(Protocol):
    """Interface: classify evidence stance towards a claim."""

    async def classify(
        self,
        claim_text: str,
        evidence_text: str,
        domain: str = "",
        published: str = "unknown",
    ) -> StanceResult:
        """Return stance + confidence for one (claim, evidence) pair."""
        ...


class LLMStanceClassifier:
    """Rubric-driven LLM stance classifier with structured output."""

    def __init__(self, settings: VerifySettings | None = None, run_store: RunStore | None = None):
        """Build the classifier from verify settings."""
        self.settings = settings or get_settings()
        self.run_store = run_store
        model = init_chat_model(
            model=self.settings.models.stance_model,
            max_tokens=1024,
            tags=["langsmith:nostream"],
        )
        self._model = model.with_structured_output(StanceResult).with_retry(
            stop_after_attempt=2
        )

    async def classify(
        self,
        claim_text: str,
        evidence_text: str,
        domain: str = "",
        published: str = "unknown",
    ) -> StanceResult:
        """Classify with the stance rubric; fail-safe to 'mentions'."""
        from open_deep_research.utils import get_today_str

        prompt = stance_prompt.format(
            date=get_today_str(),
            claim=claim_text,
            domain=domain or "unknown source",
            published=published,
            evidence=evidence_text[:_MAX_EVIDENCE_CHARS],
            firewall=CONTENT_FIREWALL,
        )
        started = time.monotonic()
        try:
            result: StanceResult = await self._model.ainvoke(prompt)
        except Exception as e:  # never let one stance call kill a claim
            logger.warning("Stance classification failed: %s", e)
            result = StanceResult(stance="mentions", score=0.0)
        if self.run_store:
            self.run_store.trace(
                node="stance",
                kind="llm",
                payload={
                    "claim": claim_text[:200],
                    "domain": domain,
                    "stance": result.stance,
                    "score": result.score,
                },
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        return result


class MiniCheckStanceClassifier:
    """MiniCheck (vLLM) for binary grounding + LLM for the hard distinctions.

    MiniCheck answers "does the document support the claim" (yes/no with a
    probability). High support -> supports. Otherwise we cannot tell refutes
    from qualifies/mentions, so those (and borderline scores) go to the LLM.
    """

    def __init__(self, settings: VerifySettings | None = None, run_store: RunStore | None = None):
        """Build the classifier; requires settings.stance.minicheck_url (vLLM)."""
        self.settings = settings or get_settings()
        if not self.settings.stance.minicheck_url:
            raise ValueError("stance.minicheck_url is not configured")
        self._llm_fallback = LLMStanceClassifier(self.settings, run_store)
        self.run_store = run_store

    async def _minicheck_support_prob(self, document: str, claim: str) -> float:
        """Call MiniCheck via the OpenAI-compatible completions API."""
        payload = {
            "model": self.settings.stance.minicheck_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Document: {document[:_MAX_EVIDENCE_CHARS]}\n"
                        f"Claim: {claim}\n"
                        "Is the claim supported by the document? Answer Yes or No."
                    ),
                }
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 5,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.settings.stance.minicheck_url.rstrip("/") + "/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        try:
            top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            probs = {t["token"].strip().lower(): 2.718281828 ** t["logprob"] for t in top}
            yes = probs.get("yes", 0.0)
            no = probs.get("no", 0.0)
            return yes / (yes + no) if (yes + no) > 0 else 0.5
        except (KeyError, IndexError):
            text = data["choices"][0]["message"]["content"].strip().lower()
            return 1.0 if text.startswith("yes") else 0.0

    async def classify(
        self,
        claim_text: str,
        evidence_text: str,
        domain: str = "",
        published: str = "unknown",
    ) -> StanceResult:
        """MiniCheck first; defer to the LLM on non-support / borderline."""
        try:
            p_support = await self._minicheck_support_prob(evidence_text, claim_text)
        except Exception as e:
            logger.warning("MiniCheck unavailable (%s); falling back to LLM", e)
            return await self._llm_fallback.classify(claim_text, evidence_text, domain, published)
        band = self.settings.stance.borderline_band
        if p_support >= 0.5 + band:
            return StanceResult(stance="supports", score=p_support)
        # Low/borderline support: the LLM decides refutes vs qualifies vs mentions.
        return await self._llm_fallback.classify(claim_text, evidence_text, domain, published)


def get_stance_classifier(
    settings: VerifySettings | None = None, run_store: RunStore | None = None
) -> StanceClassifier:
    """Build the configured stance backend."""
    settings = settings or get_settings()
    if settings.stance.backend == "minicheck" and settings.stance.minicheck_url:
        try:
            return MiniCheckStanceClassifier(settings, run_store)
        except ValueError:
            pass
    return LLMStanceClassifier(settings, run_store)


async def grounding_score(
    classifier: StanceClassifier, statement: str, document: str
) -> float:
    """P(document supports statement) via the stance backend.

    Used for: claim<->finding extraction stability (tau_extract), report
    sentence<->claim drift (tau_drift), and citation precision in evals.
    """
    if isinstance(classifier, LLMStanceClassifier):
        from open_deep_research.utils import (
            get_today_str,  # noqa: F401  (date unused here)
        )

        prompt = grounding_prompt.format(
            document=document[:_MAX_EVIDENCE_CHARS],
            statement=statement,
            firewall=CONTENT_FIREWALL,
        )
        try:
            result: StanceResult = await classifier._model.ainvoke(prompt)
        except Exception:
            return 0.5
        if result.stance == "supports":
            return result.score
        if result.stance == "refutes":
            return 1.0 - result.score
        if result.stance == "qualifies":
            return 0.5
        return max(0.0, 0.5 - 0.2 * result.score)
    result = await classifier.classify(statement, document)
    return result.score if result.stance == "supports" else 1.0 - result.score


async def grounding_scores_batch(
    classifier: StanceClassifier, pairs: list[tuple[str, str]], concurrency: int = 8
) -> list[float]:
    """Grounding scores for many (statement, document) pairs with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(statement: str, document: str) -> float:
        async with semaphore:
            return await grounding_score(classifier, statement, document)

    return list(await asyncio.gather(*(one(s, d) for s, d in pairs)))
