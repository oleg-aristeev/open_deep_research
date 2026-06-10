"""Note-taker: turn page snapshots into findings with verbatim-quote locators.

One mid-tier model call per page with structured output. Hard rule (MVP plan
§4.2): ``verbatim_quote`` must be an exact substring of the snapshot text —
checked in code, one retry on failure; still-unverified quotes are dropped
from verification input.
"""

import asyncio
import logging
import time
from typing import List

from langchain.chat_models import init_chat_model

from store.snapshots import SnapshotStore
from store.traces import RunStore
from verify.config import VerifySettings, get_settings
from verify.prompts import CONTENT_FIREWALL, notetaker_prompt
from verify.schema import Finding, NoteTakerResult

logger = logging.getLogger(__name__)

_MAX_PAGE_CHARS = 30000
_MAX_FINDINGS_PER_PAGE = 6
_CONCURRENCY = 8


def _build_model(settings: VerifySettings):
    model = init_chat_model(
        model=settings.models.notetaker_model,
        max_tokens=settings.models.max_tokens,
        tags=["langsmith:nostream"],
    )
    return model.with_structured_output(NoteTakerResult).with_retry(stop_after_attempt=2)


async def _notetake_page(
    model,
    snapshot_store: SnapshotStore,
    snapshot_id: str,
    url: str,
    topic: str,
    run_store: RunStore | None,
) -> List[Finding]:
    from open_deep_research.utils import get_today_str

    text = snapshot_store.get_text(snapshot_id)
    if not text:
        return []
    prompt = notetaker_prompt.format(
        date=get_today_str(),
        firewall=CONTENT_FIREWALL,
        topic=topic,
        url=url,
        max_findings=_MAX_FINDINGS_PER_PAGE,
        page_text=text[:_MAX_PAGE_CHARS],
    )
    findings: List[Finding] = []
    started = time.monotonic()
    for attempt in range(2):  # second pass only re-asks for failed quotes
        try:
            result: NoteTakerResult = await model.ainvoke(prompt)
        except Exception as e:
            logger.warning("Note-taker failed for %s: %s", url, e)
            break
        findings = []
        all_verified = True
        for item in result.findings[:_MAX_FINDINGS_PER_PAGE]:
            located = snapshot_store.locate(snapshot_id, item.verbatim_quote)
            finding = Finding(
                text=item.text,
                verbatim_quote=item.verbatim_quote,
                snapshot_id=snapshot_id,
                url=url,
                quote_verified=located is not None,
            )
            if located:
                finding.char_start, finding.char_end = located
            else:
                all_verified = False
            findings.append(finding)
        if all_verified or attempt == 1:
            break
        prompt += (
            "\n\nIMPORTANT: in your previous answer some verbatim_quote values were NOT "
            "exact substrings of the page text. Copy quotes character-for-character."
        )
    verified = [f for f in findings if f.quote_verified]
    dropped = len(findings) - len(verified)
    if run_store:
        run_store.trace(
            node="notetaker",
            kind="llm",
            payload={"url": url, "findings": len(verified), "dropped_unverified": dropped},
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    return verified


async def extract_findings(
    run_store: RunStore,
    snapshot_store: SnapshotStore,
    topic: str,
    settings: VerifySettings | None = None,
) -> List[Finding]:
    """Extract locator-backed findings from all snapshots captured in this run."""
    settings = settings or get_settings()
    hits = run_store.load_raw_hits()

    # Unique snapshots, first-seen order (explore visited them in relevance order).
    seen: dict[str, str] = {}
    for hit in hits:
        if hit.snapshot_id and hit.snapshot_id not in seen:
            seen[hit.snapshot_id] = hit.url
    pages = list(seen.items())[: settings.budget.max_findings_pages]
    if not pages:
        logger.warning("No snapshots recorded for run %s; nothing to note-take", run_store.run_id)
        return []

    model = _build_model(settings)
    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def one(snapshot_id: str, url: str) -> List[Finding]:
        async with semaphore:
            return await _notetake_page(model, snapshot_store, snapshot_id, url, topic, run_store)

    results = await asyncio.gather(*(one(sid, url) for sid, url in pages))
    findings = [f for page_findings in results for f in page_findings]
    run_store.save_json("findings.json", [f.model_dump(mode="json") for f in findings])
    return findings
