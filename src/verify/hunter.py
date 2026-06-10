"""Evidence hunter: targeted retrieval for one skeptic hypothesis.

Runs the hypothesis queries through the search provider (Tavily, the MVP's
single provider), snapshots every fetched page, and returns evidence
candidates with locators. Origin domains of the claim are excluded so we
actually hunt for independent counter-evidence, not for the claim's own
sources (MVP plan §4.5 / design doc §6).
"""

import logging
from datetime import date, datetime, timezone
from typing import List, Set, Tuple

from langchain_core.runnables import RunnableConfig

from store.snapshots import SnapshotStore
from store.traces import RunStore
from verify.config import VerifySettings, get_settings
from verify.schema import Evidence, EvidenceType, Hypothesis, Locator, SearchLogEntry
from verify.sources import extract_domain, get_registry

logger = logging.getLogger(__name__)

_MAX_CANDIDATE_CHARS = 4000


def _parse_published(result: dict) -> date | None:
    raw = result.get("published_date") or result.get("published_at")
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(raw)[:31], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _guess_evidence_type(url: str, tier: str) -> EvidenceType:
    """Cheap heuristic; refined in Phase 2 with real source intelligence."""
    lowered = url.lower()
    if any(token in lowered for token in ("/opinion", "/blog", "/commentary", "substack.com", "medium.com")):
        return "opinion"
    if any(token in lowered for token in (".gov", "sec.gov", "arxiv.org", "clinicaltrials", "press-release", "pressrelease", "investor")):
        return "primary"
    if any(token in lowered for token in ("data.", "/dataset", "statista", "ourworldindata")):
        return "dataset"
    return "primary" if tier == "A" else "secondary"


async def gather(
    hypothesis: Hypothesis,
    claim_id: str,
    exclude_domains: Set[str],
    seen_urls: Set[str],
    snapshot_store: SnapshotStore,
    run_store: RunStore | None = None,
    settings: VerifySettings | None = None,
    config: RunnableConfig | None = None,
    max_queries: int | None = None,
) -> Tuple[List[Evidence], List[SearchLogEntry]]:
    """Run hypothesis queries, snapshot pages, return evidence candidates.

    ``seen_urls`` is shared across hypotheses of one claim to avoid paying for
    the same page twice. The returned Evidence objects carry the page excerpt
    in ``quote`` and are not yet stance-classified.
    """
    from open_deep_research.utils import tavily_search_async

    settings = settings or get_settings()
    registry = get_registry()
    queries = hypothesis.queries[: max_queries or settings.budget.queries_per_hypothesis]
    if not queries:
        return [], []

    try:
        responses = await tavily_search_async(
            queries,
            max_results=settings.budget.results_per_query,
            topic="general",
            include_raw_content=True,
            config=config,
        )
    except Exception as e:
        logger.warning("Hunter search failed for %s: %s", hypothesis.kind, e)
        return [], [
            SearchLogEntry(hypothesis_kind=hypothesis.kind, query=q, n_results=0)
            for q in queries
        ]

    evidence: List[Evidence] = []
    search_log: List[SearchLogEntry] = []
    for query, response in zip(queries, responses):
        results = response.get("results", [])
        urls = []
        for result in results:
            url = result.get("url", "")
            domain = extract_domain(url)
            urls.append(url)
            if not url or url in seen_urls or domain in exclude_domains:
                continue
            seen_urls.add(url)
            text = result.get("raw_content") or result.get("content") or ""
            if not text.strip():
                continue
            snapshot = snapshot_store.save(url, text)
            tier = registry.tier_for(domain)
            published = _parse_published(result)
            age_days = (
                (datetime.now(timezone.utc).date() - published).days
                if published
                else None
            )
            evidence.append(
                Evidence(
                    claim_id=claim_id,
                    quote=text[:_MAX_CANDIDATE_CHARS],
                    locator=Locator(url=url, snapshot_id=snapshot.id),
                    source_domain=domain,
                    source_tier=tier,
                    evidence_type=_guess_evidence_type(url, tier),
                    published_at=published,
                    age_days=float(age_days) if age_days is not None else None,
                    hypothesis_kind=hypothesis.kind,
                    found_via_query=query,
                )
            )
        search_log.append(
            SearchLogEntry(
                hypothesis_kind=hypothesis.kind,
                query=query,
                n_results=len(results),
                urls=urls,
            )
        )
        if run_store:
            run_store.trace(
                node="hunter",
                kind="search",
                payload={
                    "claim_id": claim_id,
                    "hypothesis": hypothesis.kind,
                    "query": query,
                    "n_results": len(results),
                },
            )
    return evidence, search_log
