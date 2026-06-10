"""Source intelligence v0: static tier registry + origin clustering heuristics.

Origin clustering (MVP plan §4.6) collapses reprints of the same primary
source into one cluster so that 50 copies of a press release count as one
piece of evidence. Three heuristics in priority order:

1. attributed-origin extraction ("according to X", "sources told Y", press release);
2. near-duplicate text (shingled Jaccard similarity);
3. same publication date + shared numbers/entities.

Anything still unclustered falls back to per-domain clusters.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import yaml

from verify.config import REPO_ROOT, get_settings
from verify.schema import Evidence, SourceTier

logger = logging.getLogger(__name__)


###################
# Tier registry
###################


class SourceRegistry:
    """Domain -> tier lookup backed by configs/domains/tiers.yaml."""

    def __init__(self, tiers_file: str | None = None):
        """Load the tier map; unknown file -> empty map (everything is D)."""
        path = Path(tiers_file or get_settings().tiers_file)
        if not path.is_absolute():
            path = REPO_ROOT / path
        self._domain_tier: Dict[str, SourceTier] = {}
        self._unknown_logged: set[str] = set()
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for tier in ("A", "B", "C", "D"):
                for domain in data.get(tier, []) or []:
                    self._domain_tier[str(domain).lower()] = tier  # type: ignore[assignment]
        else:
            logger.warning("Tier file %s not found; all sources default to D", path)

    def tier_for(self, url_or_domain: str) -> SourceTier:
        """Return the tier for a URL/domain; unlisted domains are D (and logged once)."""
        domain = extract_domain(url_or_domain)
        # Exact match, then parent-domain match (ir.example.com -> example.com).
        parts = domain.split(".")
        for i in range(len(parts) - 1):
            candidate = ".".join(parts[i:])
            if candidate in self._domain_tier:
                return self._domain_tier[candidate]
        if domain and domain not in self._unknown_logged:
            self._unknown_logged.add(domain)
            logger.info("Unlisted source domain (tier D): %s", domain)
        return "D"


@lru_cache(maxsize=1)
def get_registry() -> SourceRegistry:
    """Process-wide tier registry."""
    return SourceRegistry()


def extract_domain(url_or_domain: str) -> str:
    """Normalize a URL or bare domain to a lowercase registrable-ish domain."""
    s = (url_or_domain or "").strip().lower()
    if "://" in s:
        s = urlparse(s).netloc
    s = s.split("/")[0].split(":")[0]
    if s.startswith("www."):
        s = s[4:]
    return s


###################
# Origin clustering
###################

_ATTRIBUTION_PATTERNS = [
    r"according to ([A-Za-z][\w&.\- ]{2,40})",
    r"sources told ([A-Za-z][\w&.\- ]{2,40})",
    r"([A-Za-z][\w&.\- ]{2,40}) (?:said in|announced in|published) (?:a|its) (?:press release|statement|report)",
    r"press release (?:from|by) ([A-Za-z][\w&.\- ]{2,40})",
    r"(?:a|the) report (?:from|by) ([A-Za-z][\w&.\- ]{2,40})",
    r"по данным ([\w&.\- ]{2,40})",
    r"со ссылкой на ([\w&.\- ]{2,40})",
]

# Generic trailing words that vary between reprints of the same origin
# ("Gartner" vs "Gartner research" vs "Gartner report").
_ORIGIN_STOP_TOKENS = {
    "research", "report", "reports", "data", "survey", "estimates", "analysis",
    "analysts", "study", "figures", "исследования", "отчёта", "данных",
}


def extract_attributed_origin(text: str) -> str | None:
    """Heuristic 1: pull the attributed origin out of quote text, if any."""
    for pattern in _ATTRIBUTION_PATTERNS:
        m = re.search(pattern, text or "", flags=re.IGNORECASE)
        if m:
            raw_tokens = re.sub(r"\s+", " ", m.group(1)).strip().lower().split(" ")
            tokens = [t.strip(".,;:!?") for t in raw_tokens]
            tokens = [t for t in tokens if t]
            while tokens and tokens[-1] in _ORIGIN_STOP_TOKENS:
                tokens.pop()
            if tokens:
                return " ".join(tokens[:3])
    return None


def _shingles(text: str, k: int = 5) -> set:
    words = re.findall(r"\w+", (text or "").lower())
    if len(words) < k:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two shingle sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _extract_numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:[.,]\d+)?%?", text or ""))


def cluster_origins(
    evidence: List[Evidence], near_dup_threshold: float = 0.6
) -> List[Evidence]:
    """Assign ``origin_cluster`` ids to evidence in place and return the list.

    Union-find over the three heuristics; remaining singletons share a cluster
    with same-domain evidence (a domain is at minimum one origin).
    """
    n = len(evidence)
    if n == 0:
        return evidence
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    # Heuristic 0 (baseline): same domain == same origin.
    by_domain: Dict[str, int] = {}
    for i, ev in enumerate(evidence):
        domain = extract_domain(ev.source_domain or (ev.locator.url if ev.locator else ""))
        if domain in by_domain:
            union(i, by_domain[domain])
        else:
            by_domain[domain] = i

    # Heuristic 1: shared attributed origin ("according to X" ...).
    by_origin: Dict[str, int] = {}
    for i, ev in enumerate(evidence):
        origin = extract_attributed_origin(ev.quote)
        if not origin:
            continue
        if origin in by_origin:
            union(i, by_origin[origin])
        else:
            by_origin[origin] = i

    # Heuristic 2: near-duplicate quote text.
    shingle_sets = [_shingles(ev.quote) for ev in evidence]
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(shingle_sets[i], shingle_sets[j]) >= near_dup_threshold:
                union(i, j)

    # Heuristic 3: same date + overlapping numbers (likely same underlying release).
    for i in range(n):
        for j in range(i + 1, n):
            ei, ej = evidence[i], evidence[j]
            if ei.published_at and ei.published_at == ej.published_at:
                nums_i, nums_j = _extract_numbers(ei.quote), _extract_numbers(ej.quote)
                if nums_i and nums_j and len(nums_i & nums_j) >= 2:
                    union(i, j)

    # Materialize cluster ids.
    root_to_cluster: Dict[int, str] = {}
    for i, ev in enumerate(evidence):
        root = find(i)
        if root not in root_to_cluster:
            root_to_cluster[root] = f"oc_{len(root_to_cluster):02d}"
        ev.origin_cluster = root_to_cluster[root]
    return evidence
