"""Confidence aggregator v1: log-odds with anti-amplification safeguards.

Implements MVP plan §4.7. Key properties:
* evidence is grouped by origin cluster; tanh saturation inside a cluster
  means 50 reprints of one press release count as ~one piece of evidence;
* hard cap p <= single_origin_cap when everything traces to one origin;
* ignorance ``u`` separates "we barely searched" from "sources disagree";
* CI via bootstrap resampling of the evidence set.

All constants live in configs/mvp.yaml and get tuned on the trap set.
"""

import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from verify.config import AggregateSettings, VerifySettings, get_settings
from verify.schema import Claim, Evidence, Sigma


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _cluster_contributions(
    evidence: List[Evidence], cfg: AggregateSettings, tau_days: float
) -> Dict[str, float]:
    clusters: Dict[str, List[Evidence]] = defaultdict(list)
    for ev in evidence:
        clusters[ev.origin_cluster or f"oc_solo_{ev.id}"].append(ev)
    contribs: Dict[str, float] = {}
    for cluster_id, ev_list in clusters.items():
        x = 0.0
        for ev in ev_list:
            freshness = math.exp(-(ev.age_days or 0.0) / tau_days)
            x += (
                ev.signed_score
                * cfg.tier_w.get(ev.source_tier, 0.15)
                * cfg.type_w.get(ev.evidence_type, 0.7)
                * freshness
            )
        contribs[cluster_id] = math.tanh(x / cfg.lambda_)
    return contribs


def _p_from_evidence(
    evidence: List[Evidence], claim_type: str, cfg: AggregateSettings, tau_days: float
) -> Tuple[float, Dict[str, float]]:
    contribs = _cluster_contributions(evidence, cfg, tau_days)
    L = cfg.prior.get(claim_type, 0.0) + cfg.beta * sum(contribs.values())
    p = _sigmoid(L)
    if len(contribs) == 1:
        p = min(p, cfg.single_origin_cap)
    return p, contribs


def _ignorance(n_clusters: int, coverage: float) -> float:
    """'We searched little' is not 'sources disagree' (design doc §5.4).

    Decays with the number of independent origin clusters and with skeptic
    protocol coverage (share of planned hypotheses actually executed).
    """
    cluster_term = math.exp(-(max(n_clusters, 0) - 1) / 1.5) if n_clusters > 0 else 1.0
    return min(1.0, 0.5 * cluster_term + 0.5 * (1.0 - max(0.0, min(1.0, coverage))))


def _bootstrap_ci(
    evidence: List[Evidence],
    claim_type: str,
    cfg: AggregateSettings,
    tau_days: float,
    seed: int = 17,
) -> List[float]:
    if not evidence:
        return [0.0, 1.0]
    rng = random.Random(seed)
    samples = []
    for _ in range(cfg.bootstrap_iterations):
        resampled = [evidence[rng.randrange(len(evidence))] for _ in range(len(evidence))]
        p, _ = _p_from_evidence(resampled, claim_type, cfg, tau_days)
        samples.append(p)
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[min(int(0.975 * len(samples)), len(samples) - 1)]
    return [round(lo, 3), round(hi, 3)]


def _grade(contribs: Dict[str, float], evidence: List[Evidence]) -> str:
    """GRADE-like strength of the evidence base (not of the verdict)."""
    meaningful = {cid for cid, v in contribs.items() if abs(v) > 0.05}
    tiers_by_cluster: Dict[str, set] = defaultdict(set)
    for ev in evidence:
        if ev.origin_cluster in meaningful:
            tiers_by_cluster[ev.origin_cluster].add(ev.source_tier)
    strong = sum(1 for tiers in tiers_by_cluster.values() if tiers & {"A"})
    good = sum(1 for tiers in tiers_by_cluster.values() if tiers & {"A", "B"})
    if strong >= 1 and len(meaningful) >= 2:
        return "A"
    if good >= 2:
        return "B"
    if len(meaningful) >= 1 and good >= 1:
        return "C"
    return "D"


_ICD203 = [
    (0.05, "almost no chance"),
    (0.20, "very unlikely"),
    (0.45, "unlikely"),
    (0.55, "chances about even"),
    (0.80, "likely"),
    (0.95, "very likely"),
    (1.01, "almost certain"),
]


def icd203(p: float) -> str:
    """Map probability to the ICD-203 verbal scale."""
    for threshold, verbal in _ICD203:
        if p < threshold:
            return verbal
    return "almost certain"


def aggregate(
    claim: Claim,
    settings: VerifySettings = None,
    tau_days: float = None,
) -> Tuple[Sigma, List[str]]:
    """Compute sigma and threshold flags for a claim with attached evidence."""
    settings = settings or get_settings()
    cfg = settings.aggregate
    tau = tau_days or cfg.tau_days
    evidence = claim.evidence
    coverage = claim.protocol.coverage if claim.protocol else 0.0

    if not evidence:
        sigma = Sigma(
            p_true=_sigmoid(cfg.prior.get(claim.type, 0.0)),
            ci=[0.0, 1.0],
            evidence_grade="D",
            agreement="medium",
            ignorance_u=1.0,
            verbal="not enough evidence",
            explanation="No evidence retrieved.",
        )
        return sigma, ["not_enough_evidence"]

    p, contribs = _p_from_evidence(evidence, claim.type, cfg, tau)
    n_clusters = len(contribs)
    pos = [v for v in contribs.values() if v > cfg.contrib_threshold]
    neg = [v for v in contribs.values() if v < -cfg.contrib_threshold]
    if pos and neg:
        agreement = "low"
    elif len(pos) + len(neg) >= 3:
        agreement = "high"
    else:
        agreement = "medium"
    u = _ignorance(n_clusters, coverage)
    ci = _bootstrap_ci(evidence, claim.type, cfg, tau)
    grade = _grade(contribs, evidence)

    # ---- flags (threshold rules, MVP plan §4.7) ----
    flags: List[str] = []
    # disputed: a negative cluster backed by tier>=B evidence while positive clusters exist.
    neg_cluster_ids = {cid for cid, v in contribs.items() if v < -cfg.contrib_threshold}
    neg_quality = any(
        ev.origin_cluster in neg_cluster_ids and ev.source_tier in ("A", "B")
        for ev in evidence
    )
    if pos and neg and neg_quality:
        flags.append("disputed")
    if n_clusters == 1:
        flags.append("single_origin")
    pro_ages = [ev.age_days for ev in evidence if ev.stance == "supports" and ev.age_days is not None]
    outdated_found_fresh = any(
        ev.hypothesis_kind == "OUTDATED" and (ev.age_days or 1e9) < min(pro_ages or [1e9])
        for ev in evidence
    )
    if pro_ages and min(pro_ages) > cfg.stale_age_days and not outdated_found_fresh:
        flags.append("stale")
    if u > cfg.not_enough_evidence_u:
        flags.append("not_enough_evidence")

    explanation = (
        f"{n_clusters} origin cluster(s): {len(pos)} supporting, {len(neg)} opposing "
        f"above threshold; protocol coverage {coverage:.0%}; grade {grade}."
    )
    sigma = Sigma(
        p_true=round(p, 3),
        ci=ci,
        evidence_grade=grade,
        agreement=agreement,
        ignorance_u=round(u, 3),
        verbal=icd203(p),
        explanation=explanation,
    )
    return sigma, flags


def status_from(sigma: Sigma, flags: List[str]) -> str:
    """Derive the claim lifecycle status from sigma + flags."""
    if "disputed" in flags:
        return "disputed"
    if "not_enough_evidence" in flags:
        return "not_enough_evidence"
    if "stale" in flags:
        return "stale"
    return "verified"


def verdict_from(sigma: Sigma, flags: List[str]) -> str:
    """Human-facing verdict used in the report table."""
    if "disputed" in flags:
        return "disputed"
    if "not_enough_evidence" in flags:
        return "not_enough_evidence"
    if sigma.p_true >= 0.55:
        return "supported"
    if sigma.p_true <= 0.45:
        return "refuted"
    return "uncertain"
