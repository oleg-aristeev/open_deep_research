"""Pydantic data model for the verification layer.

Mirrors the schemas in deep_research_design_doc.md §4.3 and the Postgres DDL
in deep_research_mvp_plan.md §5. These objects are the single contract between
explore, verify and compose phases, and are what gets serialized into
``claim_graph.json``.
"""

import uuid
from datetime import date, datetime, timezone
from typing import List, Literal

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def now_utc() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


###################
# Sources & snapshots
###################

SourceTier = Literal["A", "B", "C", "D"]


class Source(BaseModel):
    """A web domain with a credibility tier (A=primary/regulators .. D=UGC/unknown)."""

    id: str = Field(default_factory=lambda: _new_id("src"))
    domain: str
    tier: SourceTier = "D"
    tier_basis: List[str] = Field(default_factory=list)
    notes: str | None = None


class SnapshotRef(BaseModel):
    """Reference to a stored page snapshot (content-addressed)."""

    id: str
    url: str
    fetched_at: datetime
    content_sha: str
    text_len: int


class Locator(BaseModel):
    """Pointer to a verbatim quote inside a snapshot."""

    url: str
    snapshot_id: str
    char_start: int = -1
    char_end: int = -1


###################
# Findings (explore phase output)
###################


class Finding(BaseModel):
    """An atomic observation extracted from a page, with a verbatim quote locator.

    Invariant enforced by the note-taker: ``verbatim_quote`` must be an exact
    substring of the snapshot text (checked in code, not trusted from the LLM).
    """

    id: str = Field(default_factory=lambda: _new_id("fnd"))
    text: str
    verbatim_quote: str
    snapshot_id: str
    url: str
    char_start: int = -1
    char_end: int = -1
    published_at: date | None = None
    subquestion_id: str | None = None
    quote_verified: bool = False


class RawHit(BaseModel):
    """A raw search engine hit captured before lossy compression."""

    url: str
    title: str = ""
    snippet: str = ""
    query: str = ""
    raw_content_present: bool = False
    snapshot_id: str | None = None


###################
# Claims
###################

ClaimType = Literal[
    "factual", "statistical", "causal", "trend", "predictive", "normative", "definitional"
]

ClaimStatus = Literal[
    "pending",
    "verified",
    "disputed",
    "unverified",
    "stale",
    "not_enough_evidence",
    "extraction_unstable",
]


class ClaimScope(BaseModel):
    """Explicit scope of a claim (time/geo/population/conditions)."""

    time: str | None = None
    geo: str | None = None
    population: str | None = None
    conditions: List[str] = Field(default_factory=list)


class Quantity(BaseModel):
    """Numeric payload of a statistical claim."""

    value: float | None = None
    unit: str | None = None
    comparison_base: str | None = None


class Sigma(BaseModel):
    """Aggregated confidence: 4-component, per design doc §5.1."""

    p_true: float = 0.5
    ci: List[float] = Field(default_factory=lambda: [0.0, 1.0])
    evidence_grade: Literal["A", "B", "C", "D"] = "D"
    agreement: Literal["high", "medium", "low"] = "medium"
    ignorance_u: float = 1.0
    verbal: str = "chances about even"
    explanation: str = ""


###################
# Skeptic protocol
###################

HypothesisKind = Literal["NEGATION", "OUTDATED", "SCOPE", "NUMERIC", "CIRCULAR"]


class Hypothesis(BaseModel):
    """One adversarial hypothesis with search queries designed by a critic."""

    kind: HypothesisKind
    statement: str
    queries: List[str] = Field(default_factory=list, min_length=1)
    expected_signal: str = ""


class SearchLogEntry(BaseModel):
    """One executed search within a verification protocol (part of the Phi log)."""

    hypothesis_kind: str
    query: str
    n_results: int = 0
    urls: List[str] = Field(default_factory=list)
    executed_at: datetime = Field(default_factory=now_utc)


class Protocol(BaseModel):
    """Phi: the full record of how counter-evidence was sought for one claim."""

    id: str = Field(default_factory=lambda: _new_id("prt"))
    claim_id: str = ""
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    searches: List[SearchLogEntry] = Field(default_factory=list)
    coverage: float = 0.0  # share of planned hypotheses actually executed


###################
# Evidence
###################

Stance = Literal["supports", "refutes", "qualifies", "mentions"]
EvidenceType = Literal["primary", "secondary", "opinion", "dataset"]


class Evidence(BaseModel):
    """A quote from a snapshot with a stance towards a claim."""

    id: str = Field(default_factory=lambda: _new_id("ev"))
    claim_id: str = ""
    stance: Stance = "mentions"
    stance_score: float = 0.5  # P(claim is true | this evidence) in [0,1]
    quote: str = ""
    locator: Locator | None = None
    source_domain: str = ""
    source_tier: SourceTier = "D"
    evidence_type: EvidenceType = "secondary"
    published_at: date | None = None
    age_days: float | None = None
    origin_cluster: str | None = None
    hypothesis_kind: str | None = None
    found_via_query: str | None = None

    @property
    def signed_score(self) -> float:
        """Map stance + score into [-1, 1]: positive supports, negative refutes."""
        if self.stance == "supports":
            return self.stance_score
        if self.stance == "refutes":
            return -self.stance_score
        if self.stance == "qualifies":
            return -0.3 * self.stance_score
        return 0.0


class Claim(BaseModel):
    """An atomic, decontextualized, checkable proposition."""

    id: str = Field(default_factory=lambda: _new_id("clm"))
    text: str
    type: ClaimType = "factual"
    scope: ClaimScope = Field(default_factory=ClaimScope)
    entities: List[str] = Field(default_factory=list)
    quantity: Quantity | None = None
    decision_relevance: float = 0.5
    derived_from: List[str] = Field(default_factory=list)  # finding ids
    origin_domains: List[str] = Field(default_factory=list)
    ambiguity_note: str | None = None
    status: ClaimStatus = "pending"
    sigma: Sigma | None = None
    flags: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    protocol: Protocol | None = None

    @property
    def evidence_for(self) -> List[Evidence]:
        """Evidence with a supporting stance."""
        return [e for e in self.evidence if e.stance == "supports"]

    @property
    def evidence_against(self) -> List[Evidence]:
        """Evidence refuting or qualifying the claim."""
        return [e for e in self.evidence if e.stance in ("refutes", "qualifies")]


###################
# Budget
###################


class Budget(BaseModel):
    """Hard budgets for the verify phase (graceful degradation, never crash)."""

    max_claims: int = 12
    max_hypotheses_per_claim: int = 5
    queries_per_hypothesis: int = 2
    results_per_query: int = 3
    max_stance_checks_per_claim: int = 10
    max_findings_pages: int = 25
    max_cost_usd: float = 6.0
    max_minutes: float = 20.0


###################
# Report
###################


class LintViolation(BaseModel):
    """One violation of the report contract (invariant I1 / drift)."""

    kind: Literal["unmarked_sentence", "unknown_claim_id", "unverified_claim_ref", "drift"]
    sentence: str
    claim_id: str | None = None
    detail: str = ""


class VerifiedReport(BaseModel):
    """Final output bundle: markdown + machine-readable claim graph."""

    run_id: str
    question: str
    markdown: str
    claims: List[Claim]
    unverified_observations: List[Claim] = Field(default_factory=list)
    lint_violations: List[LintViolation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)

    def claim_graph(self) -> dict:
        """Serialize the claim graph for ``claim_graph.json``."""
        return {
            "run_id": self.run_id,
            "question": self.question,
            "created_at": self.created_at.isoformat(),
            "claims": [c.model_dump(mode="json") for c in self.claims],
            "unverified_observations": [
                c.model_dump(mode="json") for c in self.unverified_observations
            ],
            "lint_violations": [v.model_dump(mode="json") for v in self.lint_violations],
        }


###################
# Structured-output helper schemas (LLM responses)
###################


class ExtractedFinding(BaseModel):
    """Note-taker structured output for one finding."""

    text: str = Field(description="Atomic factual observation, self-contained.")
    verbatim_quote: str = Field(
        description="EXACT substring copied character-for-character from the page text."
    )


class NoteTakerResult(BaseModel):
    """Note-taker structured output for one page."""

    findings: List[ExtractedFinding] = Field(default_factory=list)


class ExtractedClaim(BaseModel):
    """Claimify structured output for one claim."""

    text: str
    type: ClaimType = "factual"
    time: str | None = None
    geo: str | None = None
    population: str | None = None
    conditions: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    quantity_value: float | None = None
    quantity_unit: str | None = None
    comparison_base: str | None = None
    decision_relevance: float = Field(
        default=0.5, description="0..1: how much the truth of this claim changes the answer."
    )
    derived_from_finding_ids: List[str] = Field(default_factory=list)
    ambiguity_note: str | None = None


class ClaimifyResult(BaseModel):
    """Claimify structured output for a batch of findings."""

    claims: List[ExtractedClaim] = Field(default_factory=list)


class SkepticProtocol(BaseModel):
    """Skeptic structured output: the hypothesis protocol for one claim."""

    hypotheses: List[Hypothesis] = Field(default_factory=list)


class StanceResult(BaseModel):
    """Stance classifier structured output for one (claim, evidence) pair."""

    stance: Stance
    score: float = Field(description="Confidence of the stance label, 0..1.")
    relevant_quote: str = Field(
        default="", description="The minimal quote from the evidence that carries the stance."
    )
