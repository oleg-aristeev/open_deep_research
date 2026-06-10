"""Offline unit tests for the verify layer (no network, no LLM calls).

Run with: uv run pytest tests/test_verify_offline.py -q
"""

import asyncio
import os
import tempfile

import pytest

os.environ.setdefault("VERIFY_STORE_DIR", tempfile.mkdtemp(prefix="verify_test_"))

from store.snapshots import SnapshotStore
from store.traces import RunStore, run_id_from_config
from verify.aggregate import aggregate, icd203, status_from, verdict_from
from verify.compose import assemble_report, render_claims_table, render_phi_summary
from verify.config import get_settings
from verify.linter import lint_report, split_sentences
from verify.schema import Claim, Evidence, Hypothesis, Protocol
from verify.sources import (
    cluster_origins,
    extract_attributed_origin,
    extract_domain,
    get_registry,
)


def make_claim(**kwargs) -> Claim:
    claim = Claim(text=kwargs.pop("text", "The market grew 25% in 2025."), **kwargs)
    claim.protocol = Protocol(
        claim_id=claim.id,
        hypotheses=[Hypothesis(kind="NEGATION", statement="x", queries=["q"])],
        coverage=1.0,
    )
    return claim


class TestConfig:
    def test_loads_mvp_yaml(self):
        settings = get_settings()
        assert settings.budget.max_claims == 12
        assert settings.aggregate.lambda_ == 1.5
        assert settings.thresholds.max_repairs == 2


class TestSourceRegistry:
    def test_tiers(self):
        registry = get_registry()
        assert registry.tier_for("https://www.sec.gov/filings/x") == "A"
        assert registry.tier_for("https://pubmed.ncbi.nlm.nih.gov/123") == "A"
        assert registry.tier_for("https://reuters.com/a") == "B"
        assert registry.tier_for("https://techcrunch.com/a") == "C"
        assert registry.tier_for("random-unknown-blog.io") == "D"

    def test_extract_domain(self):
        assert extract_domain("https://www.example.com/path?q=1") == "example.com"
        assert extract_domain("EXAMPLE.com") == "example.com"


class TestOriginClustering:
    def test_attribution_extraction(self):
        assert extract_attributed_origin("According to Gartner, X grew.") == "gartner"
        assert extract_attributed_origin("X grew, according to Gartner research.") == "gartner"
        assert extract_attributed_origin("No attribution here.") is None

    def test_reprints_collapse(self):
        e1 = Evidence(quote="According to Gartner, the market grew 25% in 2025.", source_domain="cnbc.com")
        e2 = Evidence(quote="The market grew 25% in 2025, according to Gartner research.", source_domain="forbes.com")
        e3 = Evidence(quote="An independent survey by IDC found growth of only 12%.", source_domain="idc.com")
        cluster_origins([e1, e2, e3])
        assert e1.origin_cluster == e2.origin_cluster
        assert e3.origin_cluster != e1.origin_cluster

    def test_same_domain_is_one_origin(self):
        e1 = Evidence(quote="aaa bbb ccc ddd eee fff", source_domain="x.com")
        e2 = Evidence(quote="completely different words here entirely", source_domain="x.com")
        cluster_origins([e1, e2])
        assert e1.origin_cluster == e2.origin_cluster


class TestAggregate:
    def test_single_origin_cap_and_flag(self):
        claim = make_claim(type="statistical")
        evidence = [
            Evidence(quote="According to Gartner, the market grew 25% in 2025.",
                     source_domain="cnbc.com", stance="supports", stance_score=0.9,
                     source_tier="B", age_days=30.0),
            Evidence(quote="The market grew 25% in 2025, according to Gartner research.",
                     source_domain="forbes.com", stance="supports", stance_score=0.9,
                     source_tier="B", age_days=30.0),
        ]
        claim.evidence = cluster_origins(evidence)
        sigma, flags = aggregate(claim)
        assert "single_origin" in flags
        assert sigma.p_true <= get_settings().aggregate.single_origin_cap

    def test_disputed_flag_requires_quality_counter(self):
        claim = make_claim(type="statistical")
        evidence = [
            Evidence(quote="According to Gartner, the market grew 25% in 2025.",
                     source_domain="cnbc.com", stance="supports", stance_score=0.9,
                     source_tier="B", age_days=30.0),
            Evidence(quote="An independent survey by IDC found growth of only 12%.",
                     source_domain="idc.com", stance="refutes", stance_score=0.85,
                     source_tier="A", age_days=10.0),
        ]
        claim.evidence = cluster_origins(evidence)
        sigma, flags = aggregate(claim)
        assert "disputed" in flags
        assert sigma.agreement == "low"
        assert verdict_from(sigma, flags) == "disputed"
        assert status_from(sigma, flags) == "disputed"

    def test_no_evidence_is_nei(self):
        sigma, flags = aggregate(Claim(text="no evidence claim"))
        assert "not_enough_evidence" in flags
        assert sigma.ignorance_u == 1.0

    def test_icd203(self):
        assert icd203(0.97) == "almost certain"
        assert icd203(0.5) == "chances about even"
        assert icd203(0.02) == "almost no chance"


class TestSnapshotStore:
    def test_save_locate_dedupe(self):
        store = SnapshotStore(tempfile.mkdtemp())
        ref1 = store.save("https://x.com/p", "Hello.\nThe market grew   25% in 2025. End.")
        ref2 = store.save("https://other.com/q", "Hello.\nThe market grew   25% in 2025. End.")
        assert ref1.id == ref2.id  # content-addressed dedupe
        # exact match
        assert store.locate(ref1.id, "The market grew   25% in 2025.") is not None
        # whitespace-tolerant match
        loc = store.locate(ref1.id, "The market grew 25% in 2025.")
        assert loc is not None
        assert "market grew" in store.get_text(ref1.id)[loc[0]:loc[1]]
        # not found
        assert store.locate(ref1.id, "totally absent text") is None


class TestRunStore:
    def test_traces_and_cost(self):
        store = RunStore("test_run", tempfile.mkdtemp())
        store.trace("n", "k", {"a": 1}, cost_usd=0.5)
        store.trace("n", "k", {"a": 2}, cost_usd=0.25)
        assert store.total_cost() == 0.75

    def test_run_id_resolution(self):
        assert run_id_from_config({"configurable": {"verify_run_id": "r1", "thread_id": "t"}}) == "r1"
        assert run_id_from_config({"configurable": {"thread_id": "t1"}}) == "t1"
        assert run_id_from_config(None) == "adhoc"


class TestLinter:
    def test_invariant_i1(self):
        claim = make_claim()
        claim.status = "verified"
        md = (
            "## Выводы\n"
            f"Рынок вырос на 25% в 2025 году [{claim.id}]. "
            "Это важное наблюдение без маркера которое должно быть поймано линтером.\n"
            "Контекст развивается быстро (интерпретация). "
            "Несуществующая ссылка тут стоит [clm_deadbeef]."
        )
        violations = asyncio.run(lint_report(md, [claim]))
        kinds = sorted(v.kind for v in violations)
        assert kinds == ["unknown_claim_id", "unmarked_sentence"]

    def test_unverified_claim_ref(self):
        claim = make_claim()
        claim.status = "extraction_unstable"
        md = f"## Выводы\nФакт со ссылкой на нестабильный клейм вот тут [{claim.id}]."
        violations = asyncio.run(lint_report(md, [claim]))
        assert any(v.kind == "unverified_claim_ref" for v in violations)

    def test_split_sentences_skips_tables_and_headings(self):
        md = "# H\n| a | b |\n|---|---|\n| 1 | 2 |\nReal sentence one here. Second sentence here."
        sentences = split_sentences(md)
        assert len(sentences) == 2


class TestCompose:
    def test_renders(self):
        claim = make_claim(type="statistical")
        claim.evidence = cluster_origins([
            Evidence(quote="According to Gartner, the market grew 25% in 2025.",
                     source_domain="cnbc.com", stance="supports", stance_score=0.9,
                     source_tier="B", age_days=30.0),
            Evidence(quote="An independent survey by IDC found growth of only 12%.",
                     source_domain="idc.com", stance="refutes", stance_score=0.85,
                     source_tier="A", age_days=10.0),
        ])
        claim.sigma, claim.flags = aggregate(claim)
        claim.status = status_from(claim.sigma, claim.flags)

        table = render_claims_table([claim])
        assert claim.id in table and "disputed" in table
        phi = render_phi_summary([claim])
        assert claim.id in phi
        violations = asyncio.run(
            lint_report("## Выводы\nДлинное предложение без маркера для нарушения линта.", [claim])
        )
        report = assemble_report("## Выводы\nText (интерпретация).", [claim], violations)
        assert "Таблица утверждений" in report
        assert "Φ" in report
        assert "Не прошло линт" in report


class TestGraphCompiles:
    def test_subgraph_and_parent_compile(self):
        from verify.graph import build_verify_graph

        graph = build_verify_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert {"notetake", "claimify", "rank", "verify_one", "collect",
                "compose", "lint", "repair", "finalize"} <= node_names

    def test_parent_graph_has_verification_node(self):
        from open_deep_research.deep_researcher import deep_researcher

        assert "verification" in deep_researcher.get_graph().nodes


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
