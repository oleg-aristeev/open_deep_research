"""Trap-set evaluation harness (MVP plan §6).

Runs the verified pipeline (and optional baselines) over evals/trapset/*.jsonl
and computes:

* CER (Counter-Evidence Recall) — the headline metric: share of gold
  ``must_find_counter`` items the system actually surfaced;
* verification accuracy — gold verdict vs system verdict on matched key claims;
* citation precision — ALCE-style: report sentence entailed by the cited
  claim's supporting evidence;
* false-disputed rate on control questions;
* cost / latency / verdict dispersion over repeats.

Usage::

    python evals/run_trapset.py --config configs/mvp.yaml \
        --variants verified,odr_baseline --limit 3 --repeats 1

Results land in ``evals/results/<timestamp>/`` as results.json + report.html.
"""

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

GROUNDING_MATCH_THRESHOLD = 0.5
MAX_CITATION_CHECKS = 15
CLAIM_MARKER_RE = re.compile(r"\[(clm_[0-9a-f]{8})\]")


def load_trapset(path: Path) -> list[dict]:
    """Load trap questions from a jsonl file."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def run_system(question: str, variant: str, run_id: str) -> dict:
    """Run one variant of the pipeline on one question."""
    from langgraph.checkpoint.memory import MemorySaver

    from open_deep_research.deep_researcher import deep_researcher_builder
    from store.traces import get_run_store

    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": run_id,
            "verify_run_id": run_id,
            "allow_clarification": False,
            "enable_verification": variant == "verified",
        },
        "recursion_limit": 60,
    }
    started = time.monotonic()
    state = await graph.ainvoke({"messages": [{"role": "user", "content": question}]}, config)
    latency_s = time.monotonic() - started
    run_store = get_run_store(run_id)
    claim_graph = run_store.load_json("claim_graph.json") if variant == "verified" else None
    return {
        "run_id": run_id,
        "variant": variant,
        "report": state.get("final_report", "") or "",
        "claim_graph": claim_graph,
        "latency_s": round(latency_s, 1),
        "cost_usd": run_store.total_cost(),
    }


def _report_paragraphs(report: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", report) if len(p.strip()) > 40]
    return paragraphs[:60]


async def _max_grounding(stance, statement: str, documents: list[str]) -> float:
    """Max grounding score of statement over a list of candidate documents."""
    from verify.stance import grounding_scores_batch

    if not documents:
        return 0.0
    scores = await grounding_scores_batch(stance, [(statement, d) for d in documents])
    return max(scores)


async def evaluate_outcome(trap: dict, outcome: dict, stance) -> dict:
    """Compute all metrics for one (question, system run) pair."""
    gold = trap["gold"]
    report = outcome["report"]
    claim_graph = outcome.get("claim_graph") or {}
    claims = claim_graph.get("claims", [])
    paragraphs = _report_paragraphs(report)

    # ---- CER ----
    counters_total, counters_found, counter_details = 0, 0, []
    for key_claim in gold.get("key_claims", []):
        for counter in key_claim.get("must_find_counter", []):
            counters_total += 1
            against_quotes = [
                ev.get("quote", "")
                for claim in claims
                for ev in claim.get("evidence", [])
                if ev.get("stance") in ("refutes", "qualifies") and ev.get("quote")
            ]
            score = await _max_grounding(stance, counter, against_quotes + paragraphs)
            found = score >= GROUNDING_MATCH_THRESHOLD
            counters_found += int(found)
            counter_details.append({"counter": counter, "found": found, "score": round(score, 2)})
    cer = counters_found / counters_total if counters_total else None

    # ---- verdict accuracy + false disputed (verified variant only) ----
    verdict_records, false_disputed = [], None
    if claims:
        from verify.aggregate import verdict_from
        from verify.schema import Sigma

        for key_claim in gold.get("key_claims", []):
            pattern = key_claim["text_pattern"]
            best_claim, best_score = None, 0.0
            scores = await asyncio.gather(
                *(
                    _max_grounding(stance, pattern, [c.get("text", "")])
                    for c in claims
                )
            )
            for claim, score in zip(claims, scores):
                if score > best_score:
                    best_claim, best_score = claim, score
            if best_claim is None or best_score < GROUNDING_MATCH_THRESHOLD:
                verdict_records.append(
                    {"gold": key_claim["verdict"], "system": "claim_not_extracted", "match": False}
                )
                continue
            sigma = Sigma(**best_claim["sigma"]) if best_claim.get("sigma") else Sigma()
            system_verdict = verdict_from(sigma, best_claim.get("flags", []))
            verdict_records.append(
                {
                    "gold": key_claim["verdict"],
                    "system": system_verdict,
                    "claim_id": best_claim.get("id"),
                    "match": system_verdict == key_claim["verdict"],
                }
            )
        if trap.get("category") == "control":
            false_disputed = any("disputed" in c.get("flags", []) for c in claims)

    # ---- citation precision (verified variant only) ----
    citation_precision = None
    if claims:
        from verify.linter import split_sentences

        claims_by_id = {c["id"]: c for c in claims}
        pairs = []
        for sentence in split_sentences(report):
            ids = CLAIM_MARKER_RE.findall(sentence)
            if not ids:
                continue
            bare = CLAIM_MARKER_RE.sub("", sentence).strip()
            docs = []
            for cid in ids:
                claim = claims_by_id.get(cid)
                if not claim:
                    continue
                quotes = [
                    ev.get("quote", "")
                    for ev in claim.get("evidence", [])
                    if ev.get("stance") == "supports"
                ][:3]
                docs.append(" ".join(quotes) or claim.get("text", ""))
            if docs:
                pairs.append((bare, " ".join(docs)))
            if len(pairs) >= MAX_CITATION_CHECKS:
                break
        if pairs:
            from verify.stance import grounding_scores_batch

            scores = await grounding_scores_batch(stance, pairs)
            citation_precision = sum(s >= GROUNDING_MATCH_THRESHOLD for s in scores) / len(scores)

    return {
        "trap_id": trap["id"],
        "domain": trap["domain"],
        "category": trap["category"],
        "variant": outcome["variant"],
        "run_id": outcome["run_id"],
        "cer": cer,
        "counter_details": counter_details,
        "verdicts": verdict_records,
        "verification_accuracy": (
            sum(r["match"] for r in verdict_records) / len(verdict_records)
            if verdict_records
            else None
        ),
        "citation_precision": citation_precision,
        "false_disputed": false_disputed,
        "latency_s": outcome["latency_s"],
        "cost_usd": outcome["cost_usd"],
    }


def aggregate_results(records: list[dict]) -> dict:
    """Aggregate per-run records into per-variant summary metrics."""
    summary: dict[str, dict] = {}
    by_variant = defaultdict(list)
    for record in records:
        by_variant[record["variant"]].append(record)
    for variant, recs in by_variant.items():
        def _mean(key):
            values = [r[key] for r in recs if r.get(key) is not None]
            return round(sum(values) / len(values), 3) if values else None

        controls = [r for r in recs if r["category"] == "control" and r["false_disputed"] is not None]
        summary[variant] = {
            "n_runs": len(recs),
            "CER": _mean("cer"),
            "verification_accuracy": _mean("verification_accuracy"),
            "citation_precision": _mean("citation_precision"),
            "false_disputed_rate": (
                round(sum(r["false_disputed"] for r in controls) / len(controls), 3)
                if controls
                else None
            ),
            "mean_latency_s": _mean("latency_s"),
            "mean_cost_usd": _mean("cost_usd"),
        }
    return summary


def verdict_dispersion(records: list[dict]) -> dict:
    """Share of gold key claims whose system verdict differs across repeats."""
    grouped = defaultdict(list)
    for record in records:
        if record["variant"] != "verified":
            continue
        for i, verdict in enumerate(record.get("verdicts", [])):
            grouped[(record["trap_id"], i)].append(verdict["system"])
    diverging = sum(1 for verdicts in grouped.values() if len(set(verdicts)) > 1)
    return {
        "n_key_claims": len(grouped),
        "diverging": diverging,
        "dispersion": round(diverging / len(grouped), 3) if grouped else None,
    }


def render_html(summary: dict, records: list[dict], dispersion: dict) -> str:
    """One-page HTML comparison of variants."""
    rows = "".join(
        f"<tr><td>{v}</td><td>{m['n_runs']}</td><td>{m['CER']}</td>"
        f"<td>{m['verification_accuracy']}</td><td>{m['citation_precision']}</td>"
        f"<td>{m['false_disputed_rate']}</td><td>{m['mean_latency_s']}</td>"
        f"<td>{m['mean_cost_usd']}</td></tr>"
        for v, m in summary.items()
    )
    detail_rows = "".join(
        f"<tr><td>{r['trap_id']}</td><td>{r['category']}</td><td>{r['variant']}</td>"
        f"<td>{r['cer']}</td><td>{r['verification_accuracy']}</td>"
        f"<td>{r['citation_precision']}</td><td>{r['latency_s']}</td></tr>"
        for r in records
    )
    return f"""<html><head><meta charset="utf-8"><title>Trap set results</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 10px}}</style></head><body>
<h1>Trap set evaluation</h1>
<h2>Summary (gate §9: CER ≥ 0.6, citation precision ≥ 0.85, accuracy ≥ 0.75, false-disputed ≤ 0.1)</h2>
<table><tr><th>variant</th><th>runs</th><th>CER</th><th>verif. accuracy</th>
<th>citation precision</th><th>false disputed</th><th>latency s</th><th>cost $</th></tr>{rows}</table>
<h2>Verdict dispersion over repeats</h2><p>{dispersion}</p>
<h2>Per-run details</h2>
<table><tr><th>trap</th><th>category</th><th>variant</th><th>CER</th><th>accuracy</th>
<th>cit. precision</th><th>latency</th></tr>{detail_rows}</table>
</body></html>"""


async def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trapset", default=str(REPO_ROOT / "evals" / "trapset" / "v0.jsonl"))
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "mvp.yaml"))
    parser.add_argument("--variants", default="verified", help="comma-separated: verified,odr_baseline")
    parser.add_argument("--limit", type=int, default=0, help="run only first N questions")
    parser.add_argument("--ids", default="", help="comma-separated trap ids to run")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    import os

    os.environ.setdefault("VERIFY_CONFIG", args.config)

    from verify.config import get_settings
    from verify.stance import get_stance_classifier

    stance = get_stance_classifier(get_settings())

    traps = load_trapset(Path(args.trapset))
    if args.ids:
        wanted = set(args.ids.split(","))
        traps = [t for t in traps if t["id"] in wanted]
    if args.limit:
        traps = traps[: args.limit]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    records = []
    for trap in traps:
        for variant in variants:
            for repeat in range(args.repeats):
                run_id = f"{trap['id']}__{variant}__{repeat}__{uuid.uuid4().hex[:6]}"
                print(f"[run] {run_id}: {trap['question'][:80]}")  # noqa: T201
                try:
                    outcome = await run_system(trap["question"], variant, run_id)
                    record = await evaluate_outcome(trap, outcome, stance)
                except Exception as e:  # one failed run must not kill the sweep
                    print(f"[fail] {run_id}: {e}")  # noqa: T201
                    record = {
                        "trap_id": trap["id"], "category": trap["category"],
                        "variant": variant, "run_id": run_id, "error": str(e),
                        "cer": None, "verification_accuracy": None,
                        "citation_precision": None, "false_disputed": None,
                        "latency_s": None, "cost_usd": None, "verdicts": [],
                    }
                records.append(record)

    summary = aggregate_results(records)
    dispersion = verdict_dispersion(records) if args.repeats > 1 else {}

    out_dir = REPO_ROOT / "evals" / "results" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {"summary": summary, "dispersion": dispersion, "records": records},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "report.html").write_text(render_html(summary, records, dispersion), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # noqa: T201
    print(f"Results: {out_dir}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
