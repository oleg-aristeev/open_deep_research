"""Streamlit annotation form for claim-level labeling (MVP plan §6.3).

Shows claim + evidence quote + highlighted snapshot; collects three labels per
pair (stance, verdict correctness, scope correctness) into labels.csv.

Run::

    streamlit run evals/annotate/app.py -- --run-dir .verify_store/runs/<run_id>
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

LABELS_CSV = Path(__file__).parent / "labels.csv"
FIELDNAMES = [
    "ts", "annotator", "run_id", "claim_id", "evidence_id",
    "stance_label", "verdict_ok", "scope_ok", "comment",
]


def parse_args() -> argparse.Namespace:
    """Parse args passed after `--` by streamlit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def load_pairs(run_dir: Path) -> list[dict]:
    """Flatten claim_graph.json into (claim, evidence) pairs."""
    graph = json.loads((run_dir / "claim_graph.json").read_text(encoding="utf-8"))
    pairs = []
    for claim in graph.get("claims", []):
        for ev in claim.get("evidence", []):
            pairs.append({"claim": claim, "evidence": ev, "run_id": graph.get("run_id", "")})
    return pairs


def snapshot_highlight(snapshot_id: str, start: int, end: int) -> str:
    """Return snapshot text with the quote region marked."""
    from store.snapshots import get_snapshot_store

    text = get_snapshot_store().get_text(snapshot_id) or ""
    if not text or start < 0:
        return text[:3000]
    left = max(0, start - 600)
    return (
        text[left:start] + "\n\n>>> QUOTE >>>\n" + text[start:end] + "\n<<< QUOTE <<<\n\n"
        + text[end : end + 600]
    )


def append_label(row: dict) -> None:
    """Append one label row to labels.csv."""
    exists = LABELS_CSV.exists()
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Streamlit app body."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    st.set_page_config(layout="wide", page_title="Claim annotation")
    st.title("Разметка клеймов")
    st.caption(f"Run: {run_dir}")

    annotator = st.sidebar.text_input("Аннотатор (A/B)", value="A")
    pairs = load_pairs(run_dir)
    if not pairs:
        st.warning("В claim_graph.json нет evidence-пар.")
        return
    idx = st.sidebar.number_input("Пара #", 0, len(pairs) - 1, 0)
    pair = pairs[int(idx)]
    claim, evidence = pair["claim"], pair["evidence"]

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Клейм")
        st.markdown(f"**{claim['text']}**")
        st.json({k: claim.get(k) for k in ("id", "type", "status", "flags")})
        if claim.get("sigma"):
            st.json(claim["sigma"])
        st.subheader("Цитата-свидетельство")
        st.markdown(f"> {evidence.get('quote', '')}")
        st.caption(
            f"{evidence.get('source_domain')} · stance системы: {evidence.get('stance')} "
            f"({evidence.get('stance_score')}) · гипотеза: {evidence.get('hypothesis_kind')}"
        )
    with col_right:
        st.subheader("Снапшот (контекст цитаты)")
        locator = evidence.get("locator") or {}
        st.text_area(
            "snapshot",
            snapshot_highlight(
                locator.get("snapshot_id", ""),
                locator.get("char_start", -1),
                locator.get("char_end", -1),
            ),
            height=420,
        )

    st.divider()
    stance_label = st.radio(
        "1. Поддерживает ли цитата клейм?",
        ["supports", "refutes", "qualifies", "mentions", "quote_not_found"],
        horizontal=True,
    )
    verdict_ok = st.radio("2. Вердикт системы верен?", ["да", "частично", "нет"], horizontal=True)
    scope_ok = st.radio("3. Scope клейма верен?", ["да", "нет"], horizontal=True)
    comment = st.text_input("Комментарий")

    if st.button("Сохранить разметку", type="primary"):
        append_label(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "annotator": annotator,
                "run_id": pair["run_id"],
                "claim_id": claim["id"],
                "evidence_id": evidence["id"],
                "stance_label": stance_label,
                "verdict_ok": verdict_ok,
                "scope_ok": scope_ok,
                "comment": comment,
            }
        )
        st.success(f"Сохранено в {LABELS_CSV}")


main()
