"""Run-scoped store: raw search hits, traces of LLM/search calls, artifacts.

Every LLM and search call is logged as ``(run_id, node, kind, payload, cost,
latency)``. Without this neither evals nor incident analysis are possible
(MVP plan §4.1). Mirrors to Langfuse/Postgres are additive and optional.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from verify.schema import RawHit

DEFAULT_STORE_DIR = os.environ.get("VERIFY_STORE_DIR", ".verify_store")


class RunStore:
    """Filesystem store for one research run (traces, raw hits, artifacts)."""

    def __init__(self, run_id: str, root: str | None = None):
        """Open (or create) the directory for ``run_id``."""
        self.run_id = run_id
        self.root = Path(root or DEFAULT_STORE_DIR) / "runs" / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---- traces ----

    def trace(
        self,
        node: str,
        kind: str,
        payload: Any,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
    ) -> None:
        """Append one trace record (LLM call, search call, decision)."""
        record = {
            "run_id": self.run_id,
            "node": node,
            "kind": kind,
            "payload": payload,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._append("traces.jsonl", record)

    # ---- raw search hits (the tee before lossy compression) ----

    def record_raw_hits(self, hits: List[RawHit]) -> None:
        """Persist raw search results captured before compression."""
        for hit in hits:
            self._append("raw_hits.jsonl", hit.model_dump(mode="json"))

    def load_raw_hits(self) -> List[RawHit]:
        """Load all raw hits recorded for this run."""
        return [RawHit(**rec) for rec in self._read_jsonl("raw_hits.jsonl")]

    # ---- artifacts ----

    def save_json(self, name: str, data: Any) -> Path:
        """Write a JSON artifact (e.g. claim_graph.json) into the run dir."""
        path = self.root / name
        with self._lock:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        return path

    def save_text(self, name: str, text: str) -> Path:
        """Write a text artifact (e.g. report.md) into the run dir."""
        path = self.root / name
        with self._lock:
            path.write_text(text, encoding="utf-8")
        return path

    def load_json(self, name: str) -> Any | None:
        """Read a JSON artifact, or None if missing."""
        path = self.root / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ---- internals ----

    def _append(self, name: str, record: dict) -> None:
        with self._lock:
            with open(self.root / name, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _read_jsonl(self, name: str) -> List[dict]:
        path = self.root / name
        if not path.exists():
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def total_cost(self) -> float:
        """Sum of cost_usd over all trace records."""
        return sum(r.get("cost_usd") or 0.0 for r in self._read_jsonl("traces.jsonl"))


def run_id_from_config(config: dict | None) -> str:
    """Derive the run id shared by the explore tee and the verify phase.

    Priority: explicit ``configurable.verify_run_id`` (set by eval harness /
    ExploreBackend) -> LangGraph ``thread_id`` -> 'adhoc' fallback.
    """
    configurable = (config or {}).get("configurable", {}) or {}
    return str(configurable.get("verify_run_id") or configurable.get("thread_id") or "adhoc")


_run_stores: dict[str, RunStore] = {}
_lock = threading.Lock()


def get_run_store(run_id: str) -> RunStore:
    """Return (and cache) the RunStore for ``run_id``."""
    with _lock:
        if run_id not in _run_stores:
            _run_stores[run_id] = RunStore(run_id)
        return _run_stores[run_id]
