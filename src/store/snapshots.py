"""Content-addressed snapshot store for fetched pages.

Every page touched by the pipeline is stored as ``(url, fetched_at,
content_sha, clean text[, raw html])``. Citations in the final report point at
snapshots + character offsets, never at a paraphrase — this is the foundation
the whole verify layer stands on (MVP plan §4.1).
"""

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from verify.schema import SnapshotRef

DEFAULT_STORE_DIR = os.environ.get("VERIFY_STORE_DIR", ".verify_store")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalize_ws(text: str) -> str:
    """Collapse whitespace runs; used for tolerant quote matching."""
    return re.sub(r"\s+", " ", text).strip()


class SnapshotStore:
    """Filesystem snapshot store with content-addressed deduplication."""

    def __init__(self, root: str | None = None):
        """Create the store under ``root`` (default: VERIFY_STORE_DIR)."""
        self.root = Path(root or DEFAULT_STORE_DIR) / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.jsonl"
        self._lock = threading.Lock()
        self._index: dict[str, dict] = {}
        self._load_index()

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        with open(self._index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta = json.loads(line)
                self._index[meta["id"]] = meta

    def save(self, url: str, text: str, html: str | None = None) -> SnapshotRef:
        """Store page text (and optional raw html); dedupe by content sha."""
        sha = _sha256(text)
        snapshot_id = f"snp_{sha[:12]}"
        with self._lock:
            if snapshot_id not in self._index:
                (self.root / f"{sha}.txt").write_text(text, encoding="utf-8")
                if html:
                    (self.root / f"{sha}.html").write_text(html, encoding="utf-8")
                meta = {
                    "id": snapshot_id,
                    "url": url,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "content_sha": sha,
                    "text_len": len(text),
                }
                self._index[snapshot_id] = meta
                with open(self._index_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            meta = self._index[snapshot_id]
        return SnapshotRef(
            id=meta["id"],
            url=meta["url"],
            fetched_at=datetime.fromisoformat(meta["fetched_at"]),
            content_sha=meta["content_sha"],
            text_len=meta["text_len"],
        )

    def get_meta(self, snapshot_id: str) -> dict | None:
        """Return snapshot metadata or None."""
        return self._index.get(snapshot_id)

    def get_text(self, snapshot_id: str) -> str | None:
        """Return the stored clean text of a snapshot."""
        meta = self._index.get(snapshot_id)
        if not meta:
            return None
        path = self.root / f"{meta['content_sha']}.txt"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def locate(self, snapshot_id: str, quote: str) -> Tuple[int, int] | None:
        """Find exact char offsets of ``quote`` in the snapshot text.

        Falls back to whitespace-normalized matching; returns None if the
        quote is not a substring of the snapshot (the caller must then treat
        the quote as unverified).
        """
        text = self.get_text(snapshot_id)
        if text is None or not quote:
            return None
        start = text.find(quote)
        if start >= 0:
            return start, start + len(quote)
        # Tolerant match: normalize whitespace on both sides, then map back.
        norm_quote = normalize_ws(quote)
        if not norm_quote:
            return None
        # Build a regex where any whitespace run in the quote matches any run.
        pattern = r"\s+".join(re.escape(part) for part in norm_quote.split(" "))
        m = re.search(pattern, text)
        if m:
            return m.start(), m.end()
        return None


_default_store: SnapshotStore | None = None
_store_lock = threading.Lock()


def get_snapshot_store() -> SnapshotStore:
    """Return the process-wide default snapshot store."""
    global _default_store
    with _store_lock:
        if _default_store is None:
            _default_store = SnapshotStore()
        return _default_store
