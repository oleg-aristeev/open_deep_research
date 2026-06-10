"""Persistence layer: page snapshots, raw search hits and LLM/search traces.

Default backend is the local filesystem (``VERIFY_STORE_DIR``, default
``./.verify_store``) so the pipeline runs with zero infrastructure. The
Postgres schema from the MVP plan §5 lives in ``db.py`` / ``deploy/sql`` and
can be enabled later without touching callers.
"""

from store.snapshots import SnapshotStore, get_snapshot_store
from store.traces import RunStore, get_run_store

__all__ = ["SnapshotStore", "get_snapshot_store", "RunStore", "get_run_store"]
