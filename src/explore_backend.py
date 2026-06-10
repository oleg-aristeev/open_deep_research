"""ExploreBackend: hedge interface over the explore phase (MVP plan §1.2).

The verify layer only needs ``question -> (brief, notes, snapshots in store)``.
If ODR's supervisor proves unstable or expensive, the backend can be swapped
for GPT Researcher in ~2 days without touching the verify loop. That insurance
costs exactly this one Protocol.
"""

import logging
import uuid
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExploreResult(BaseModel):
    """Output of the explore phase. Raw hits/snapshots land in the run store."""

    run_id: str
    brief: str = ""
    notes: list[str] = Field(default_factory=list)


@runtime_checkable
class ExploreBackend(Protocol):
    """Interface: research a question, persist raw pages, return notes."""

    async def explore(self, question: str, run_id: str | None = None, config: dict | None = None) -> ExploreResult:
        """Run the explore phase for ``question``."""
        ...


class ODRExploreBackend:
    """Default backend: ODR's brief + supervisor + researchers (no report node).

    Reuses upstream nodes as-is; the raw-results tee inside ``tavily_search``
    fills the snapshot/run stores keyed by ``verify_run_id``.
    """

    def __init__(self):
        """Lazily compile a brief->supervisor-only graph from ODR nodes."""
        from langgraph.graph import START, StateGraph

        from open_deep_research.configuration import Configuration
        from open_deep_research.deep_researcher import (
            supervisor_subgraph,
            write_research_brief,
        )
        from open_deep_research.state import AgentInputState, AgentState

        builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
        builder.add_node("write_research_brief", write_research_brief)
        builder.add_node("research_supervisor", supervisor_subgraph)
        builder.add_edge(START, "write_research_brief")
        # write_research_brief routes to research_supervisor via Command;
        # the supervisor subgraph ends the run.
        self._graph = builder.compile()

    async def explore(self, question: str, run_id: str | None = None, config: dict | None = None) -> ExploreResult:
        """Run ODR explore; notes are the supervisor's compressed researcher outputs."""
        run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
        invoke_config = {"configurable": {"verify_run_id": run_id, **((config or {}).get("configurable", {}))}}
        if config:
            invoke_config = {**config, "configurable": invoke_config["configurable"]}
        state = await self._graph.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            invoke_config,
        )
        return ExploreResult(
            run_id=run_id,
            brief=state.get("research_brief", "") or question,
            notes=list(state.get("notes", []) or []),
        )


class GPTResearcherExploreBackend:
    """Alternative backend: gpt-researcher as a library (donor, not a fork).

    Requires ``pip install gpt-researcher``. Visited pages are snapshotted into
    the same stores so the verify layer works unchanged.
    """

    async def explore(self, question: str, run_id: str | None = None, config: dict | None = None) -> ExploreResult:
        """Run GPT Researcher and tee its context/pages into the run store."""
        try:
            from gpt_researcher import GPTResearcher
        except ImportError as e:
            raise ImportError(
                "GPTResearcherExploreBackend requires 'pip install gpt-researcher'"
            ) from e

        from store.snapshots import get_snapshot_store
        from store.traces import get_run_store
        from verify.schema import RawHit

        run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
        run_store = get_run_store(run_id)
        snapshot_store = get_snapshot_store()

        researcher = GPTResearcher(query=question, report_type="research_report")
        context = await researcher.conduct_research()

        hits = []
        for item in getattr(researcher, "visited_urls", []) or []:
            hits.append(RawHit(url=str(item), query=question))
        # research context chunks become snapshots (best effort, no raw html)
        chunks = context if isinstance(context, list) else [str(context)]
        for chunk in chunks:
            text = str(chunk)
            if len(text.strip()) < 200:
                continue
            snapshot = snapshot_store.save(url="gpt-researcher://context", text=text)
            hits.append(
                RawHit(
                    url="gpt-researcher://context",
                    query=question,
                    raw_content_present=True,
                    snapshot_id=snapshot.id,
                )
            )
        run_store.record_raw_hits(hits)
        return ExploreResult(run_id=run_id, brief=question, notes=[str(c) for c in chunks])


def get_explore_backend(name: str = "odr") -> ExploreBackend:
    """Backend factory: 'odr' (default) or 'gpt_researcher'."""
    if name == "gpt_researcher":
        return GPTResearcherExploreBackend()
    return ODRExploreBackend()
