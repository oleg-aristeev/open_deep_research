"""Baselines for blind comparisons (MVP plan §6.2).

Two baselines:
* ``odr_baseline`` — the fork with verification disabled ("голый" ODR);
  available directly in run_trapset.py via ``--variants verified,odr_baseline``.
* ``gpt_researcher`` — GPT Researcher as a library (donor project, not forked).

Usage::

    python evals/baselines.py --backend gpt_researcher --question "..."
"""

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


async def run_gpt_researcher(question: str) -> str:
    """Plain GPT Researcher report (requires `pip install gpt-researcher`)."""
    try:
        from gpt_researcher import GPTResearcher
    except ImportError:
        raise SystemExit("Install the baseline first: pip install gpt-researcher")
    researcher = GPTResearcher(query=question, report_type="research_report")
    await researcher.conduct_research()
    return await researcher.write_report()


async def run_odr_baseline(question: str) -> str:
    """Fork with verification off: upstream ODR behavior."""
    from evals.run_trapset import run_system

    run_id = f"baseline_{uuid.uuid4().hex[:8]}"
    outcome = await run_system(question, variant="odr_baseline", run_id=run_id)
    return outcome["report"]


async def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["odr_baseline", "gpt_researcher"], default="odr_baseline")
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.backend == "gpt_researcher":
        report = await run_gpt_researcher(args.question)
    else:
        report = await run_odr_baseline(args.question)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Saved to {args.out}")  # noqa: T201
    else:
        print(report)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
