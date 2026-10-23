"""Recover why rows were set aside, for days that ran before it was recorded.

`daily_ingest.py` now writes a reason breakdown into every log line, but the
first days of the run only recorded the count. The reports themselves are
still on disk, so the breakdown can be recovered by reading them again.

    python replay_reasons.py                 # every day not already covered
    python replay_reasons.py --date 2026-10-14
    python replay_reasons.py --all           # redo days already covered too

WHAT THIS DOES NOT DO

It does not create cases and it does not touch the case store: every file is
read with `dry_run=True`. It does not rewrite `ingest_log.jsonl` either. That
log is the append-only record of what happened at the time, and editing it
afterwards to add numbers produced later would make it worthless as
evidence — the one thing it exists to be.

The breakdown goes in `skip_reasons.json` instead, marked as a retrospective
re-read, so anyone looking can tell which figures were recorded live and
which were recovered.

COST

One Textract call per report, so roughly a day's worth per run. Do not run
it at 02:30 UTC, when the morning job is already using the same service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date as Date
from pathlib import Path
from typing import Any

from daily_ingest import _reason_counts, run_day

HERE = Path(__file__).parent
OUT = HERE / "skip_reasons.json"
INGEST_LOG = HERE / "ingest_log.jsonl"
REPORT_ROOT = HERE / "reports"


def load_existing(path: Path = OUT) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"_comment": (
            "Why rows were set aside, per day. Days marked source=live were "
            "recorded by daily_ingest.py as it ran. Days marked "
            "source=replay were recovered afterwards by re-reading the "
            "stored PDFs, because the breakdown was not recorded at the "
            "time. ingest_log.jsonl is never edited."
        ), "days": {}}


def days_in_log(path: Path = INGEST_LOG) -> dict[str, dict[str, Any]]:
    """What the live log already knows, keyed by date."""
    found: dict[str, dict[str, Any]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("day"):
                found[entry["day"]] = entry
    except OSError:
        pass
    return found


def days_on_disk(root: Path = REPORT_ROOT) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


async def replay(day: str) -> dict[str, Any]:
    """Re-read one day's reports without creating anything."""
    summary = await run_day(
        Date.fromisoformat(day), dry_run=True, generate=False
    )
    return {
        "source": "replay",
        "reports": summary.get("reports", 0),
        "rows_proposed": summary.get("rows_proposed", 0),
        "rows_accepted": summary.get("rows_accepted", 0),
        "rows_skipped": summary.get("rows_skipped", 0),
        "reasons": _reason_counts(summary.get("files") or []),
    }


async def main_async(args) -> None:
    store = load_existing()
    live = days_in_log()
    available = days_on_disk()

    if args.date:
        wanted = [args.date]
    else:
        wanted = available

    if not available:
        print(f"No report folders under {REPORT_ROOT}. Nothing to replay.")
        return

    for day in wanted:
        if day not in available:
            print(f"{day}  no reports on disk, skipping")
            continue

        # A day whose live log line already carries the breakdown needs no
        # Textract calls; taking the recorded figures is both cheaper and
        # more truthful than re-deriving them.
        recorded = (live.get(day) or {}).get("skip_reasons")
        if recorded and not args.all:
            store["days"][day] = {
                "source": "live",
                "reports": live[day].get("reports", 0),
                "rows_proposed": live[day].get("rows_proposed", 0),
                "rows_accepted": live[day].get("rows_accepted", 0),
                "rows_skipped": live[day].get("rows_skipped", 0),
                "reasons": recorded,
            }
            print(f"{day}  from the live log, no re-read needed")
            continue

        if day in store["days"] and not args.all:
            print(f"{day}  already covered, skipping (use --all to redo)")
            continue

        print(f"{day}  re-reading…", flush=True)
        try:
            store["days"][day] = await replay(day)
        except Exception as exc:
            print(f"{day}  FAILED: {type(exc).__name__}: {exc}")
            continue
        got = store["days"][day]
        print(f"{day}  {got['reports']} reports, {got['rows_skipped']} set aside")

    OUT.write_text(json.dumps(store, indent=2), encoding="utf-8")

    totals: dict[str, int] = {}
    for entry in store["days"].values():
        for reason, count in (entry.get("reasons") or {}).items():
            totals[reason] = totals.get(reason, 0) + count

    grand = sum(totals.values())
    print(f"\nWHY ROWS WERE SET ASIDE  ({grand} across "
          f"{len(store['days'])} days)")
    for reason, count in sorted(totals.items(), key=lambda kv: -kv[1]):
        share = f"{count / grand:.0%}" if grand else "—"
        print(f"   {count:>5}  {share:>5}  {reason}")
    print(f"\nwritten to {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="YYYY-MM-DD, default every day found")
    parser.add_argument("--all", action="store_true",
                        help="redo days already covered")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
