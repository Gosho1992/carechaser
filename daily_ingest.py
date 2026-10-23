"""One morning's post: scanned reports in, tracked cases out.

Run once a day by a systemd timer. Every step goes through the deployed MCP
server rather than touching the store directly, so the operating run exercises
the same path a person clicking in the browser would.

    ward_sim.generate_day   write the day's scanned reports
    scan_intake             Textract reads each page, columns rejoined
    MCP read_report         rows proposed, every clinical value stripped
    MCP confirm_report_rows the rows that classified cleanly become cases

WHAT IS DELIBERATELY NOT AUTOMATED

In the product a human reviews the proposed rows before anything is created.
There is no human at 07:30, so this job applies the strictest possible stand-in
rule: a row is accepted only if intake classified it completely and raised no
issues. Anything else — a header line, an unrecognised status, a test with no
stage — is recorded as skipped and never guessed at.

That is why the skipped count belongs in the evidence rather than being hidden.
"How many lines did it refuse to interpret" is a more useful number than a
clean sweep, and a run with no skips would mean the guard was not working.

Each run appends one line to ingest_log.jsonl: what arrived, what was read,
what became a case, and what was set aside.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date as Date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).parent
INGEST_LOG = Path(os.environ.get("CARECHASER_INGEST_LOG", HERE / "ingest_log.jsonl"))


# Why a row was set aside, bucketed. Intake writes a sentence with the
# offending text inside it ("status 'reviewd, pendng' is not one this
# workflow knows"), which is right for a human reading one row and useless
# for counting: every row would be its own category. Half of all OCR'd rows
# get set aside, and "why" is the first thing anyone asks.
SKIP_REASONS: tuple[tuple[str, str], ...] = (
    ("document metadata", "not_a_test_row"),
    ("duplicate test", "duplicate_on_report"),
    ("no status given", "no_status_printed"),
    ("is not one this workflow knows", "status_not_recognised"),
    ("is not a stage a case can start in", "stage_cannot_start"),
    ("has no valid owner role", "no_owner_role_configured"),
)


def classify_skip(why: str) -> str:
    """Bucket one skip reason. Unknown shapes keep their own bucket rather
    than being folded into a category they do not belong to."""
    lowered = (why or "").lower()
    for needle, bucket in SKIP_REASONS:
        if needle in lowered:
            return bucket
    return "other"
MCP_URL = os.environ.get("CARECHASER_MCP_URL", "http://127.0.0.1:8000/mcp")


async def _mcp(tool: str, args: dict) -> dict:
    from mcp import Client

    async with Client(MCP_URL) as client:
        result = await client.call_tool(tool, args, read_timeout_seconds=60)
        text = "".join(b.text for b in result.content if getattr(b, "text", None))
        return json.loads(text) if text else {}


async def ingest_file(path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """One report: read the page, propose rows, create the clean ones."""
    from scan_intake import ScanError, read_report_file

    outcome: dict[str, Any] = {"file": path.name}

    try:
        scanned = read_report_file(path)
    except ScanError as exc:
        return {**outcome, "ok": False, "error": str(exc)}

    outcome["lines"] = scanned.line_count
    outcome["rows_on_page"] = scanned.row_count

    proposed = await _mcp("read_report", {"text": scanned.text})
    rows = proposed.get("rows", [])
    patient = proposed.get("patient")

    clean = [row for row in rows if row.get("ok")]
    skipped = [
        {
            "text": row.get("test_name", ""),
            "why": (row.get("issues") or ["unclassified"])[0],
            "reason": classify_skip((row.get("issues") or ["unclassified"])[0]),
        }
        for row in rows
        if not row.get("ok")
    ]

    outcome["proposed"] = len(rows)
    # Page furniture, counted apart from the tests. Including it in the
    # denominator is what made a run with no failed test rows report 51%.
    outcome["metadata"] = sum(1 for row in rows if row.get("metadata"))
    outcome["accepted"] = len(clean)
    outcome["skipped"] = skipped
    outcome["removed_fragments"] = sum(len(row.get("removed") or []) for row in rows)

    if dry_run or not clean:
        outcome["ok"] = True
        outcome["created"] = []
        return outcome

    confirmed = await _mcp("confirm_report_rows", {
        "rows": [
            {
                "test_name": row["test_name"],
                "state": row["state"],
                "owner_role": row.get("owner_role"),
                "hours_waiting": row.get("hours_waiting", 0.0),
                "patient": row.get("patient") or patient,
                "urgency": row.get("urgency", "routine"),
            }
            for row in clean
        ],
        "patient": patient,
    })

    outcome["ok"] = True
    outcome["created"] = confirmed.get("created", [])
    outcome["refused"] = confirmed.get("skipped", [])
    return outcome


def _reason_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    """How many rows were set aside for each reason, across one day."""
    counts: dict[str, int] = {}
    for outcome in results:
        for row in outcome.get("skipped") or []:
            bucket = row.get("reason") or classify_skip(row.get("why", ""))
            counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


async def run_day(
    day: Optional[Date] = None,
    *,
    count: Optional[int] = None,
    dry_run: bool = False,
    generate: bool = True,
) -> dict[str, Any]:
    from ward_sim import KARACHI, REPORT_ROOT, generate_day

    day = day or datetime.now(KARACHI).date()
    started = datetime.now(timezone.utc)

    if generate:
        paths = generate_day(day, count=count)
    else:
        folder = REPORT_ROOT / day.isoformat()
        paths = sorted(folder.glob("*.pdf")) if folder.exists() else []

    results = []
    for path in paths:
        results.append(await ingest_file(path, dry_run=dry_run))

    created = sum(len(r.get("created") or []) for r in results)
    accepted = sum(r.get("accepted", 0) for r in results)
    proposed = sum(r.get("proposed", 0) for r in results)
    skipped = sum(len(r.get("skipped") or []) for r in results)
    metadata = sum(r.get("metadata", 0) for r in results)
    removed = sum(r.get("removed_fragments", 0) for r in results)
    failed = [r["file"] for r in results if not r.get("ok")]

    summary = {
        "day": day.isoformat(),
        "weekday": day.strftime("%A"),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "reports": len(paths),
        "rows_proposed": proposed,
        "rows_metadata": metadata,
        "rows_tests": proposed - metadata,
        "rows_accepted": accepted,
        "rows_skipped": skipped,
        # The breakdown, not just the count. Without this the log records
        # that half the rows were set aside and nothing about why, which is
        # the one question the number provokes.
        "skip_reasons": _reason_counts(results),
        "cases_created": created,
        "clinical_fragments_removed": removed,
        "unreadable_files": failed,
        "dry_run": dry_run,
    }

    if not dry_run:
        INGEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with INGEST_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")

    summary["files"] = results
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, default today")
    parser.add_argument("--count", type=int, default=None,
                        help="reports to post; default follows the weekday")
    parser.add_argument("--dry-run", action="store_true",
                        help="read everything, create nothing, log nothing")
    parser.add_argument("--no-generate", action="store_true",
                        help="ingest the day's existing folder instead of writing new reports")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    day = Date.fromisoformat(args.date) if args.date else None
    summary = asyncio.run(run_day(
        day, count=args.count, dry_run=args.dry_run, generate=not args.no_generate
    ))

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"{summary['day']} ({summary['weekday']})"
          f"{'  DRY RUN' if summary['dry_run'] else ''}")
    print(f"  reports posted      {summary['reports']}")
    print(f"  rows proposed       {summary['rows_proposed']}")
    print(f"  rows accepted       {summary['rows_accepted']}")
    print(f"  rows set aside      {summary['rows_skipped']}")
    print(f"  cases created       {summary['cases_created']}")
    print(f"  values stripped     {summary['clinical_fragments_removed']}")
    if summary["unreadable_files"]:
        print(f"  UNREADABLE          {', '.join(summary['unreadable_files'])}")
        sys.exit(1)


if __name__ == "__main__":
    main()
