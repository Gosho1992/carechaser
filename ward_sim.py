"""A synthetic surgical ward, running on real time.

CareChaser needs history before it can say anything useful about a ward, and
history takes days. This module manufactures those days honestly: it posts
scanned reports into the inbox each morning and lets clinicians clear them at
the times they actually would.

Two parts, two schedules:

    generate   once a day      writes N scanned reports into reports/<date>/
    tick       every 30 min    each clinician may clear what they own

THE POINT OF THE HIDDEN PROFILES

Every clinician below has a clearing window and a rate, and the weekday
modifies both. Those numbers are the ward's behaviour, and they are written
to ward_profiles.json purely so the claim can be checked afterwards.

`insights.py` never opens that file. It recovers the pattern from the audit
log alone, and the Insights page prints inferred beside actual. That is the
measurable version of "the system learns from its own history": not an
assertion, a comparison you can read in ten seconds.

So: do not import PROFILES into the analytics. The whole exercise is void if
the answer is read rather than derived.

THE PATTERN BEING HIDDEN

Modelled on a surgical hospital: most clinicians clear results after the
operating list finishes, mid-afternoon, not during the morning round, because
in the morning they are in theatre. Monday is the worst day — the weekend's
unreviewed results land at once and the lists are full.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

HERE = Path(__file__).parent
PROFILE_PATH = HERE / "ward_profiles.json"
REPORT_ROOT = HERE / "reports"
MCP_URL = os.environ.get("CARECHASER_MCP_URL", "http://127.0.0.1:8000/mcp")

KARACHI = timezone(timedelta(hours=5))


# --------------------------------------------------------- hidden behaviour


@dataclass(frozen=True)
class Profile:
    """One person's real working pattern. Never exposed to the analytics."""

    name: str
    role: str
    window: tuple[int, int]      # local hours they clear work in, [start, end)
    rate: float                  # chance of clearing one case per 30-min tick
    note: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["window"] = list(self.window)
        return data


PROFILES: dict[str, Profile] = {
    "Dr. Khan": Profile(
        "Dr. Khan", "responsible_clinician", (14, 17), 0.45,
        "Operating in the morning; clears results once the list finishes.",
    ),
    "Dr. Ali": Profile(
        "Dr. Ali", "responsible_clinician", (15, 18), 0.40,
        "Later list, so clears later in the afternoon.",
    ),
    "Dr. Ahmed": Profile(
        "Dr. Ahmed", "responsible_clinician", (8, 10), 0.35,
        "Clinic rather than theatre, so reviews during the morning round.",
    ),
    "Dr. Riaz": Profile(
        "Dr. Riaz", "covering_clinician", (13, 19), 0.25,
        "Covering: picks up escalations across a wide window, slowly.",
    ),
    "Lab Desk": Profile(
        "Lab Desk", "laboratory_team", (8, 18), 0.60,
        "Laboratory runs through the working day.",
    ),
    "Nurse Ayesha": Profile(
        "Nurse Ayesha", "authorized_sample_collector", (7, 10), 0.70,
        "Phlebotomy round first thing.",
    ),
    "Ward 2 Care Team": Profile(
        "Ward 2 Care Team", "responsible_care_team", (11, 19), 0.35,
        "Patient communication fits around the ward day.",
    ),
}

# Monday carries the weekend's backlog: more arrives, less gets cleared.
# Monday = 0 through Sunday = 6.
WEEKDAY_ARRIVALS = {0: 1.7, 1: 1.0, 2: 1.0, 3: 1.1, 4: 1.2, 5: 0.4, 6: 0.3}
WEEKDAY_CLEARING = {0: 0.55, 1: 1.0, 2: 1.0, 3: 0.95, 4: 0.85, 5: 0.3, 6: 0.2}


def write_profiles(path: Path = PROFILE_PATH) -> Path:
    """Record the ground truth so the inference can be marked against it."""
    payload = {
        "_comment": (
            "Ground truth for the synthetic ward. insights.py MUST NOT read "
            "this file: it exists only so inferred behaviour can be compared "
            "with actual behaviour on the Insights page."
        ),
        "written_at": datetime.now(timezone.utc).isoformat(),
        "timezone": "Asia/Karachi (UTC+05:00)",
        "clinicians": {name: p.to_dict() for name, p in PROFILES.items()},
        "weekday_arrivals": WEEKDAY_ARRIVALS,
        "weekday_clearing": WEEKDAY_CLEARING,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------ the decision


def clearing_chance(
    profile: Profile, when: datetime, *, weekday_clearing: dict[int, float] | None = None
) -> float:
    """How likely this person is to clear one case in the half hour at `when`.

    Zero outside their window. Inside it, their base rate scaled by the day.
    Deliberately a pure function of profile and time, so the behaviour under
    test is the behaviour that runs.
    """
    local = when.astimezone(KARACHI)
    start, end = profile.window
    if not (start <= local.hour < end):
        return 0.0
    factors = weekday_clearing or WEEKDAY_CLEARING
    return profile.rate * factors.get(local.weekday(), 1.0)


def should_clear(profile: Profile, when: datetime, rng: random.Random) -> bool:
    chance = clearing_chance(profile, when)
    return chance > 0 and rng.random() < chance


def arrivals_for(day: Date, base: int = 10, *, rng: random.Random | None = None) -> int:
    """How many reports land on a given day."""
    rng = rng or random.Random(day.toordinal())
    scaled = base * WEEKDAY_ARRIVALS.get(day.weekday(), 1.0)
    jitter = rng.uniform(-0.15, 0.15)
    return max(1, round(scaled * (1 + jitter)))


# ------------------------------------------------------------ the reports


TESTS: Sequence[tuple[str, str]] = [
    ("Serum potassium", "critical"),
    ("Serum sodium", "routine"),
    ("Full blood count", "routine"),
    ("Urea and electrolytes", "routine"),
    ("Liver function", "routine"),
    ("Coagulation screen", "critical"),
    ("Group and save", "routine"),
    ("Crossmatch 2 units", "critical"),
    ("Blood culture", "critical"),
    ("Wound swab culture", "routine"),
    ("Histopathology (biopsy)", "routine"),
    ("Frozen section", "critical"),
    ("Chest imaging report", "routine"),
    ("Abdominal ultrasound", "routine"),
    ("Thyroid panel", "routine"),
    ("HbA1c", "routine"),
    ("Vitamin B12", "routine"),
    ("COVID-19 PCR", "routine"),
    ("Troponin", "critical"),
    ("Arterial blood gas", "critical"),
]

STATUS_LINES = [
    ("ordered", "requested, awaiting collection"),
    ("sample_collected", "specimen collected, in transit"),
    ("sample_processed", "received in lab, processing"),
    ("result_produced", "report released, awaiting review"),
    ("result_produced", "released, not yet acknowledged"),
    ("result_reviewed", "reviewed, follow-up pending"),
]

# Values are paired to their test. CareChaser strips every one of them at
# intake, but a report that reads "Vitamin B12 1.4 INR" is not a report any
# clinician would recognise, and the demo is watched by people who would.
VALUES_BY_TEST: dict[str, Sequence[str]] = {
    "Serum potassium": ("6.8 mmol/L", "3.1 mmol/L", "5.4 mmol/L"),
    "Serum sodium": ("128 mmol/L", "142 mmol/L", "149 mmol/L"),
    "Full blood count": ("Hb 9.4 g/dL", "WCC 13.5 x10^9/L", "Plt 94 x10^9/L"),
    "Urea and electrolytes": ("Urea 14.2 mmol/L", "Creatinine 188 umol/L"),
    "Liver function": ("ALT 212 U/L", "Bilirubin 48 umol/L", "ALP 310 U/L"),
    "Coagulation screen": ("INR 3.4", "APTT 58 s", "INR 1.1"),
    "Group and save": ("O Rh positive", "A Rh negative", "B Rh positive"),
    "Crossmatch 2 units": ("2 units reserved", "units not yet reserved"),
    "Blood culture": ("growth at 18 h", "no growth at 48 h"),
    "Wound swab culture": ("mixed growth", "no significant growth"),
    "Histopathology (biopsy)": ("report issued", "awaiting second opinion"),
    "Frozen section": ("margins reported", "deferred to paraffin"),
    "Chest imaging report": ("report issued", "comparison requested"),
    "Abdominal ultrasound": ("report issued", "repeat advised"),
    "Thyroid panel": ("TSH 8.9 mIU/L", "TSH 0.2 mIU/L", "T4 9 pmol/L"),
    "HbA1c": ("HbA1c 9.1 %", "HbA1c 6.4 %"),
    "Vitamin B12": ("142 ng/L", "310 ng/L"),
    "COVID-19 PCR": ("not detected", "detected"),
    "Troponin": ("0.31 ng/mL", "0.04 ng/mL"),
    "Arterial blood gas": ("pH 7.29", "pH 7.44", "pCO2 7.1 kPa"),
}
GENERIC_VALUES = ("within range", "see report", "repeat requested")


def report_lines(day: Date, index: int, rng: random.Random) -> tuple[list[str], list[str]]:
    """One ward printout. Returns its lines and the test names on it."""
    patient = f"MK-{rng.randint(4400, 4699)}"
    chosen = rng.sample(list(TESTS), rng.randint(2, 4))

    lines = [
        "MUZAFFAR KHAN SURGICAL HOSPITAL",
        f"LABORATORY REPORT  -  WARD {rng.choice([1, 2, 3])}",
        "",
        f"Patient ref: {patient}        Date: {day.strftime('%d/%m/%Y')}",
        f"Batch: {day.strftime('%Y%m%d')}-{index:02d}",
        "",
    ]
    names = []
    for test_name, urgency in chosen:
        _state, status = rng.choice(STATUS_LINES)
        # The laboratory flags a result urgent; CareChaser only reads the flag.
        # Not every critical test is flagged, because not every result is out
        # of range — which keeps the critical cases a minority, as they are.
        if urgency == "critical" and rng.random() < 0.6:
            status = f"STAT {status}"
        value = rng.choice(list(VALUES_BY_TEST.get(test_name, GENERIC_VALUES)))
        # Two columns at the same height, the way a printout actually sets
        # them: test and result on the left, status on the right. A tab marks
        # the column break for the renderer. Printed as one long line it runs
        # off the page and Textract reads the truncation; printed on two rows
        # the status arrives detached from its test. Side by side, Textract
        # returns two blocks at the same height and scan_intake rejoins them.
        dots = "." * max(2, 27 - len(test_name))
        lines.append(f"{test_name} {dots} {value}\t{status}")
        names.append(test_name)

    lines += ["", f"Released by: Lab Desk", "Not yet acknowledged"]
    return lines, names


def render_pdf(lines: Sequence[str], path: Path) -> Path:
    """Render as an image and save as PDF — a scan, with no text layer."""
    from PIL import Image, ImageDraw, ImageFont

    # A4 at 200 dpi rather than 150: at 150 the two columns collide and the
    # right-hand status is cut off at the page edge, which Textract then reads
    # as truncated words.
    page = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(page)

    font = None
    for candidate in (
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, 28)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()

    y = 150
    for line in lines:
        left, tab, right = line.partition("\t")
        draw.text((120, y), left, fill="black", font=font)
        if tab:
            draw.text((960, y), right, fill="black", font=font)
        y += 52

    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path, "PDF", resolution=150.0)
    return path


def generate_day(
    day: Optional[Date] = None,
    *,
    count: Optional[int] = None,
    root: Path = REPORT_ROOT,
    seed: Optional[int] = None,
) -> list[Path]:
    """Write one day's scanned reports. Returns the paths written."""
    day = day or datetime.now(KARACHI).date()
    rng = random.Random(seed if seed is not None else day.toordinal())
    count = count if count is not None else arrivals_for(day, rng=rng)

    folder = root / day.isoformat()
    written = []
    for index in range(1, count + 1):
        lines, _names = report_lines(day, index, rng)
        path = folder / f"report-{index:02d}.pdf"
        render_pdf(lines, path)
        written.append(path)
    return written


# ----------------------------------------------------------------- the tick


async def _mcp(tool: str, args: dict) -> dict:
    from mcp import Client

    async with Client(MCP_URL) as client:
        result = await client.call_tool(tool, args, read_timeout_seconds=30)
        text = "".join(b.text for b in result.content if getattr(b, "text", None))
        return json.loads(text) if text else {}


async def tick(*, now: Optional[datetime] = None, rng: Optional[random.Random] = None,
               dry_run: bool = False) -> dict[str, Any]:
    """One half-hour of ward life. Each owner may clear one case they hold.

    Acknowledgements go through the MCP server rather than the store, so the
    run exercises the same path a clinician would, and every clearance is a
    real tool call in the audit trail.
    """
    now = now or datetime.now(timezone.utc)
    rng = rng or random.Random()

    board = await _mcp("get_board_snapshot", {"limit": 100})
    cases = board.get("cases", [])

    acted, skipped = [], 0
    # One case per person per tick. A clinician clearing a whole board in
    # thirty minutes is not a ward, it is a cron job.
    already: set[str] = set()

    for case in cases:
        owner = case.get("owner")
        profile = PROFILES.get(owner)
        if profile is None:
            skipped += 1
            continue
        if owner in already:
            continue
        if not should_clear(profile, now, rng):
            continue
        already.add(owner)

        event = {
            "authorized_sample_collector": "sample_collected",
            "laboratory_team": "result_released",
            "responsible_clinician": "result_reviewed",
            "covering_clinician": "result_reviewed",
            "responsible_care_team": "patient_informed",
        }.get(profile.role, "result_reviewed")

        if dry_run:
            acted.append({"case_id": case["case_id"], "by": owner, "event": event,
                          "written": False})
            continue

        result = await _mcp("acknowledge_result", {
            "case_id": case["case_id"],
            "actor_id": owner,
            "actor_role": profile.role,
            "event_type": event,
            "note": "Cleared during the ward round (synthetic ward activity)",
            "event_id": f"sim-{case['case_id']}-{now.isoformat(timespec='minutes')}",
        })
        acted.append({"case_id": case["case_id"], "by": owner, "event": event,
                      "written": bool(result.get("ok", True)),
                      "message": result.get("message", "")})

    return {
        "at": now.isoformat(),
        "considered": len(cases),
        "cleared": len(acted),
        "no_profile": skipped,
        "actions": acted,
    }


# ----------------------------------------------------------------- the CLI


def main() -> None:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write one day's scanned reports")
    gen.add_argument("--count", type=int, default=None)
    gen.add_argument("--date", default=None, help="YYYY-MM-DD, default today")
    gen.add_argument("--seed", type=int, default=None)

    run = sub.add_parser("tick", help="one half hour of ward activity")
    run.add_argument("--dry-run", action="store_true")

    sub.add_parser("profiles", help="write ward_profiles.json")

    args = parser.parse_args()

    if args.command == "profiles":
        path = write_profiles()
        print(f"wrote {path.name}")
        for name, profile in PROFILES.items():
            start, end = profile.window
            print(f"  {name:<18} clears {start:02d}:00-{end:02d}:00  rate {profile.rate}")
        return

    if args.command == "generate":
        day = Date.fromisoformat(args.date) if args.date else datetime.now(KARACHI).date()
        paths = generate_day(day, count=args.count, seed=args.seed)
        print(f"{len(paths)} reports for {day} ({day.strftime('%A')})")
        for path in paths:
            print(f"  {path.relative_to(HERE)}")
        return

    if args.command == "tick":
        outcome = asyncio.run(tick(dry_run=args.dry_run))
        print(f"{outcome['at']}  considered {outcome['considered']}, "
              f"cleared {outcome['cleared']}")
        for action in outcome["actions"]:
            mark = "would clear" if args.dry_run else "cleared"
            print(f"  {mark} {action['case_id']} by {action['by']} "
                  f"({action['event']})")


if __name__ == "__main__":
    main()
