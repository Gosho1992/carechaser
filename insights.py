"""What the audit trail knows about how this ward actually works.

Every completed stage leaves a HistoryEntry: which stage, when it was entered,
when it was completed, and by whom. Nobody has ever read those back. Three
questions they can answer:

    where does the time go        median hours per stage, against its limit
    when does each person work    the hours of the day they actually clear
    which day backs up            breaches by weekday

And one thing those answers are good for: if a clinician has never
acknowledged anything outside 14:00-17:00, a chase fired at 23:00 is a
notification nobody will see. The inferred window tells the chase engine when
to send instead.

WHAT THIS MODULE MUST NOT DO

The ward it is reading is synthetic, and the behaviour it is recovering was
generated from parameters held elsewhere. This module never reads those
parameters. Everything here is derived from the audit trail, exactly as it
would be on a real ward where no such file exists. A test asserts that this
file does not so much as mention them, because a comparison between inferred
and actual is worth nothing if the answer was available to the inferrer.

Nothing here is a clinical judgement. "Dr. Khan clears results in the
afternoon" is a statement about a workflow, counted from timestamps.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

# Enough observations before a pattern is worth stating out loud. Below this
# the numbers are still reported, flagged as thin, and never spoken as fact.
CONFIDENT_AT = 8

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]


def local_zone(workflow: Mapping[str, Any]) -> timezone:
    """The ward's own clock. Hour-of-day means nothing in UTC on a ward in
    Karachi, and every window in this module is an hour of the working day."""
    name = str(
        (workflow.get("engine_settings") or {}).get("timezone_default", "")
    ).strip()
    # A fixed offset is enough: these are working hours, not scheduling, and
    # the alternative is a tzdata dependency for one lookup.
    offsets = {
        "Asia/Karachi": 5, "Asia/Kolkata": 5.5, "Asia/Dhaka": 6,
        "Asia/Dubai": 4, "Europe/London": 0, "UTC": 0,
    }
    return timezone(timedelta(hours=offsets.get(name, 0)))


# ----------------------------------------------------------------- the facts


@dataclass(frozen=True)
class CompletedStage:
    """One stage, start to finish, as the audit trail recorded it."""

    case_id: str
    state: str
    entered_at: datetime
    completed_at: datetime
    by: str
    role: str
    urgency: str = "routine"

    @property
    def hours(self) -> float:
        delta = self.completed_at.astimezone(timezone.utc) - self.entered_at.astimezone(timezone.utc)
        return delta.total_seconds() / 3600

    def local_hour(self, zone: timezone) -> int:
        return self.completed_at.astimezone(zone).hour

    def local_weekday(self, zone: timezone) -> int:
        return self.completed_at.astimezone(zone).weekday()


def completed_stages(cases: Iterable[Any]) -> list[CompletedStage]:
    """Flatten every case's history into one list of finished stages."""
    out: list[CompletedStage] = []
    for case in cases:
        urgency = getattr(case, "result_urgency", "routine")
        for entry in getattr(case, "history", ()) or ():
            try:
                out.append(CompletedStage(
                    case_id=case.case_id,
                    state=entry.state,
                    entered_at=entry.entered_at,
                    completed_at=entry.completed_at,
                    by=entry.completed_by,
                    role=entry.completed_by_role,
                    urgency=urgency,
                ))
            except AttributeError:
                continue          # a malformed entry must not hide the rest
    return out


def _sla_for(state: str, workflow: Mapping[str, Any]) -> Optional[float]:
    for item in workflow.get("states", []):
        if item.get("state") == state:
            hours = item.get("sla_hours")
            return float(hours) if hours else None
    return None


def _display_name(state: str, workflow: Mapping[str, Any]) -> str:
    for item in workflow.get("states", []):
        if item.get("state") == state:
            return str(item.get("display_name", state))
    return state


# --------------------------------------------------- 1. where the time goes


def stage_durations(
    stages: Sequence[CompletedStage], workflow: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Median hours spent in each stage, against that stage's limit.

    Median rather than mean: one case left over a weekend drags a mean into
    nonsense, and the question is what a typical case does.
    """
    grouped: dict[str, list[CompletedStage]] = defaultdict(list)
    for stage in stages:
        grouped[stage.state].append(stage)

    rows = []
    for state, items in grouped.items():
        hours = sorted(item.hours for item in items)
        sla = _sla_for(state, workflow)
        breached = [h for h in hours if sla is not None and h > sla]
        rows.append({
            "state": state,
            "display_name": _display_name(state, workflow),
            "n": len(hours),
            "median_hours": round(statistics.median(hours), 2),
            "worst_hours": round(hours[-1], 2),
            "sla_hours": sla,
            "breached": len(breached),
            "breach_rate": round(len(breached) / len(hours), 3) if hours else 0.0,
            "confident": len(hours) >= CONFIDENT_AT,
        })

    # The slowest stage relative to its own limit is the bottleneck, not the
    # one with the biggest number: four hours against a four hour limit is a
    # worse problem than twenty against twenty-four.
    rows.sort(
        key=lambda row: (row["median_hours"] / row["sla_hours"])
        if row["sla_hours"] else 0,
        reverse=True,
    )
    return rows


# ------------------------------------------------- 2. when each person works


def clearing_hours(
    stages: Sequence[CompletedStage], zone: timezone
) -> dict[str, dict[int, int]]:
    """Per person, how many stages they completed in each hour of the day."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for stage in stages:
        counts[stage.by][stage.local_hour(zone)] += 1
    return {person: dict(hours) for person, hours in counts.items()}


def infer_window(
    hour_counts: Mapping[int, int], *, coverage: float = 0.8
) -> Optional[tuple[int, int]]:
    """The narrowest run of hours holding `coverage` of someone's work.

    Returned as [start, end) in local hours. None when there is nothing to go
    on. Narrowest rather than first-found: a person who clears at 09:00 and
    again at 16:00 should not be described as working 09:00-17:00.
    """
    total = sum(hour_counts.values())
    if total <= 0:
        return None

    needed = total * coverage
    best: Optional[tuple[int, int]] = None
    for start in range(24):
        running = 0
        for length in range(1, 25):
            hour = (start + length - 1) % 24
            running += hour_counts.get(hour, 0)
            if running >= needed:
                if best is None or length < (best[1] - best[0]):
                    best = (start, start + length)
                break
    if best is None:
        return None
    return (best[0] % 24, ((best[1] - 1) % 24) + 1)


def person_patterns(
    stages: Sequence[CompletedStage], zone: timezone
) -> list[dict[str, Any]]:
    """One row per person: when they work, and when to reach them."""
    rows = []
    for person, hours in clearing_hours(stages, zone).items():
        total = sum(hours.values())
        window = infer_window(hours)
        busiest = max(hours.items(), key=lambda item: item[1])[0] if hours else None
        rows.append({
            "person": person,
            "n": total,
            "hours": hours,
            "window": list(window) if window else None,
            "busiest_hour": busiest,
            "chase_at": chase_hour(window) if window else None,
            "confident": total >= CONFIDENT_AT,
        })
    rows.sort(key=lambda row: row["n"], reverse=True)
    return rows


def chase_hour(window: tuple[int, int]) -> int:
    """When to send so it is waiting when they sit down.

    The hour before they start. Sent during the window it competes with the
    work itself; sent after, it waits until tomorrow.
    """
    start, _end = window
    return (start - 1) % 24


def describe_person(row: Mapping[str, Any]) -> str:
    """One speakable line. Hedged while the sample is thin."""
    person = row["person"]
    if not row["window"]:
        return f"No completed work recorded for {person} yet."
    start, end = row["window"]
    span = f"{start:02d}:00 and {end:02d}:00"
    if not row["confident"]:
        return (
            f"{person} has cleared {row['n']} "
            f"{'case' if row['n'] == 1 else 'cases'} so far, all between "
            f"{span}. Too few to rely on yet."
        )
    return (
        f"{person} clears results between {span}. "
        f"A chase sent by {row['chase_at']:02d}:00 is waiting when they start."
    )


# ----------------------------------------------------- 3. the worst weekday


def breaches_by_weekday(
    stages: Sequence[CompletedStage], workflow: Mapping[str, Any], zone: timezone
) -> list[dict[str, Any]]:
    """How often a stage ran past its limit, by the day it was completed."""
    totals = Counter()
    breaches = Counter()
    for stage in stages:
        day = stage.local_weekday(zone)
        totals[day] += 1
        sla = _sla_for(stage.state, workflow)
        if sla is not None and stage.hours > sla:
            breaches[day] += 1

    rows = []
    for day in range(7):
        seen = totals.get(day, 0)
        late = breaches.get(day, 0)
        rows.append({
            "weekday": day,
            "name": WEEKDAYS[day],
            "n": seen,
            "breached": late,
            "breach_rate": round(late / seen, 3) if seen else 0.0,
            "confident": seen >= CONFIDENT_AT,
        })
    return rows


# -------------------------------------------------------------- everything


@dataclass
class Insights:
    generated_at: datetime
    days_observed: float
    cases: int
    stages_completed: int
    stages: list[dict[str, Any]] = field(default_factory=list)
    people: list[dict[str, Any]] = field(default_factory=list)
    weekdays: list[dict[str, Any]] = field(default_factory=list)
    headline: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "days_observed": self.days_observed,
            "cases": self.cases,
            "stages_completed": self.stages_completed,
            "stages": self.stages,
            "people": self.people,
            "weekdays": self.weekdays,
            "headline": self.headline,
            "confident_at": CONFIDENT_AT,
        }


def build(
    cases: Iterable[Any],
    workflow: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
) -> Insights:
    """Everything the history can say, with its own sample size attached."""
    cases = list(cases)
    now = now or datetime.now(timezone.utc)
    zone = local_zone(workflow)
    stages = completed_stages(cases)

    if stages:
        earliest = min(stage.completed_at for stage in stages)
        days = round(
            (now.astimezone(timezone.utc) - earliest.astimezone(timezone.utc))
            .total_seconds() / 86400, 2
        )
    else:
        days = 0.0

    stage_rows = stage_durations(stages, workflow)
    people = person_patterns(stages, zone)
    weekdays = breaches_by_weekday(stages, workflow, zone)

    return Insights(
        generated_at=now,
        days_observed=max(days, 0.0),
        cases=len(cases),
        stages_completed=len(stages),
        stages=stage_rows,
        people=people,
        weekdays=weekdays,
        headline=_headline(stage_rows, people, weekdays, len(stages)),
    )


def _headline(
    stages: Sequence[Mapping[str, Any]],
    people: Sequence[Mapping[str, Any]],
    weekdays: Sequence[Mapping[str, Any]],
    total: int,
) -> str:
    """The one sentence worth saying, or an honest refusal to say one."""
    if total == 0:
        return "No completed stages yet. Nothing to read from the history."
    if total < CONFIDENT_AT:
        return (
            f"Only {total} completed "
            f"{'stage' if total == 1 else 'stages'} so far. "
            f"The patterns below are still forming."
        )

    confident_stages = [row for row in stages if row["confident"] and row["sla_hours"]]
    if confident_stages:
        worst = confident_stages[0]
        share = worst["median_hours"] / worst["sla_hours"]
        if share >= 1:
            return (
                f"{worst['display_name']} is the bottleneck: a typical case "
                f"sits there {worst['median_hours']:g} hours against a "
                f"{worst['sla_hours']:g} hour limit."
            )
        return (
            f"{worst['display_name']} takes the largest share of its limit: "
            f"{worst['median_hours']:g} of {worst['sla_hours']:g} hours."
        )

    confident_people = [row for row in people if row["confident"]]
    if confident_people:
        return describe_person(confident_people[0])

    late = [row for row in weekdays if row["confident"] and row["breached"]]
    if late:
        worst = max(late, key=lambda row: row["breach_rate"])
        return (
            f"{worst['name']} runs late most often: "
            f"{worst['breached']} of {worst['n']} stages past their limit."
        )
    return f"{total} completed stages recorded. No stage is running late."


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from case_store import CaseStore

    parser = argparse.ArgumentParser(description="Read the ward's own history")
    parser.add_argument("--cases", default="carechaser_cases.json")
    parser.add_argument("--workflow", default="workflow.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    store = CaseStore(Path(args.cases), Path(args.workflow))
    result = build(store.all_cases(), store.workflow)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        raise SystemExit

    print(f"\n{result.headline}\n")
    print(f"observed over {result.days_observed:g} days  ·  "
          f"{result.cases} cases  ·  {result.stages_completed} completed stages\n")

    print("WHERE THE TIME GOES")
    for row in result.stages:
        limit = f"{row['sla_hours']:g}h limit" if row["sla_hours"] else "no limit"
        flag = "" if row["confident"] else "   (thin)"
        print(f"  {row['display_name'][:38]:<38} median {row['median_hours']:>6.1f}h  "
              f"{limit:<10}  {row['breached']}/{row['n']} late{flag}")

    print("\nWHEN EACH PERSON CLEARS WORK")
    for row in result.people:
        print(f"  {describe_person(row)}")

    print("\nBREACHES BY WEEKDAY")
    for row in result.weekdays:
        if row["n"]:
            print(f"  {row['name']:<10} {row['breached']:>3} of {row['n']:>3} late"
                  f"{'' if row['confident'] else '   (thin)'}")
