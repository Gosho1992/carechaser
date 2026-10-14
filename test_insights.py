"""Tests for reading the ward's own history.

Everything here is built from hand-made history entries, so the arithmetic is
checkable by hand and nothing depends on a server, AWS, or the simulator.

The last test in this file is the important one: it asserts that insights.py
never reads the generator's parameters. A comparison between inferred and
actual behaviour proves nothing if the answer was available to the inferrer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import insights
from carechaser_engine import load_workflow
from insights import (
    CONFIDENT_AT,
    CompletedStage,
    breaches_by_weekday,
    build,
    chase_hour,
    completed_stages,
    describe_person,
    infer_window,
    local_zone,
    person_patterns,
    stage_durations,
)

WORKFLOW = load_workflow(Path(__file__).parent / "workflow.yaml")
KARACHI = timezone(timedelta(hours=5))


def stage(
    *, state="result_produced", hours=2.0, by="Dr. Khan", at_hour=15,
    weekday=1, role="responsible_clinician", urgency="routine", case_id="CC-001",
) -> CompletedStage:
    """A completed stage finishing at a chosen local hour and weekday."""
    # 2026-10-12 is a Monday.
    day = datetime(2026, 10, 12, tzinfo=KARACHI) + timedelta(days=weekday)
    completed = day.replace(hour=at_hour)
    return CompletedStage(
        case_id=case_id,
        state=state,
        entered_at=completed - timedelta(hours=hours),
        completed_at=completed,
        by=by,
        role=role,
        urgency=urgency,
    )


# ------------------------------------------------------------- the reading


def test_the_ward_clock_is_read_from_the_workflow():
    assert local_zone(WORKFLOW).utcoffset(None) == timedelta(hours=5)


def test_an_unknown_timezone_falls_back_to_utc():
    assert local_zone({"engine_settings": {"timezone_default": "Mars/Olympus"}}) \
        .utcoffset(None) == timedelta(0)


def test_hours_are_measured_across_timezones_correctly():
    item = stage(hours=3.5)
    assert item.hours == pytest.approx(3.5)


def test_the_local_hour_is_the_wards_hour_not_utc():
    item = stage(at_hour=15)
    assert item.completed_at.astimezone(timezone.utc).hour == 10
    assert item.local_hour(KARACHI) == 15


class FakeEntry:
    def __init__(self, state, entered_at, completed_at, by, role):
        self.state = state
        self.entered_at = entered_at
        self.completed_at = completed_at
        self.completed_by = by
        self.completed_by_role = role


class FakeCase:
    def __init__(self, case_id, history, urgency="routine"):
        self.case_id = case_id
        self.history = history
        self.result_urgency = urgency


def test_history_is_flattened_across_cases():
    now = datetime(2026, 10, 14, 15, tzinfo=KARACHI)
    cases = [
        FakeCase("CC-001", [
            FakeEntry("ordered", now - timedelta(hours=5), now - timedelta(hours=3),
                      "Nurse Ayesha", "authorized_sample_collector"),
            FakeEntry("result_produced", now - timedelta(hours=3), now,
                      "Dr. Khan", "responsible_clinician"),
        ]),
        FakeCase("CC-002", []),
    ]
    found = completed_stages(cases)
    assert len(found) == 2
    assert {item.case_id for item in found} == {"CC-001"}
    assert found[1].by == "Dr. Khan"


def test_a_malformed_entry_does_not_hide_the_rest():
    now = datetime(2026, 10, 14, 15, tzinfo=KARACHI)
    good = FakeEntry("ordered", now - timedelta(hours=2), now, "Lab Desk", "laboratory_team")
    cases = [FakeCase("CC-001", [object(), good])]
    assert len(completed_stages(cases)) == 1


# -------------------------------------------------- 1. where the time goes


def test_the_median_is_used_not_the_mean():
    """One case left over a weekend must not define a typical case."""
    stages = [stage(hours=h) for h in (1, 1, 1, 1, 200)]
    row = stage_durations(stages, WORKFLOW)[0]
    assert row["median_hours"] == 1.0
    assert row["worst_hours"] == 200.0


def test_breaches_are_counted_against_the_stages_own_limit():
    # result_produced carries a 24 hour limit.
    stages = [stage(hours=h) for h in (2, 10, 30, 40)]
    row = stage_durations(stages, WORKFLOW)[0]
    assert row["sla_hours"] == 24.0
    assert row["breached"] == 2
    assert row["breach_rate"] == 0.5


def test_the_bottleneck_is_relative_to_the_limit_not_the_raw_hours():
    """4 hours against a 4 hour limit is worse than 20 against 24."""
    stages = (
        [stage(state="sample_collected", hours=4.5) for _ in range(5)]
        + [stage(state="result_produced", hours=20) for _ in range(5)]
    )
    rows = stage_durations(stages, WORKFLOW)
    assert rows[0]["state"] == "sample_collected"


def test_a_thin_sample_is_marked_thin():
    rows = stage_durations([stage()], WORKFLOW)
    assert rows[0]["n"] == 1
    assert rows[0]["confident"] is False


def test_a_full_sample_is_not_marked_thin():
    rows = stage_durations([stage() for _ in range(CONFIDENT_AT)], WORKFLOW)
    assert rows[0]["confident"] is True


# ------------------------------------------------ 2. when each person works


def test_a_window_is_the_narrowest_run_holding_most_of_the_work():
    counts = {14: 5, 15: 6, 16: 4, 9: 1}
    assert infer_window(counts) == (14, 17)


def test_a_person_who_works_twice_a_day_is_not_described_as_working_all_day():
    counts = {9: 10, 16: 1}
    start, end = infer_window(counts)
    assert (start, end) == (9, 10)


def test_a_window_can_run_over_midnight():
    counts = {23: 5, 0: 5, 1: 5}
    assert infer_window(counts) == (23, 2)


def test_no_work_gives_no_window():
    assert infer_window({}) is None


def test_the_chase_goes_out_the_hour_before_they_start():
    assert chase_hour((14, 17)) == 13
    assert chase_hour((0, 3)) == 23


def test_an_afternoon_surgeon_is_recovered_from_the_log_alone():
    """The pattern the simulator hides; found here only from timestamps."""
    stages = (
        [stage(by="Dr. Khan", at_hour=14) for _ in range(4)]
        + [stage(by="Dr. Khan", at_hour=15) for _ in range(5)]
        + [stage(by="Dr. Khan", at_hour=16) for _ in range(3)]
    )
    row = person_patterns(stages, KARACHI)[0]
    assert row["person"] == "Dr. Khan"
    assert row["window"] == [14, 17]
    assert row["chase_at"] == 13
    assert row["confident"] is True


def test_people_are_ordered_by_how_much_they_have_done():
    stages = (
        [stage(by="Dr. Khan") for _ in range(3)]
        + [stage(by="Lab Desk") for _ in range(9)]
    )
    assert [row["person"] for row in person_patterns(stages, KARACHI)] == \
        ["Lab Desk", "Dr. Khan"]


def test_a_thin_pattern_is_spoken_with_a_caveat():
    stages = [stage(by="Dr. Ali", at_hour=16)]
    said = describe_person(person_patterns(stages, KARACHI)[0])
    assert "Too few to rely on" in said


def test_a_solid_pattern_is_spoken_as_advice():
    stages = [stage(by="Dr. Ali", at_hour=16) for _ in range(CONFIDENT_AT)]
    said = describe_person(person_patterns(stages, KARACHI)[0])
    assert "clears results between" in said
    assert "15:00" in said          # chase the hour before


# ---------------------------------------------------- 3. the worst weekday


def test_breaches_are_grouped_by_the_day_they_happened():
    stages = (
        [stage(weekday=0, hours=30) for _ in range(3)]      # Monday, late
        + [stage(weekday=0, hours=2) for _ in range(1)]
        + [stage(weekday=2, hours=2) for _ in range(4)]     # Wednesday, fine
    )
    rows = {row["name"]: row for row in breaches_by_weekday(stages, WORKFLOW, KARACHI)}
    assert rows["Monday"]["breached"] == 3
    assert rows["Monday"]["n"] == 4
    assert rows["Wednesday"]["breached"] == 0


def test_every_weekday_appears_even_with_no_work():
    rows = breaches_by_weekday([], WORKFLOW, KARACHI)
    assert len(rows) == 7
    assert all(row["n"] == 0 for row in rows)


# ----------------------------------------------------------- the whole lot


def test_an_empty_history_says_so_rather_than_inventing_a_pattern():
    result = build([], WORKFLOW)
    assert result.stages_completed == 0
    assert "Nothing to read" in result.headline


def test_a_thin_history_refuses_to_state_a_pattern():
    now = datetime(2026, 10, 14, 15, tzinfo=KARACHI)
    case = FakeCase("CC-001", [
        FakeEntry("result_produced", now - timedelta(hours=2), now,
                  "Dr. Khan", "responsible_clinician"),
    ])
    result = build([case], WORKFLOW, now=now)
    assert "still forming" in result.headline


def test_the_headline_names_the_bottleneck_once_there_is_enough():
    now = datetime(2026, 10, 14, 15, tzinfo=KARACHI)
    history = [
        FakeEntry("sample_collected", now - timedelta(hours=9), now - timedelta(hours=1),
                  "Lab Desk", "laboratory_team")
        for _ in range(CONFIDENT_AT)
    ]
    result = build([FakeCase("CC-001", history)], WORKFLOW, now=now)
    assert "bottleneck" in result.headline
    assert "Sample Collected" in result.headline


def test_days_observed_counts_from_the_earliest_completion():
    now = datetime(2026, 10, 14, 12, tzinfo=KARACHI)
    case = FakeCase("CC-001", [
        FakeEntry("ordered", now - timedelta(days=3, hours=2),
                  now - timedelta(days=3), "Lab Desk", "laboratory_team"),
    ])
    assert build([case], WORKFLOW, now=now).days_observed == pytest.approx(3.0, abs=0.01)


def test_the_result_serialises_for_the_api():
    payload = build([], WORKFLOW).to_dict()
    assert set(payload) >= {
        "generated_at", "days_observed", "cases", "stages_completed",
        "stages", "people", "weekdays", "headline", "confident_at",
    }


# -------------------------------------------------------------- the guard


def test_the_analytics_never_read_the_generators_parameters():
    """Without this, inferred-versus-actual proves nothing."""
    source = Path(insights.__file__).read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1]        # skip the module docstring
    for forbidden in ("ward_profiles", "ward_sim", "PROFILES",
                      "WEEKDAY_ARRIVALS", "WEEKDAY_CLEARING"):
        assert forbidden not in body, f"insights.py must not reference {forbidden}"
