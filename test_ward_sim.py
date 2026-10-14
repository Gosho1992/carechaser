"""Tests for the synthetic ward.

The behaviour under test is the behaviour that runs: `clearing_chance` and
`arrivals_for` are pure functions of a profile and a moment, so the pattern
the analytics will later have to recover is pinned down here.
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta, timezone

import pytest

import ward_sim
from ward_sim import (
    KARACHI,
    PROFILES,
    Profile,
    arrivals_for,
    clearing_chance,
    report_lines,
    should_clear,
    write_profiles,
)

KHAN = PROFILES["Dr. Khan"]


def at(hour: int, *, weekday: int = 1, minute: int = 0) -> datetime:
    """A Karachi-local time on a chosen weekday. Default Tuesday."""
    # 2026-10-12 is a Monday, so Monday + weekday lands where we want.
    day = date(2026, 10, 12) + timedelta(days=weekday)
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=KARACHI)


# ------------------------------------------------- the window is respected


def test_nobody_clears_before_their_window_opens():
    assert clearing_chance(KHAN, at(9)) == 0.0


def test_nobody_clears_after_their_window_closes():
    assert clearing_chance(KHAN, at(18)) == 0.0


def test_the_window_includes_its_first_hour_and_excludes_its_last():
    start, end = KHAN.window
    assert clearing_chance(KHAN, at(start)) > 0
    assert clearing_chance(KHAN, at(end)) == 0.0


def test_a_surgeon_clears_in_the_afternoon_not_the_morning():
    """The pattern the Insights page has to recover from the log alone."""
    assert clearing_chance(KHAN, at(15)) > clearing_chance(KHAN, at(9))
    assert clearing_chance(KHAN, at(9)) == 0.0


def test_the_clinic_doctor_is_the_morning_exception():
    ahmed = PROFILES["Dr. Ahmed"]
    assert clearing_chance(ahmed, at(9)) > 0
    assert clearing_chance(ahmed, at(15)) == 0.0


def test_windows_are_expressed_in_karachi_time_not_utc():
    """15:00 Karachi is 10:00 UTC; the same instant must read as in-window."""
    local = at(15)
    utc = local.astimezone(timezone.utc)
    assert utc.hour == 10
    assert clearing_chance(KHAN, utc) == clearing_chance(KHAN, local) > 0


# ------------------------------------------------------- Monday is the worst


def test_monday_clears_more_slowly_than_midweek():
    assert clearing_chance(KHAN, at(15, weekday=0)) < clearing_chance(KHAN, at(15, weekday=1))


def test_monday_brings_the_most_work():
    monday = date(2026, 10, 12)
    tuesday = date(2026, 10, 13)
    assert monday.weekday() == 0
    assert arrivals_for(monday, base=10) > arrivals_for(tuesday, base=10)


def test_the_weekend_is_quiet_on_both_counts():
    saturday = date(2026, 10, 17)
    assert saturday.weekday() == 5
    assert arrivals_for(saturday, base=10) < arrivals_for(date(2026, 10, 13), base=10)
    assert clearing_chance(KHAN, at(15, weekday=5)) < clearing_chance(KHAN, at(15, weekday=1))


def test_a_day_always_brings_at_least_one_report():
    for offset in range(14):
        day = date(2026, 10, 12) + timedelta(days=offset)
        assert arrivals_for(day, base=1) >= 1


def test_arrivals_for_a_given_day_are_stable():
    day = date(2026, 10, 14)
    assert arrivals_for(day) == arrivals_for(day)


# ------------------------------------------------------------- the rolling


def test_should_clear_is_never_true_outside_the_window():
    rng = random.Random(0)
    assert not any(should_clear(KHAN, at(3), rng) for _ in range(200))


def test_should_clear_fires_sometimes_inside_the_window():
    rng = random.Random(7)
    fired = sum(should_clear(KHAN, at(15), rng) for _ in range(200))
    assert 0 < fired < 200, "a rate of 0.45 should neither never nor always fire"


def test_the_same_seed_gives_the_same_ward():
    a = random.Random(42)
    b = random.Random(42)
    rolls_a = [should_clear(KHAN, at(15), a) for _ in range(50)]
    rolls_b = [should_clear(KHAN, at(15), b) for _ in range(50)]
    assert rolls_a == rolls_b


# -------------------------------------------------------------- the reports


def test_a_report_carries_several_tests_and_a_patient_reference():
    lines, names = report_lines(date(2026, 10, 14), 1, random.Random(1))
    body = "\n".join(lines)
    assert "MUZAFFAR KHAN SURGICAL HOSPITAL" in body
    assert "Patient ref: MK-" in body
    assert 2 <= len(names) <= 4


def test_reports_differ_from_one_another():
    rng = random.Random(3)
    first, _ = report_lines(date(2026, 10, 14), 1, rng)
    second, _ = report_lines(date(2026, 10, 14), 2, rng)
    assert first != second


def test_a_report_carries_values_for_intake_to_strip():
    """The reports must be realistic enough to exercise the boundary."""
    lines, _ = report_lines(date(2026, 10, 14), 1, random.Random(5))
    body = "\n".join(lines)
    assert any(unit in body for unit in ("mmol/L", "g/dL", "umol/L", "INR", "ng/mL"))


def test_values_are_paired_to_their_own_test():
    """A report nobody clinical would recognise undermines the demo."""
    from ward_sim import VALUES_BY_TEST

    assert "mIU/L" in " ".join(VALUES_BY_TEST["Thyroid panel"])
    assert "ng/L" in " ".join(VALUES_BY_TEST["Vitamin B12"])
    assert not any("INR" in v for v in VALUES_BY_TEST["Vitamin B12"])
    assert all(VALUES_BY_TEST.get(name) for name, _ in ward_sim.TESTS), \
        "every test in the pool needs its own plausible values"


# ------------------------------------------------------- the ground truth


def test_profiles_are_written_for_marking_not_for_reading(tmp_path):
    path = write_profiles(tmp_path / "ward_profiles.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert "MUST NOT read" in data["_comment"]
    assert set(data["clinicians"]) == set(PROFILES)
    assert data["clinicians"]["Dr. Khan"]["window"] == [14, 17]
    assert data["weekday_arrivals"]["0"] > data["weekday_arrivals"]["1"]


def test_the_analytics_never_import_the_ground_truth():
    """A guard against the one mistake that would void the whole claim."""
    from pathlib import Path

    insights = Path(__file__).parent / "insights.py"
    if not insights.exists():
        pytest.skip("insights.py not written yet")
    source = insights.read_text(encoding="utf-8")
    assert "ward_profiles" not in source
    assert "ward_sim" not in source
    assert "PROFILES" not in source


def test_every_profile_has_a_plausible_window():
    for name, profile in PROFILES.items():
        start, end = profile.window
        assert 0 <= start < end <= 24, name
        assert 0 < profile.rate <= 1, name
        assert profile.note, f"{name} should say why it works this way"


def test_the_roles_are_ones_the_workflow_knows():
    from pathlib import Path

    from carechaser_engine import load_workflow

    workflow = load_workflow(Path(__file__).parent / "workflow.yaml")
    known = set(workflow["roles"])
    for profile in PROFILES.values():
        assert profile.role in known, profile.role


def test_the_status_is_a_second_column_not_a_second_row():
    """Printed as one long line it runs off the page and Textract reads the
    truncation; printed below, the status arrives detached from its test."""
    lines, _ = report_lines(date(2026, 10, 14), 1, random.Random(2))
    test_lines = [line for line in lines if "\t" in line]
    assert test_lines, "every test line needs a column break"
    for line in test_lines:
        left, _, right = line.partition("\t")
        assert "." * 3 in left, "the left column leads with dots to its value"
        assert right.strip(), "the right column carries the status"
        assert len(left) <= 52, f"left column too wide for the page: {left!r}"
        assert len(right) <= 38, f"right column too wide for the page: {right!r}"


def test_a_rendered_report_survives_rejoining_and_intake(tmp_path):
    """The whole input path, with Textract's block shapes simulated."""
    from pathlib import Path as _Path

    from carechaser_engine import load_workflow
    from intake import RuleExtractor, extract_report
    from scan_intake import rows_from_blocks

    lines, names = report_lines(date(2026, 10, 14), 1, random.Random(4))

    blocks, top = [], 0.08
    for line in lines:
        if not line.strip():
            top += 0.022
            continue
        left, tab, right = line.partition("\t")
        blocks.append({"BlockType": "LINE", "Text": left.strip(), "Geometry": {
            "BoundingBox": {"Top": top, "Left": 0.07, "Height": 0.018, "Width": 0.5}}})
        if tab:
            blocks.append({"BlockType": "LINE", "Text": right.strip(), "Geometry": {
                "BoundingBox": {"Top": top + 0.0005, "Left": 0.60, "Height": 0.018,
                                "Width": 0.3}}})
        top += 0.022

    text = "\n".join(rows_from_blocks(blocks))
    workflow = load_workflow(_Path(__file__).parent / "workflow.yaml")
    result = extract_report(text, workflow, RuleExtractor())

    staged = {r["test_name"]: r["state"] for r in result["rows"] if r["state"]}
    assert staged, "at least one test should reach a stage"
    assert set(staged) <= set(names)
    assert result["patient"].startswith("MK-")
    for name in staged:
        assert not any(u in name for u in ("mmol", "ng/L", "mIU", "kPa", "umol"))


def test_a_critical_test_is_sometimes_flagged_stat():
    """Urgency has to reach the engine, or the ten-minute SLA never fires."""
    flagged = 0
    for seed in range(60):
        lines, _ = report_lines(date(2026, 10, 14), seed, random.Random(seed))
        if any("STAT" in line for line in lines):
            flagged += 1
    assert 0 < flagged < 60, "some reports carry an urgent flag, not all of them"


def test_only_critical_tests_are_ever_flagged():
    critical = {name for name, urgency in ward_sim.TESTS if urgency == "critical"}
    for seed in range(40):
        lines, _ = report_lines(date(2026, 10, 14), seed, random.Random(seed))
        for line in lines:
            if "STAT" in line:
                name = line.split("...")[0].strip()
                assert name in critical, f"{name} is not a critical test"
