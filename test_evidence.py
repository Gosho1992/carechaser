"""Tests for the run's evidence.

Two halves, matching the module: the ingest totals, and the marking of
inferred windows against the generator's real ones.

The marking is the part that needs testing hardest, because it is the part a
judge will trust. A scorer that is generous with itself is worse than no
scorer at all, so most of what follows checks that it *fails* when it should:
an invented working hour, a thin sample, a person with no ground truth.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import evidence
import insights
from carechaser_engine import load_workflow
from evidence import (
    SCORE_AT,
    Truth,
    build,
    hours_in,
    load_truth,
    mark_people,
    mark_weekday,
    run_totals,
    score_window,
)

WORKFLOW = load_workflow(Path(__file__).parent / "workflow.yaml")
KARACHI = timezone(timedelta(hours=5))


def truth(**windows) -> Truth:
    """Ground truth for the named people, Monday loaded up like the real one."""
    return Truth(
        windows={name: tuple(window) for name, window in windows.items()},
        rates={name: 0.4 for name in windows},
        notes={},
        weekday_arrivals={0: 1.7, 1: 1.0, 2: 1.0, 3: 1.1, 4: 1.2, 5: 0.4, 6: 0.3},
        weekday_clearing={0: 0.55, 1: 1.0, 2: 1.0, 3: 0.95, 4: 0.85, 5: 0.3, 6: 0.2},
    )


def person(name, window, n) -> dict:
    """An insights row, as person_patterns() would return it."""
    return {"person": name, "window": list(window) if window else None, "n": n}


# -------------------------------------------------------------- the hours


def test_a_window_is_the_hours_it_covers():
    assert hours_in([14, 17]) == {14, 15, 16}


def test_a_window_across_midnight_is_not_empty():
    """A night shift scored as zero hours would make every verdict wrong."""
    assert hours_in([22, 2]) == {22, 23, 0, 1}


def test_a_full_day_window_covers_every_hour():
    assert len(hours_in([0, 24])) == 24


# ------------------------------------------------------------- the marking


def test_a_window_inside_the_truth_is_clean():
    score = score_window([15, 17], [14, 18])
    assert score["contained"] is True
    assert score["false_hours"] == []


def test_partial_recovery_is_reported_as_partial_not_wrong():
    """Few observations recover a sub-interval. That is a sample size
    statement, not an error, and the two must not be conflated."""
    score = score_window([15, 17], [14, 18])
    assert score["contained"] is True
    assert score["coverage"] == 0.5          # 2 hours of 4


def test_an_invented_working_hour_fails_containment():
    """The error that matters: a chase sent at an hour nobody works."""
    score = score_window([13, 17], [14, 18])
    assert score["contained"] is False
    assert score["false_hours"] == [13]


def test_exact_recovery_is_full_coverage():
    score = score_window([14, 17], [14, 17])
    assert score["contained"] is True
    assert score["coverage"] == 1.0


def test_nothing_inferred_is_not_scored_as_clean():
    """Absence of a claim must never read as a correct claim."""
    score = score_window(None, [14, 17])
    assert score["contained"] is None
    assert score["coverage"] == 0.0


# ------------------------------------------------------------- the people


def test_a_person_with_enough_observations_is_scored():
    rows = mark_people(
        [person("Dr. Khan", (15, 17), SCORE_AT)],
        truth(**{"Dr. Khan": (14, 17)}),
    )
    assert rows[0]["scored"] is True
    assert rows[0]["contained"] is True


def test_a_thin_sample_is_reported_but_not_scored():
    rows = mark_people(
        [person("Dr. Khan", (15, 16), SCORE_AT - 1)],
        truth(**{"Dr. Khan": (14, 17)}),
    )
    assert rows[0]["scored"] is False
    assert rows[0]["inferred_window"] == [15, 16]


def test_a_clinician_who_has_cleared_nothing_still_appears():
    """Dropping them would quietly shrink the denominator."""
    rows = mark_people([], truth(**{"Dr. Ali": (15, 18)}))
    assert len(rows) == 1
    assert rows[0]["inferred_window"] is None
    assert rows[0]["n"] == 0


def test_someone_outside_the_profiles_is_listed_without_a_verdict():
    """A human pressing buttons during a demo has no ground truth."""
    rows = mark_people(
        [person("Judge", (9, 10), 20)], truth(**{"Dr. Khan": (14, 17)}),
    )
    judge = next(row for row in rows if row["person"] == "Judge")
    assert judge["actual_window"] is None
    assert judge["scored"] is False
    assert judge["contained"] is None


# ------------------------------------------------------------ the weekday


def test_the_two_weekday_parameters_are_reported_separately():
    """The generator receives most on Monday and clears least on Sunday.
    Different days, answering different questions."""
    facts = truth(**{"Dr. Khan": (14, 17)})
    assert facts.busiest_weekday == 0          # Monday, 1.7x arrivals
    assert facts.slowest_weekday == 6          # Sunday, 0.2x clearing


def test_breaches_are_marked_against_clearing_not_arrivals():
    """A breach is one stage over its limit. Arrivals change how many cases
    exist; the clearing rate changes how long each waits. Marking the
    breach count against arrivals made a correct inference read as a miss."""
    facts = truth(**{"Dr. Khan": (14, 17)})
    assert facts.hardest_weekday == facts.slowest_weekday
    assert facts.hardest_weekday != facts.busiest_weekday


def test_the_worst_observed_weekday_is_marked_against_clearing():
    weekdays = [
        {"weekday": 6, "name": "Sunday", "n": 19, "breached": 17,
         "breach_rate": 0.89, "confident": True},
        {"weekday": 1, "name": "Tuesday", "n": 20, "breached": 2,
         "breach_rate": 0.1, "confident": True},
    ]
    marked = mark_weekday(weekdays, truth(**{"Dr. Khan": (14, 17)}))
    assert marked["observed_name"] == "Sunday"
    assert marked["expected_name"] == "Sunday"
    assert marked["correct"] is True


def test_both_generator_parameters_appear_in_the_result():
    """Changing which parameter you compare against after seeing the answer
    is how a check gets rigged. Both are reported, always."""
    marked = mark_weekday(
        [{"weekday": 6, "name": "Sunday", "n": 19, "breached": 17,
          "breach_rate": 0.89, "confident": True}],
        truth(**{"Dr. Khan": (14, 17)}),
    )
    assert marked["expected_basis"] == "lowest clearing rate"
    assert marked["busiest_name"] == "Monday"
    assert marked["arrival_rate"] == 1.7
    assert marked["clearing_rate"] == 0.2


def test_a_thin_week_is_not_scored_either_way():
    weekdays = [{"weekday": 3, "name": "Thursday", "n": 2, "breached": 2,
                 "breach_rate": 1.0, "confident": False}]
    marked = mark_weekday(weekdays, truth(**{"Dr. Khan": (14, 17)}))
    assert marked["scored"] is False
    assert marked["correct"] is None


# ------------------------------------------------------------- the totals


def test_the_ingest_totals_add_up(tmp_path):
    log = tmp_path / "ingest_log.jsonl"
    log.write_text("\n".join(json.dumps(day) for day in [
        {"day": "2026-10-14", "reports": 11, "rows_proposed": 67,
         "rows_accepted": 34, "rows_skipped": 33, "cases_created": 34,
         "clinical_fragments_removed": 56, "unreadable_files": []},
        {"day": "2026-10-15", "reports": 9, "rows_proposed": 50,
         "rows_accepted": 30, "rows_skipped": 20, "cases_created": 30,
         "clinical_fragments_removed": 40, "unreadable_files": ["r7.pdf"]},
    ]) + "\n", encoding="utf-8")

    totals = run_totals(log)
    assert totals["days"] == 2
    assert totals["reports"] == 20
    assert totals["rows_proposed"] == 117
    assert totals["rows_accepted"] == 64
    assert totals["first_day"] == "2026-10-14"
    assert totals["last_day"] == "2026-10-15"


def test_a_failure_stays_in_the_denominator(tmp_path):
    """The whole point of recording the run is not quietly dropping the
    reports that did not work."""
    log = tmp_path / "ingest_log.jsonl"
    log.write_text(json.dumps({
        "day": "2026-10-16", "reports": 10, "rows_proposed": 60,
        "rows_accepted": 30, "rows_skipped": 30, "cases_created": 30,
        "clinical_fragments_removed": 1, "unreadable_files": ["a.pdf", "b.pdf"],
    }) + "\n", encoding="utf-8")

    totals = run_totals(log)
    assert totals["unreadable_count"] == 2
    assert totals["reports"] == 10           # not 8
    assert totals["accept_rate"] == 0.5


def test_a_corrupt_line_does_not_lose_the_rest(tmp_path):
    log = tmp_path / "ingest_log.jsonl"
    log.write_text(
        '{"day": "2026-10-14", "reports": 5}\nnot json at all\n'
        '{"day": "2026-10-15", "reports": 5}\n', encoding="utf-8")
    assert run_totals(log)["reports"] == 10


def test_no_log_yet_is_zeroes_not_a_crash(tmp_path):
    totals = run_totals(tmp_path / "missing.jsonl")
    assert totals["days"] == 0
    assert totals["accept_rate"] == 0.0


# ------------------------------------------------------- reading the truth


def test_the_profiles_file_is_read_as_written(tmp_path):
    """Written by ward_sim.write_profiles(), read back here."""
    path = tmp_path / "ward_profiles.json"
    path.write_text(json.dumps({
        "clinicians": {
            "Dr. Khan": {"name": "Dr. Khan", "role": "responsible_clinician",
                         "window": [14, 17], "rate": 0.45, "note": "theatre"},
        },
        "weekday_arrivals": {"0": 1.7, "1": 1.0},
        "weekday_clearing": {"0": 0.55, "1": 1.0},
    }), encoding="utf-8")

    loaded = load_truth(path)
    assert loaded.windows == {"Dr. Khan": (14, 17)}
    assert loaded.rates["Dr. Khan"] == 0.45
    assert loaded.hardest_weekday == 0


def test_no_profiles_file_is_not_an_error(tmp_path):
    """On a real ward there is no ground truth, and only the totals apply."""
    assert load_truth(tmp_path / "nothing.json") is None


def test_the_real_profiles_round_trip_through_the_marker(tmp_path):
    """The file ward_sim actually writes must be markable, not just a
    hand-made fixture that happens to fit."""
    import ward_sim

    path = ward_sim.write_profiles(tmp_path / "ward_profiles.json")
    loaded = load_truth(path)
    assert loaded is not None
    assert set(loaded.windows) == set(ward_sim.PROFILES)

    rows = mark_people([person("Dr. Khan", (15, 17), SCORE_AT)], loaded)
    khan = next(row for row in rows if row["person"] == "Dr. Khan")
    assert khan["contained"] is True
    assert khan["actual_window"] == list(ward_sim.PROFILES["Dr. Khan"].window)


# ---------------------------------------------------------- the whole thing


def test_with_no_truth_the_headline_says_so(tmp_path):
    result = build([], WORKFLOW, truth=None, ingest_log=tmp_path / "none.jsonl")
    assert result.has_truth is False
    assert "nothing to mark" in result.headline.lower()


def test_an_early_run_refuses_to_claim_a_score(tmp_path):
    result = build([], WORKFLOW, truth=truth(**{"Dr. Khan": (14, 17)}),
                   ingest_log=tmp_path / "none.jsonl")
    assert "too early" in result.headline.lower()
    assert str(SCORE_AT) in result.headline


def test_the_result_serialises_for_the_api(tmp_path):
    payload = build([], WORKFLOW, truth=truth(**{"Dr. Khan": (14, 17)}),
                    ingest_log=tmp_path / "none.jsonl").to_dict()
    assert set(payload) >= {
        "generated_at", "run", "has_truth", "people", "weekday",
        "headline", "score_at",
    }
    json.dumps(payload)                      # must be JSON-safe end to end


def test_the_marker_is_allowed_to_read_the_truth_but_insights_is_not():
    """Stated once, in the place someone would check it: the asymmetry is
    the whole design. evidence.py marks; insights.py infers blind."""
    marker = Path(evidence.__file__).read_text(encoding="utf-8")
    assert "ward_profiles" in marker

    body = Path(insights.__file__).read_text(encoding="utf-8").split('"""', 2)[-1]
    assert "ward_profiles" not in body


def test_a_quiet_day_cannot_hold_the_weekday_verdict():
    """A Saturday with 2 late of 12 must not outrank a Monday with 60 behind
    it: the claim would rest on the smallest sample in the week."""
    weekdays = [
        {"weekday": 0, "name": "Monday", "n": 60, "breached": 18,
         "breach_rate": 0.3, "confident": True},
        {"weekday": 5, "name": "Saturday", "n": 3, "breached": 2,
         "breach_rate": 0.667, "confident": False},
    ]
    marked = mark_weekday(weekdays, truth(**{"Dr. Khan": (14, 17)}))
    assert marked["observed_name"] == "Monday"


def test_asking_for_an_unmarked_report_does_not_silently_mark_it(tmp_path, monkeypatch):
    """truth=None is the real-ward case. If it fell through to a disk lookup,
    a caller asking for an unmarked report would get a marked one."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ward_profiles.json").write_text(json.dumps({
        "clinicians": {"Dr. Khan": {"window": [14, 17], "rate": 0.4}},
    }), encoding="utf-8")
    result = build([], WORKFLOW, truth=None, ingest_log=tmp_path / "none.jsonl")
    assert result.has_truth is False
    assert result.people == []


# ------------------------------------------------------ why rows were cut


def test_skip_reasons_are_totalled_from_the_live_log(tmp_path):
    log = tmp_path / "ingest_log.jsonl"
    log.write_text("\n".join(json.dumps(day) for day in [
        {"day": "2026-10-18", "reports": 10, "rows_proposed": 60,
         "rows_accepted": 30, "rows_skipped": 30, "cases_created": 30,
         "clinical_fragments_removed": 40, "unreadable_files": [],
         "skip_reasons": {"status_not_recognised": 20, "no_status_printed": 10}},
        {"day": "2026-10-19", "reports": 8, "rows_proposed": 40,
         "rows_accepted": 25, "rows_skipped": 15, "cases_created": 25,
         "clinical_fragments_removed": 30, "unreadable_files": [],
         "skip_reasons": {"status_not_recognised": 15}},
    ]) + "\n", encoding="utf-8")

    totals = run_totals(log)
    assert totals["skip_reasons"] == {"status_not_recognised": 35,
                                      "no_status_printed": 10}
    assert totals["skip_reasons_days"] == 2


def test_earlier_days_are_filled_in_from_the_replay_file(tmp_path):
    """The first days of the run recorded only a count. The breakdown is
    recovered by re-reading the PDFs and kept in a separate file."""
    log = tmp_path / "ingest_log.jsonl"
    log.write_text(json.dumps({
        "day": "2026-10-18", "reports": 10, "rows_proposed": 60,
        "rows_accepted": 30, "rows_skipped": 30, "cases_created": 30,
        "clinical_fragments_removed": 40, "unreadable_files": [],
        "skip_reasons": {"status_not_recognised": 30},
    }) + "\n", encoding="utf-8")
    (tmp_path / "skip_reasons.json").write_text(json.dumps({
        "days": {"2026-10-14": {"source": "replay",
                                "reasons": {"no_status_printed": 12}}},
    }), encoding="utf-8")

    totals = run_totals(log)
    assert totals["skip_reasons"] == {"status_not_recognised": 30,
                                      "no_status_printed": 12}
    assert totals["skip_reasons_days"] == 2


def test_a_live_figure_is_never_overwritten_by_a_re_read(tmp_path):
    """The log is what happened; a later re-read is an approximation of it."""
    log = tmp_path / "ingest_log.jsonl"
    log.write_text(json.dumps({
        "day": "2026-10-18", "reports": 1, "rows_proposed": 10,
        "rows_accepted": 4, "rows_skipped": 6, "cases_created": 4,
        "clinical_fragments_removed": 1, "unreadable_files": [],
        "skip_reasons": {"no_status_printed": 6},
    }) + "\n", encoding="utf-8")
    (tmp_path / "skip_reasons.json").write_text(json.dumps({
        "days": {"2026-10-18": {"source": "replay",
                                "reasons": {"no_status_printed": 999}}},
    }), encoding="utf-8")

    assert run_totals(log)["skip_reasons"] == {"no_status_printed": 6}


def test_no_breakdown_anywhere_is_empty_not_invented(tmp_path):
    log = tmp_path / "ingest_log.jsonl"
    log.write_text(json.dumps({
        "day": "2026-10-14", "reports": 5, "rows_proposed": 30,
        "rows_accepted": 15, "rows_skipped": 15, "cases_created": 15,
        "clinical_fragments_removed": 9, "unreadable_files": [],
    }) + "\n", encoding="utf-8")
    totals = run_totals(log)
    assert totals["skip_reasons"] == {}
    assert totals["rows_set_aside"] == 15


def test_the_accept_rate_is_measured_against_test_rows(tmp_path):
    """Letterhead and signature lines are neither extracted nor failed.
    Counting them as failures reported a flawless run as 51% accurate."""
    log = tmp_path / "ingest_log.jsonl"
    log.write_text(json.dumps({
        "day": "2026-10-18", "reports": 10, "rows_proposed": 60,
        "rows_metadata": 30, "rows_tests": 30, "rows_accepted": 30,
        "rows_skipped": 30, "cases_created": 30,
        "clinical_fragments_removed": 40, "unreadable_files": [],
        "skip_reasons": {"not_a_test_row": 30},
    }) + "\n", encoding="utf-8")

    totals = run_totals(log)
    assert totals["rows_tests"] == 30
    assert totals["accept_rate"] == 1.0
    assert totals["accept_rate_basis"] == "test rows"


def test_days_recorded_before_metadata_was_counted_still_report(tmp_path):
    """The first days of the run have no metadata field. They must fall
    back to the old denominator rather than dividing by zero or implying
    a precision they do not have."""
    log = tmp_path / "ingest_log.jsonl"
    log.write_text(json.dumps({
        "day": "2026-10-14", "reports": 11, "rows_proposed": 67,
        "rows_accepted": 34, "rows_skipped": 33, "cases_created": 34,
        "clinical_fragments_removed": 56, "unreadable_files": [],
    }) + "\n", encoding="utf-8")

    totals = run_totals(log)
    assert totals["rows_tests"] == 67
    assert totals["accept_rate"] == round(34 / 67, 3)
    assert totals["accept_rate_basis"] == "all rows read"


def test_early_days_are_recounted_from_the_replay(tmp_path):
    """Days 14-17 ran before metadata was counted, but their PDFs were
    re-read afterwards. Using that is a recomputation over the same
    documents, so the rate is corrected and the basis says so."""
    log = tmp_path / "ingest_log.jsonl"
    log.write_text(json.dumps({
        "day": "2026-10-14", "reports": 11, "rows_proposed": 67,
        "rows_accepted": 34, "rows_skipped": 33, "cases_created": 34,
        "clinical_fragments_removed": 56, "unreadable_files": [],
    }) + "\n", encoding="utf-8")
    (tmp_path / "skip_reasons.json").write_text(json.dumps({
        "days": {"2026-10-14": {"source": "replay",
                                "reasons": {"not_a_test_row": 33}}},
    }), encoding="utf-8")

    totals = run_totals(log)
    assert totals["rows_metadata"] == 33
    assert totals["rows_tests"] == 34
    assert totals["accept_rate"] == 1.0
    assert "recounted" in totals["accept_rate_basis"]
    assert totals["recounted_days"] == ["2026-10-14"]


def test_a_live_metadata_count_is_not_double_counted(tmp_path):
    """A day with the figure recorded live must not also take it from the
    replay file, or the denominator shrinks twice."""
    log = tmp_path / "ingest_log.jsonl"
    log.write_text(json.dumps({
        "day": "2026-10-18", "reports": 10, "rows_proposed": 60,
        "rows_metadata": 30, "rows_tests": 30, "rows_accepted": 30,
        "rows_skipped": 30, "cases_created": 30,
        "clinical_fragments_removed": 40, "unreadable_files": [],
    }) + "\n", encoding="utf-8")
    (tmp_path / "skip_reasons.json").write_text(json.dumps({
        "days": {"2026-10-18": {"reasons": {"not_a_test_row": 30}}},
    }), encoding="utf-8")

    totals = run_totals(log)
    assert totals["rows_metadata"] == 30        # not 60
    assert totals["recounted_days"] == []


def test_a_run_that_changed_mid_way_counts_every_day(tmp_path):
    """The real failure. Days after the fix record rows_tests; days before
    it do not. Summing the field alone drops the earlier days from the
    denominator and reports 255 accepted out of 145 — 176%."""
    log = tmp_path / "ingest_log.jsonl"
    log.write_text("\n".join(json.dumps(day) for day in [
        # Before the fix: no rows_tests, no rows_metadata.
        {"day": "2026-10-14", "reports": 11, "rows_proposed": 67,
         "rows_accepted": 34, "rows_skipped": 33, "cases_created": 34,
         "clinical_fragments_removed": 56, "unreadable_files": []},
        # After the fix: both recorded live.
        {"day": "2026-10-18", "reports": 10, "rows_proposed": 60,
         "rows_metadata": 30, "rows_tests": 30, "rows_accepted": 30,
         "rows_skipped": 30, "cases_created": 30,
         "clinical_fragments_removed": 40, "unreadable_files": []},
    ]) + "\n", encoding="utf-8")
    (tmp_path / "skip_reasons.json").write_text(json.dumps({
        "days": {"2026-10-14": {"reasons": {"not_a_test_row": 33}}},
    }), encoding="utf-8")

    totals = run_totals(log)
    assert totals["rows_tests"] == 64          # 34 earlier + 30 later
    assert totals["rows_accepted"] == 64
    assert totals["accept_rate"] == 1.0


def test_the_accept_rate_can_never_exceed_one(tmp_path):
    """A rate above 100% means the denominator lost a day. Nothing should
    be able to produce one."""
    log = tmp_path / "ingest_log.jsonl"
    log.write_text("\n".join(json.dumps(day) for day in [
        {"day": "2026-10-14", "reports": 5, "rows_proposed": 40,
         "rows_accepted": 20, "rows_skipped": 20, "cases_created": 20,
         "clinical_fragments_removed": 10, "unreadable_files": []},
        {"day": "2026-10-15", "reports": 5, "rows_proposed": 40,
         "rows_metadata": 20, "rows_tests": 20, "rows_accepted": 20,
         "rows_skipped": 20, "cases_created": 20,
         "clinical_fragments_removed": 10, "unreadable_files": []},
        {"day": "2026-10-16", "reports": 5, "rows_proposed": 40,
         "rows_accepted": 20, "rows_skipped": 20, "cases_created": 20,
         "clinical_fragments_removed": 10, "unreadable_files": []},
    ]) + "\n", encoding="utf-8")
    assert run_totals(log)["accept_rate"] <= 1.0


def test_metadata_and_test_rows_account_for_every_row_read(tmp_path):
    """518 proposed = 261 page furniture + 257 test rows. If the two sides
    stop adding up, one of them is being counted on the wrong basis."""
    log = tmp_path / "ingest_log.jsonl"
    log.write_text(json.dumps({
        "day": "2026-10-18", "reports": 10, "rows_proposed": 60,
        "rows_metadata": 30, "rows_tests": 30, "rows_accepted": 29,
        "rows_skipped": 31, "cases_created": 29,
        "clinical_fragments_removed": 40, "unreadable_files": [],
    }) + "\n", encoding="utf-8")

    totals = run_totals(log)
    assert totals["rows_metadata"] + totals["rows_tests"] == totals["rows_proposed"]
