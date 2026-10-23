"""Tests for the morning job's record of what it did.

Mostly about one number: roughly half of all OCR'd rows are set aside for a
human, and the count alone invites the question "why". These check the
breakdown that answers it, and that the buckets do not quietly absorb a
reason they were not written for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daily_ingest import SKIP_REASONS, _reason_counts, classify_skip


@pytest.mark.parametrize("why,bucket", [
    ("duplicate test on this report", "duplicate_on_report"),
    ("no status given", "no_status_printed"),
    ("status 'reviewd pendng' is not one this workflow knows",
     "status_not_recognised"),
    ("'closed' is not a stage a case can start in", "stage_cannot_start"),
    ("stage 'ordered' has no valid owner role configured",
     "no_owner_role_configured"),
])
def test_each_reason_intake_writes_has_a_bucket(why, bucket):
    assert classify_skip(why) == bucket


def test_an_unfamiliar_reason_gets_its_own_bucket():
    """Folding it into an existing category would report a reason that was
    never given, which is worse than reporting 'other'."""
    assert classify_skip("something nobody has written yet") == "other"
    assert classify_skip("") == "other"
    assert classify_skip(None) == "other"


def test_the_offending_text_never_becomes_the_bucket_name():
    """Intake puts the OCR'd text inside the sentence. Counting raw strings
    would give every row its own category and no usable total."""
    a = classify_skip("status 'reviewd, pendng' is not one this workflow knows")
    b = classify_skip("status 'awaiting revew' is not one this workflow knows")
    assert a == b == "status_not_recognised"


def test_counts_add_up_across_files():
    results = [
        {"skipped": [{"reason": "no_status_printed"},
                     {"reason": "status_not_recognised"}]},
        {"skipped": [{"reason": "status_not_recognised"}]},
        {"skipped": []},
        {},
    ]
    assert _reason_counts(results) == {"status_not_recognised": 2,
                                       "no_status_printed": 1}


def test_counts_are_ordered_largest_first():
    results = [{"skipped": [{"reason": "a"}, {"reason": "b"}, {"reason": "b"}]}]
    assert list(_reason_counts(results)) == ["b", "a"]


def test_an_older_row_without_a_bucket_is_classified_on_the_fly():
    """Rows recorded before the bucket field existed still carry `why`."""
    results = [{"skipped": [{"why": "no status given"}]}]
    assert _reason_counts(results) == {"no_status_printed": 1}


def test_the_buckets_match_what_intake_actually_writes():
    """The one test here that is not circular.

    Matching the needles against themselves proves nothing: if intake
    rewords an issue, the needle still matches itself and the rows quietly
    become "other". So this runs the real extractor over a report built to
    trip every failure, and asserts that nothing it produces falls through.
    """
    from carechaser_engine import load_workflow
    from intake import RuleExtractor, extract_report

    workflow = load_workflow(Path(__file__).parent / "workflow.yaml")
    report = """MUZAFFAR KHAN SURGICAL HOSPITAL
Patient ref: MK-4471

Serum potassium .......... released, not yet acknowledged
Serum potassium .......... released, not yet acknowledged
Full blood count ......... banana status here
Liver function ...........
Chest imaging ............ workflow closed
"""
    rows = extract_report(report, workflow, RuleExtractor())["rows"]
    issues = [row["issues"][0] for row in rows if not row["ok"]]
    assert issues, "the fixture stopped producing failures"

    for issue in issues:
        assert classify_skip(issue) != "other", (
            f"intake writes {issue!r} and no bucket matches it — "
            f"the wording drifted"
        )
    # And the three distinct failure shapes land in three distinct buckets.
    assert len({classify_skip(i) for i in issues}) >= 3
