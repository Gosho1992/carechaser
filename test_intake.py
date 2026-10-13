"""Tests for report intake.

The important ones are the value-stripping tests: they prove CareChaser
cannot carry a clinical measurement from a report into the workflow, even
if an extractor hands one over.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carechaser_engine import load_workflow
from intake import (
    RuleExtractor,
    extract_report,
    strip_values,
    validate,
)

ROOT = Path(__file__).parent
WORKFLOW = load_workflow(ROOT / "workflow.yaml")

REPORT = """Musafah Surgical Hospital
Patient MK-4471

CBC — completed
Liver function — completed
Serum potassium — result available
Chest X-Ray — not completed
"""


class StubExtractor:
    name = "bedrock"

    def __init__(self, payload):
        self.payload = payload

    def extract(self, text):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


# ------------------------------------------------------- the safety boundary


@pytest.mark.parametrize("dirty,clean", [
    # with units
    ("Serum potassium 6.8 mmol/L", "Serum potassium"),
    ("Haemoglobin 9.2 g/dL", "Haemoglobin"),
    ("6.8 mmol/L", ""),
    # bare numbers, the case that used to slip through
    ("Potassium 6.8", "Potassium"),
    ("Glucose 120", "Glucose"),
    ("Troponin 0.8", "Troponin"),
    ("CBC result 6.8", "CBC"),
    ("Creatinine: 2.4", "Creatinine"),
    # ranges, with and without units
    ("Sodium 135-145", "Sodium"),
    ("Sodium 135 - 145", "Sodium"),
    ("Sodium 135-145 mmol/L", "Sodium"),
    # clinical judgements
    ("Potassium HIGH", "Potassium"),
    ("TSH elevated", "TSH"),
    ("Culture positive", "Culture"),
    ("Glucose abnormal", "Glucose"),
])
def test_measurements_and_judgements_are_stripped(dirty, clean):
    assert strip_values(dirty) == clean


@pytest.mark.parametrize("name", [
    "Vitamin B12", "HbA1c", "T3", "T4", "COVID-19 PCR",
    "25-OH Vitamin D", "24 hour urine protein", "Chest X-Ray", "Liver function",
])
def test_real_test_names_with_numbers_survive(name):
    """Numbers bound into a name are part of the name, not a result."""
    assert strip_values(name) == name


@pytest.mark.parametrize("raw", [
    "Potassium 6.8", "Glucose 120", "Troponin 0.8", "Sodium 135-145",
    "Serum potassium 6.8 mmol/L", "Haemoglobin 9.2 g/dL",
])
def test_no_result_digits_survive_into_the_fields_that_become_a_case(raw):
    payload = {"patient": "P-1", "items": [{"test_name": raw, "status": "completed"}]}
    row = validate(payload, WORKFLOW)[0]
    for digit in ["6.8", "120", "0.8", "135", "145", "9.2"]:
        assert digit not in row.test_name
        assert digit not in row.status_text


def test_a_line_that_is_only_a_value_is_dropped_entirely():
    """"Result: 6.8" has no test name once the value is removed, so there is
    nothing to track and the row disappears rather than becoming a case."""
    payload = {"patient": "P-1", "items": [{"test_name": "Result: 6.8", "status": "completed"}]}
    assert validate(payload, WORKFLOW) == []


def test_a_result_value_never_becomes_case_data():
    """A value may be shown to the reviewer as 'removed', but it is never
    part of the data that becomes a case: not the name, not the status."""
    payload = {"patient": "P-1", "items": [
        {"test_name": "Serum potassium 6.8 mmol/L", "status": "result available"},
    ]}
    row = validate(payload, WORKFLOW)[0]
    assert row.test_name == "Serum potassium"
    assert "6.8" not in row.test_name
    assert "6.8" not in row.status_text
    assert "6.8" in row.removed                 # visible to the human, deliberately


def test_abnormal_language_does_not_become_urgency():
    payload = {"patient": "P-1", "items": [
        {"test_name": "Potassium", "status": "result available abnormal"},
    ]}
    row = validate(payload, WORKFLOW)[0]
    assert "abnormal" not in row.test_name.lower()
    assert "abnormal" not in row.status_text.lower()
    assert "abnormal" in [r.lower() for r in row.removed]


# ----------------------------------------------------------- the happy path


def test_a_report_becomes_reviewable_rows():
    result = extract_report(REPORT, WORKFLOW, RuleExtractor())
    assert result["ok"] is True
    assert result["total"] == 4
    assert result["patient"] == "MK-4471"
    names = [row["test_name"] for row in result["rows"]]
    assert names == ["CBC", "Liver function", "Serum potassium", "Chest X-Ray"]


def test_done_and_pending_land_in_different_stages():
    rows = {r["test_name"]: r for r in extract_report(REPORT, WORKFLOW, RuleExtractor())["rows"]}
    assert rows["CBC"]["state"] == "result_produced"             # done, awaiting review
    assert rows["Chest X-Ray"]["state"] == "sample_processed"    # not done yet
    assert rows["Serum potassium"]["state"] == "result_produced"


def test_owner_roles_come_from_the_workflow_not_the_report():
    rows = extract_report(REPORT, WORKFLOW, RuleExtractor())["rows"]
    roles = set(WORKFLOW["roles"])
    for row in rows:
        assert row["owner_role"] in roles


def test_csv_reports_are_accepted():
    csv_text = "test,status\nCBC,completed\nUrine culture,pending\n"
    result = extract_report(csv_text, WORKFLOW, RuleExtractor())
    assert result["total"] == 2
    assert result["rows"][1]["state"] == "sample_processed"


def test_nothing_is_created_by_extraction():
    """Extraction returns rows for review and writes nothing anywhere."""
    result = extract_report(REPORT, WORKFLOW, RuleExtractor())
    assert "Nothing has been created yet" in result["message"]


# ----------------------------------------------------------- adversarial


def test_unknown_status_is_flagged_not_guessed():
    payload = {"patient": "P", "items": [{"test_name": "CBC", "status": "banana"}]}
    row = validate(payload, WORKFLOW)[0]
    assert row.ok is False
    assert "not one this workflow knows" in row.issues[0]
    assert row.state is None


def test_a_model_invented_stage_cannot_be_used():
    # Even a plausible-sounding status that is not configured must flag.
    payload = {"patient": "P", "items": [{"test_name": "MRI", "status": "radiology queue"}]}
    assert validate(payload, WORKFLOW)[0].ok is False


def test_terminal_status_is_rejected():
    payload = {"patient": "P", "items": [{"test_name": "CBC", "status": "closed"}]}
    assert validate(payload, WORKFLOW)[0].ok is False


def test_duplicate_tests_are_flagged():
    payload = {"patient": "P", "items": [
        {"test_name": "CBC", "status": "completed"},
        {"test_name": "cbc", "status": "completed"},
    ]}
    rows = validate(payload, WORKFLOW)
    assert rows[0].ok is True
    assert "duplicate" in rows[1].issues[0]


def test_missing_status_is_flagged():
    payload = {"patient": "P", "items": [{"test_name": "CBC", "status": ""}]}
    assert "no status" in validate(payload, WORKFLOW)[0].issues[0]


def test_empty_report_is_handled():
    result = extract_report("   ", WORKFLOW, RuleExtractor())
    assert result["ok"] is False
    assert "empty" in result["message"]


def test_report_with_no_tests_is_handled():
    result = extract_report("Hospital letterhead\nNo tests here", WORKFLOW, RuleExtractor())
    assert result["ok"] is False
    assert "No tests" in result["message"]


def test_extractor_failure_is_reported_not_crashed():
    result = extract_report(REPORT, WORKFLOW, StubExtractor(RuntimeError("bedrock down")))
    assert result["ok"] is False
    assert "bedrock down" in result["message"]


def test_malformed_model_output_is_survivable():
    result = extract_report(REPORT, WORKFLOW, StubExtractor({"items": "not a list"}))
    assert result["ok"] is False


def test_rows_carry_their_issues_for_the_human_to_fix():
    payload = {"patient": "P", "items": [
        {"test_name": "CBC", "status": "completed"},
        {"test_name": "MRI", "status": "banana"},
    ]}
    result = extract_report("x", WORKFLOW, StubExtractor(payload))
    assert result["ready"] == 1
    assert result["flagged"] == 1
    assert "needing a look" in result["message"]


# ------------------------------------------------- explainability chain


def test_removed_fragments_are_reported_for_review():
    """A person can see exactly what was taken out, not just trust that it was."""
    payload = {"patient": "P-1", "items": [
        {"test_name": "Serum potassium 6.8 mmol/L", "status": "result available"},
        {"test_name": "TSH", "status": "completed elevated"},
        {"test_name": "CBC", "status": "completed"},
    ]}
    rows = validate(payload, WORKFLOW)
    assert "6.8" in rows[0].removed
    assert "elevated" in rows[1].removed
    assert rows[2].removed == []


def test_raw_text_is_kept_for_review_only():
    payload = {"patient": "P-1", "items": [
        {"test_name": "Potassium 6.8", "status": "completed"}]}
    row = validate(payload, WORKFLOW)[0]
    assert row.raw_text.startswith("Potassium 6.8")   # shown to the reviewer
    assert row.test_name == "Potassium"               # what the engine gets
