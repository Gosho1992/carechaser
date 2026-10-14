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
    _loose_state,
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


# ------------------------------------------- printed reports, dotted leaders
#
# A real ward printout leads the eye from the test to its result with a row of
# dots, and writes the status in the same row. Both broke the parser: the dots
# were kept in the test name, and the comma in "released, not yet
# acknowledged" split the line in the middle of the status.


def test_a_dotted_leader_separates_the_test_from_its_result():
    pair = RuleExtractor()._split("Thyroid panel ........... TSH 0.2 mIU/L   released")
    assert pair == ("Thyroid panel", "TSH 0.2 mIU/L   released")


def test_a_comma_inside_the_status_does_not_split_the_line():
    line = "Vitamin B12 ....... 310 ng/L  LOW  released, not yet acknowledged"
    name, status = RuleExtractor()._split(line)
    assert name == "Vitamin B12"
    assert status.startswith("310 ng/L")
    assert "not yet acknowledged" in status


def test_a_printed_report_yields_clean_names_and_real_stages():
    report = "\n".join([
        "MUZAFFAR KHAN SURGICAL HOSPITAL",
        "Patient ref: MK-4631        Date: 13/10/2026",
        "",
        "Thyroid panel ........... TSH 0.2 mIU/L   released, not yet acknowledged",
        "Group and save .......... B Rh positive   requested, awaiting collection",
    ])
    result = extract_report(report, WORKFLOW, RuleExtractor())
    staged = {r["test_name"]: r["state"] for r in result["rows"] if r["state"]}

    assert staged["Thyroid panel"] == "result_produced"
    assert staged["Group and save"] == "ordered"
    assert result["patient"] == "MK-4631"


def test_no_unit_survives_a_dotted_line():
    report = "Serum potassium ..... 6.8 mmol/L  HIGH  released, awaiting review"
    rows = extract_report(report, WORKFLOW, RuleExtractor())["rows"]
    names = " ".join(r["test_name"] for r in rows)
    assert "Serum potassium" in names
    for leak in ("6.8", "mmol", "HIGH"):
        assert leak not in names


def test_the_patient_label_is_not_mistaken_for_the_reference():
    """'Patient ref: MK-4471' must not come back as 'ref'."""
    assert RuleExtractor._patient_in("Patient ref: MK-4471") == "MK-4471"
    assert RuleExtractor._patient_in("Patient ID MK-4471") == "MK-4471"
    assert RuleExtractor._patient_in("patient reference  DEMO-P001") == "DEMO-P001"


def test_prose_is_not_mistaken_for_a_patient_reference():
    assert RuleExtractor._patient_in("Patient was seen on the ward") is None
    assert RuleExtractor._patient_in("no patient here") is None


# ------------------------------------------- what OCR actually hands us back
#
# Textract does not return the dotted leaders: it reads a printed row as
# separate blocks, which scan_intake rejoins with spaces. So on a scanned
# report the column gap is the separator that survives, not the dots.


def test_a_column_gap_separates_the_test_from_its_status():
    pair = RuleExtractor()._split("Frozen section   margins reported   reviewed")
    assert pair == ("Frozen section", "margins reported   reviewed")


def test_a_single_space_is_not_a_column_gap():
    assert RuleExtractor()._split("Frozen section margins reported") is None


def test_real_textract_output_stages_correctly():
    """Verbatim from a Textract read of a generated ward report."""
    report = "\n".join([
        "MUZAFFAR KHAN SURGICAL HOSPITAL",
        "Patient ref: MK-4418   Date: 13/10/2026",
        "Frozen section   margins reported   reviewed, follow-up pending",
        "Blood culture   growth at 18 h   specimen collected, in transit",
        "Thyroid panel   T4 9 pmol/L   report released, awaiting review",
    ])
    rows = extract_report(report, WORKFLOW, RuleExtractor())
    staged = {r["test_name"]: r["state"] for r in rows["rows"] if r["state"]}

    assert staged["Frozen section"] == "result_reviewed"
    assert staged["Blood culture"] == "sample_collected"
    assert staged["Thyroid panel"] == "result_produced"
    assert rows["patient"] == "MK-4418"


# ------------------------------------------ a status holding two known words


def test_the_word_the_writer_led_with_wins():
    """'reviewed, follow-up pending' holds both 'reviewed' and 'pending'."""
    assert _loose_state("reviewed, follow-up pending") == "result_reviewed"
    assert _loose_state("pending review") == "sample_processed"


def test_a_longer_phrase_beats_a_shorter_one_at_the_same_position():
    assert _loose_state("result released to the ward") == "result_produced"


def test_an_unknown_status_is_refused_rather_than_guessed():
    assert _loose_state("something nobody wrote down") is None


def test_an_unknown_status_leaves_the_row_for_a_human():
    rows = extract_report(
        "Mystery test   whatever happened here", WORKFLOW, RuleExtractor()
    )["rows"]
    row = next(r for r in rows if r["test_name"] == "Mystery test")
    assert row["state"] is None
    assert not row["ok"]
    assert any("not one this workflow knows" in issue for issue in row["issues"])


# ----------------------------------------------- urgency is read, not judged


def test_a_printed_urgency_label_is_read():
    rows = extract_report(
        "Serum potassium   6.8 mmol/L   STAT released, awaiting review",
        WORKFLOW, RuleExtractor(),
    )["rows"]
    assert rows[0]["urgency"] == "critical"


def test_critical_and_urgent_are_read_too():
    for label in ("CRITICAL", "urgent", "Stat"):
        rows = extract_report(
            f"Troponin   0.31 ng/mL   {label} released, awaiting review",
            WORKFLOW, RuleExtractor(),
        )["rows"]
        assert rows[0]["urgency"] == "critical", label


def test_a_report_with_no_label_stays_routine():
    """A dangerous-looking number is still routine. CareChaser does not judge."""
    rows = extract_report(
        "Serum potassium   9.9 mmol/L   released, awaiting review",
        WORKFLOW, RuleExtractor(),
    )["rows"]
    assert rows[0]["urgency"] == "routine"


def test_a_word_containing_stat_is_not_an_urgency_label():
    rows = extract_report(
        "Thyroid panel   understated reading   released, awaiting review",
        WORKFLOW, RuleExtractor(),
    )["rows"]
    assert rows[0]["urgency"] == "routine"
