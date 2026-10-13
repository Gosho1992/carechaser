"""Tests for the chase layer.

The important ones here are the guardrail tests: they prove a language model
cannot put an invented number, a missing case id, or clinical language into a
message that reaches a clinician.
"""

from __future__ import annotations

import json

import pytest

from chase import (
    ChaseFacts,
    Chaser,
    LogChannel,
    TemplateDrafter,
    check_draft,
    say_duration,
    say_limit,
)


def facts(**overrides) -> ChaseFacts:
    data = {
        "case_id": "CC-001",
        "test_type": "CBC",
        "patient": "DEMO-P001",
        "state": "Result Released",
        "hours_in_state": 31.0,
        "sla_hours": 24.0,
        "owner": "Dr. Khan",
        "chase_target": "Dr. Riaz",
        "urgency": "routine",
        "escalated": True,
    }
    data.update(overrides)
    return ChaseFacts(**data)


class StubDrafter:
    """Stands in for Bedrock so guardrail behaviour can be tested exactly."""

    name = "bedrock"

    def __init__(self, text: str):
        self.text = text

    def draft(self, facts: ChaseFacts) -> str:
        return self.text


class BrokenDrafter:
    name = "bedrock"

    def draft(self, facts: ChaseFacts) -> str:
        raise RuntimeError("bedrock unavailable")


# ----------------------------------------------------------------- the facts


def test_hours_overdue_is_derived_not_supplied():
    assert facts().hours_overdue == 7.0
    assert facts(sla_hours=None).hours_overdue is None


def test_prompt_contains_only_given_facts():
    prompt = facts().to_prompt()
    assert "CC-001" in prompt
    assert "31" in prompt
    assert "escalated" in prompt


# ------------------------------------------------- critical, sub-hour limits
#
# A critical result's limit is ten minutes. Everything below guards against
# the bug where whole-hour rounding made the most urgent case on the board
# announce itself as "0 hours past the 0.17 hour limit".


def critical_facts(**overrides) -> ChaseFacts:
    return facts(
        case_id="CC-002",
        test_type="Serum potassium",
        hours_in_state=1.0,
        sla_hours=0.17,
        urgency="critical",
        **overrides,
    )


def test_durations_under_an_hour_are_said_in_minutes():
    assert say_duration(0.17) == "10 minutes"
    assert say_duration(0.83) == "50 minutes"
    assert say_limit(0.17) == "10 minute"


def test_durations_of_an_hour_or_more_are_said_in_hours():
    assert say_duration(1.0) == "1 hour"
    assert say_duration(31.0) == "31 hours"
    assert say_limit(24.0) == "24 hour"


def test_a_critical_breach_is_never_reported_as_zero():
    text = TemplateDrafter().draft(critical_facts())
    assert "0 hours past" not in text
    assert "50 minutes past the 10 minute limit" in text


def test_whole_hour_durations_are_singular():
    text = TemplateDrafter().draft(critical_facts())
    assert "for 1 hour," in text
    assert "1 hours" not in text


def test_a_case_inside_its_limit_claims_no_breach():
    text = TemplateDrafter().draft(facts(hours_in_state=2.0, sla_hours=24.0))
    assert "past the" not in text


def test_the_guardrail_accepts_a_breach_stated_in_minutes():
    text = (
        "CC-002 has been waiting 1 hour at Result Released, 50 minutes past "
        "the 10 minute limit. Please record the next step."
    )
    assert check_draft(text, critical_facts()) is None


def test_the_guardrail_still_rejects_an_invented_minute_count():
    text = (
        "CC-002 has been waiting 1 hour at Result Released, 400 minutes past "
        "the 10 minute limit. Please record the next step."
    )
    assert "unsupported number" in check_draft(text, critical_facts())


# ------------------------------------------------------------- the guardrail


def test_good_draft_passes():
    text = "CC-001 has been waiting 31 hours at Result Released, 7 hours past the 24 hour limit. Please record the next step."
    assert check_draft(text, facts()) is None


def test_draft_without_the_case_id_is_rejected():
    text = "This blood test has been waiting 31 hours and needs attention now please."
    assert "case id" in check_draft(text, facts())


def test_invented_number_is_rejected():
    text = "CC-001 has been waiting 96 hours at Result Released. Please record the next step."
    assert "unsupported number" in check_draft(text, facts())


def test_clinical_language_is_rejected():
    text = "CC-001 has been waiting 31 hours and the abnormal result suggests follow-up is needed."
    assert "clinical language" in check_draft(text, facts())


def test_empty_or_tiny_draft_is_rejected():
    assert "empty" in check_draft("CC-001", facts())
    assert "empty" in check_draft("", facts())


def test_numbers_inside_identifiers_are_not_treated_as_claims():
    # DEMO-P001 and CC-001 both contain digits that are not claims.
    text = "CC-001 for patient DEMO-P001 has waited 31 hours. Please record the next step."
    assert check_draft(text, facts()) is None


# ---------------------------------------------------------------- the runner


def test_template_drafter_is_always_factual():
    text = TemplateDrafter().draft(facts())
    assert check_draft(text, facts()) is None
    assert "CC-001" in text and "31 hours" in text


def test_bad_model_draft_falls_back_to_the_template():
    chaser = Chaser(drafter=StubDrafter("CC-001 is 500 hours late."), channel=LogChannel("/dev/null"))
    draft = chaser.compose(facts())
    assert draft.source == "template-after-rejection"
    assert "unsupported number" in draft.rejected_reason
    assert check_draft(draft.text, facts()) is None


def test_drafter_failure_falls_back_rather_than_crashing():
    chaser = Chaser(drafter=BrokenDrafter(), channel=LogChannel("/dev/null"))
    draft = chaser.compose(facts())
    assert draft.source == "template-after-rejection"
    assert "bedrock unavailable" in draft.rejected_reason


def test_good_model_draft_is_kept():
    good = "CC-001 has been waiting 31 hours at Result Released, 7 hours past the limit. Please record the next step."
    chaser = Chaser(drafter=StubDrafter(good), channel=LogChannel("/dev/null"))
    draft = chaser.compose(facts())
    assert draft.source == "bedrock"
    assert draft.text == good


# --------------------------------------------------------------- the channel


def test_dry_run_drafts_without_delivering(tmp_path):
    log = tmp_path / "chase.jsonl"
    chaser = Chaser(drafter=TemplateDrafter(), channel=LogChannel(log))
    record = chaser.send(facts(), dry_run=True)
    assert record.delivered is False
    assert record.detail["dry_run"] is True
    assert not log.exists()


def test_send_writes_to_the_channel(tmp_path):
    log = tmp_path / "chase.jsonl"
    chaser = Chaser(drafter=TemplateDrafter(), channel=LogChannel(log))
    record = chaser.send(facts())
    assert record.delivered is True
    entry = json.loads(log.read_text(encoding="utf-8").strip())
    assert entry["case_id"] == "CC-001"
    assert entry["to"] == "Dr. Riaz"


def test_log_is_append_only(tmp_path):
    log = tmp_path / "chase.jsonl"
    chaser = Chaser(drafter=TemplateDrafter(), channel=LogChannel(log))
    chaser.send(facts())
    chaser.send(facts(case_id="CC-002"))
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
