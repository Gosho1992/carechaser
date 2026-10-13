"""Tests for follow-up memory and cross-session continuity."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from case_store import CaseStore
from followups import FollowUp, FollowUpLog, describe_gap, unresolved_followups

ROOT = Path(__file__).parent
DAY_ONE = datetime.fromisoformat("2026-09-28T08:00:00+05:00")
DAY_TWO = DAY_ONE + timedelta(days=1)


def seed():
    return [
        {
            "case_id": "F-001",
            "patient_id_or_local_identifier": "P1",
            "test_type": "Biopsy",
            "ordering_clinician": "Dr. Khan",
            "current_owner": "Dr. Khan",
            "current_owner_role": "responsible_clinician",
            "current_state": "result_produced",
            "state_entered_at": "2026-09-25T06:00:00+05:00",
            "result_urgency": "routine",
        },
        {
            "case_id": "F-002",
            "patient_id_or_local_identifier": "P2",
            "test_type": "CBC",
            "ordering_clinician": "Dr. Ali",
            "current_owner": "Lab Desk",
            "current_owner_role": "laboratory_team",
            "current_state": "sample_processed",
            "state_entered_at": "2026-09-28T06:00:00+05:00",
            "result_urgency": "routine",
        },
    ]


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(seed()), encoding="utf-8")
    return CaseStore(path, ROOT / "workflow.yaml")


@pytest.fixture
def log(tmp_path):
    return FollowUpLog(tmp_path / "chase_log.jsonl")


# ------------------------------------------------------------------- the log


def test_log_round_trips(log):
    log.record("F-001", "escalation", "Dr. Riaz", at=DAY_ONE)
    rows = log.all()
    assert len(rows) == 1
    assert rows[0].case_id == "F-001"
    assert rows[0].kind == "escalation"


def test_log_is_append_only(log):
    log.record("F-001", "escalation", "Dr. Riaz", at=DAY_ONE)
    log.record("F-001", "chase", "Dr. Riaz", at=DAY_TWO)
    assert len(log.for_case("F-001")) == 2
    assert log.last_for_case("F-001").kind == "chase"


def test_corrupt_line_does_not_hide_the_rest(log):
    log.record("F-001", "chase", "Dr. Riaz", at=DAY_ONE)
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    log.record("F-002", "chase", "Lab Desk", at=DAY_TWO)
    assert len(log.all()) == 2


def test_hours_ago_and_spoken_gap():
    entry = FollowUp("F-001", "chase", "Dr. Riaz", DAY_ONE)
    assert entry.hours_ago(DAY_TWO) == 24.0
    assert describe_gap(0.5) == "less than an hour ago"
    assert describe_gap(6) == "6 hours ago"
    assert describe_gap(24) == "yesterday"
    assert describe_gap(72) == "3 days ago"


# ---------------------------------------------------------------- continuity


def test_followup_is_remembered_the_next_day(store, log):
    log.record("F-001", "escalation", "Dr. Riaz", at=DAY_ONE)
    rows = unresolved_followups(store, log, now=DAY_TWO)
    assert len(rows) == 1
    assert rows[0]["case_id"] == "F-001"
    assert rows[0]["hours_since_followup"] == 24.0
    assert rows[0]["last_kind"] == "escalation"


def test_case_that_moved_after_the_followup_drops_out(store, log):
    log.record("F-001", "escalation", "Dr. Riaz", at=DAY_ONE)
    store.apply(
        "F-001",
        {
            "event_id": "EV-1",
            "event_type": "result_reviewed",
            "actor_id": "Dr. Khan",
            "actor_role": "responsible_clinician",
            "occurred_at": DAY_TWO.isoformat(),
            "received_at": DAY_TWO.isoformat(),
        },
        now=DAY_TWO,
    )
    assert unresolved_followups(store, log, now=DAY_TWO) == []


def test_closed_case_drops_out(store, log):
    log.record("F-002", "chase", "Lab Desk", at=DAY_ONE)
    for event_id, event_type, actor, role in [
        ("E1", "result_released", "Lab Desk", "laboratory_team"),
        ("E2", "result_reviewed", "Dr. Ali", "responsible_clinician"),
        ("E3", "patient_informed", "Ward", "responsible_care_team"),
        ("E4", "close", "Ward", "responsible_care_team"),
    ]:
        store.apply(
            "F-002",
            {
                "event_id": event_id,
                "event_type": event_type,
                "actor_id": actor,
                "actor_role": role,
                "occurred_at": DAY_TWO.isoformat(),
                "received_at": DAY_TWO.isoformat(),
                "reason": "Follow-up not required",
                "evidence": ["done"],
            },
            now=DAY_TWO,
        )
    assert unresolved_followups(store, log, now=DAY_TWO) == []


def test_unknown_case_in_the_log_is_ignored(store, log):
    log.record("GONE-999", "chase", "Dr. Riaz", at=DAY_ONE)
    assert unresolved_followups(store, log, now=DAY_TWO) == []


def test_min_hours_since_filters_recent_followups(store, log):
    log.record("F-001", "chase", "Dr. Riaz", at=DAY_TWO - timedelta(hours=2))
    assert unresolved_followups(store, log, now=DAY_TWO, min_hours_since=1)
    assert not unresolved_followups(store, log, now=DAY_TWO, min_hours_since=6)


def test_most_neglected_case_is_listed_first(store, log):
    log.record("F-001", "escalation", "Dr. Riaz", at=DAY_ONE)
    log.record("F-002", "chase", "Lab Desk", at=DAY_TWO - timedelta(hours=1))
    rows = unresolved_followups(store, log, now=DAY_TWO)
    assert [row["case_id"] for row in rows] == ["F-001", "F-002"]


def test_repeated_followups_are_counted(store, log):
    log.record("F-001", "escalation", "Dr. Riaz", at=DAY_ONE)
    log.record("F-001", "chase", "Dr. Riaz", at=DAY_ONE + timedelta(hours=6))
    row = unresolved_followups(store, log, now=DAY_TWO)[0]
    assert row["followups"] == 2
    assert row["last_kind"] == "chase"


def test_briefing_survives_a_case_dated_after_the_clock(store, log, monkeypatch):
    """Resetting the demo clock can leave a case starting in the future.

    The engine refuses to judge such a case, which is correct, but one of
    them must not take the whole briefing down.
    """
    log.record("F-001", "escalation", "Dr. Riaz", at=DAY_ONE)
    store.apply(
        "F-001",
        {
            "event_id": "EV-FUTURE",
            "event_type": "result_reviewed",
            "actor_id": "Dr. Khan",
            "actor_role": "responsible_clinician",
            "occurred_at": DAY_TWO.isoformat(),
            "received_at": DAY_TWO.isoformat(),
        },
        now=DAY_TWO,
    )
    # Now ask as though the clock went back a day: F-001 started "later".
    rows = unresolved_followups(store, log, now=DAY_ONE)
    assert isinstance(rows, list)          # no exception
