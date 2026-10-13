import json
from datetime import datetime
from pathlib import Path

import pytest

from carechaser_engine import TransitionRejected
from case_store import CaseNotFound, CaseStore

ROOT = Path(__file__).parent
NOW = datetime.fromisoformat("2026-09-28T08:00:00+05:00")


def seed_cases():
    return [
        {
            "case_id": "S-001",
            "patient_id_or_local_identifier": "P1",
            "test_type": "CBC",
            "ordering_clinician": "Dr. Khan",
            "current_owner": "Dr. Khan",
            "current_owner_role": "responsible_clinician",
            "current_state": "result_produced",
            "state_entered_at": "2026-09-26T00:00:00+05:00",
            "result_urgency": "routine",
        },
        {
            "case_id": "S-002",
            "patient_id_or_local_identifier": "P2",
            "test_type": "LFT",
            "ordering_clinician": "Dr. Ahmed",
            "current_owner": "Lab Desk",
            "current_owner_role": "laboratory_team",
            "current_state": "sample_processed",
            "state_entered_at": "2026-09-28T02:00:00+05:00",
            "result_urgency": "routine",
        },
        {
            "case_id": "S-003",
            "patient_id_or_local_identifier": "P3",
            "test_type": "Biopsy",
            "ordering_clinician": "Dr. Ali",
            "current_owner": "Ward 2",
            "current_owner_role": "responsible_care_team",
            "current_state": "closed",
            "state_entered_at": "2026-09-01T00:00:00+05:00",
            "closure_status": "closed",
        },
    ]


@pytest.fixture
def store(tmp_path):
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps(seed_cases()), encoding="utf-8")
    return CaseStore(cases_file, ROOT / "workflow.yaml")


def test_loads_all_cases(store):
    assert len(store) == 3
    assert store.get("S-001").test_type == "CBC"


def test_missing_case_raises(store):
    with pytest.raises(CaseNotFound):
        store.get("NOPE")


def test_stuck_cases_exclude_terminal_and_on_time(store):
    rows = store.stuck_cases(now=NOW)
    ids = [case.case_id for case, _ in rows]
    assert ids == ["S-001"]  # S-002 on time, S-003 terminal


def test_stuck_cases_sorted_worst_first(tmp_path):
    cases = seed_cases()
    cases.append({
        **cases[0],
        "case_id": "S-004",
        "state_entered_at": "2026-09-20T00:00:00+05:00",
    })
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    store = CaseStore(path, ROOT / "workflow.yaml")
    rows = store.stuck_cases(now=NOW)
    assert [c.case_id for c, _ in rows] == ["S-004", "S-001"]


def test_min_hours_overdue_filters_on_lateness_not_elapsed(store):
    # S-001 is 56h elapsed against a 24h SLA, so 32h overdue.
    assert store.stuck_cases(now=NOW, min_hours_overdue=30)
    assert not store.stuck_cases(now=NOW, min_hours_overdue=40)


def test_pending_for_matches_person_role_and_roster(store):
    assert [c.case_id for c, _ in store.pending_for("Dr. Khan", now=NOW)] == ["S-001"]
    assert [c.case_id for c, _ in store.pending_for("laboratory_team", now=NOW)] == ["S-002"]
    # "Lab Desk" is the rostered person for laboratory_team
    assert [c.case_id for c, _ in store.pending_for("Lab Desk", now=NOW)] == ["S-002"]


def test_pending_excludes_closed_cases(store):
    assert store.pending_for("Ward 2", now=NOW) == []


def test_apply_event_persists_to_disk(store):
    result = store.apply(
        "S-001",
        {
            "event_id": "EV-1",
            "event_type": "result_reviewed",
            "actor_id": "Dr. Khan",
            "actor_role": "responsible_clinician",
            "occurred_at": "2026-09-28T07:00:00+05:00",
            "received_at": "2026-09-28T07:00:00+05:00",
            "evidence": ["Reviewed"],
        },
        now=NOW,
    )
    assert result.applied is True
    reloaded = CaseStore(store.cases_path, store.workflow_path)
    assert reloaded.get("S-001").current_state == "result_reviewed"
    assert len(reloaded.get("S-001").history) == 1


def test_rejected_event_leaves_stored_case_untouched(store):
    with pytest.raises(TransitionRejected):
        store.apply(
            "S-001",
            {
                "event_id": "EV-BAD",
                "event_type": "result_reviewed",
                "actor_id": "Lab Desk",
                "actor_role": "laboratory_team",
                "occurred_at": "2026-09-28T07:00:00+05:00",
                "received_at": "2026-09-28T07:00:00+05:00",
            },
            now=NOW,
        )
    reloaded = CaseStore(store.cases_path, store.workflow_path)
    assert reloaded.get("S-001").current_state == "result_produced"
    assert reloaded.get("S-001").history == ()


def test_duplicate_event_is_a_noop(store):
    event = {
        "event_id": "EV-DUP",
        "event_type": "result_reviewed",
        "actor_id": "Dr. Khan",
        "actor_role": "responsible_clinician",
        "occurred_at": "2026-09-28T07:00:00+05:00",
        "received_at": "2026-09-28T07:00:00+05:00",
    }
    store.apply("S-001", event, now=NOW)
    second = store.apply("S-001", event, now=NOW)
    assert second.duplicate is True
    reloaded = CaseStore(store.cases_path, store.workflow_path)
    assert len(reloaded.get("S-001").history) == 1


def test_history_survives_a_round_trip(store):
    store.apply(
        "S-001",
        {
            "event_id": "EV-H",
            "event_type": "result_reviewed",
            "actor_id": "Dr. Khan",
            "actor_role": "responsible_clinician",
            "occurred_at": "2026-09-28T07:00:00+05:00",
            "received_at": "2026-09-28T07:00:00+05:00",
            "evidence": ["Reviewed", "Plan documented"],
        },
        now=NOW,
    )
    reloaded = CaseStore(store.cases_path, store.workflow_path)
    entry = reloaded.get("S-001").history[0]
    assert entry.state == "result_produced"
    assert entry.completed_by == "Dr. Khan"
    assert list(entry.evidence_recorded) == ["Reviewed", "Plan documented"]
    # idempotency survives restart too
    assert "EV-H" in reloaded.get("S-001").processed_event_ids


def test_person_name_does_not_match_everyone_sharing_their_role(tmp_path):
    """A person's name must not expand to every case carrying their role."""
    cases = seed_cases()
    cases.append({
        **cases[0],
        "case_id": "S-OTHER",
        "current_owner": "Dr. Ahmed",          # same role, different person
        "current_owner_role": "responsible_clinician",
        "state_entered_at": "2026-09-28T06:00:00+05:00",
    })
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    store = CaseStore(path, ROOT / "workflow.yaml")

    khan = [case.case_id for case, _ in store.pending_for("Dr. Khan", now=NOW)]
    assert khan == ["S-001"]                    # not S-OTHER

    by_role = {case.case_id for case, _ in store.pending_for("responsible_clinician", now=NOW)}
    assert by_role == {"S-001", "S-OTHER"}      # the role does cover both


def test_escalation_target_sees_the_case_once_escalation_is_due(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(seed_cases()), encoding="utf-8")
    store = CaseStore(path, ROOT / "workflow.yaml")
    # S-001 is 56h into a 24h state, so escalation to covering_clinician is due.
    riaz = [case.case_id for case, _ in store.pending_for("Dr. Riaz", now=NOW)]
    assert riaz == ["S-001"]
    # S-002 is on time, so its escalation target must not see it.
    assert [c.case_id for c, _ in store.pending_for("Dr. Saleem", now=NOW)] == []


def test_board_evaluation_skips_cases_that_have_not_started(tmp_path):
    """A case created while the clock was ahead must not break the board."""
    cases = seed_cases()
    cases.append({
        **cases[0],
        "case_id": "S-FUTURE",
        "state_entered_at": "2026-09-29T08:00:00+05:00",   # after NOW
    })
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    store = CaseStore(path, ROOT / "workflow.yaml")

    rows = store.evaluate_all(now=NOW)
    assert "S-FUTURE" not in [case.case_id for case, _ in rows]
    assert len(rows) == 3                  # the rest still evaluate
    assert store.stuck_cases(now=NOW)      # and the board still works
