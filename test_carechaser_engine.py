from datetime import datetime
from pathlib import Path

import pytest

from carechaser_engine import (
    CaseRecord,
    TransitionRejected,
    WorkflowEvent,
    apply_event,
    evaluate_case,
    load_workflow,
)

ROOT = Path(__file__).parent
WORKFLOW = load_workflow(ROOT / "workflow.yaml")
NOW = datetime.fromisoformat("2026-09-28T08:00:00+05:00")


def make_case(**overrides):
    data = {
        "case_id": "TEST-001",
        "patient_id_or_local_identifier": "DEMO-P",
        "test_type": "CBC",
        "ordering_clinician": "Dr. Test",
        "current_owner": "Dr. Test",
        "current_owner_role": "responsible_clinician",
        "current_state": "result_produced",
        "state_entered_at": "2026-09-27T20:00:00+05:00",
        "result_urgency": "routine",
    }
    data.update(overrides)
    return CaseRecord.from_dict(data)


def event(**overrides):
    data = {
        "event_id": "EV-001",
        "event_type": "result_reviewed",
        "actor_id": "Dr. Test",
        "actor_role": "responsible_clinician",
        "occurred_at": "2026-09-28T07:00:00+05:00",
        "received_at": "2026-09-28T07:01:00+05:00",
        "evidence": ["Result reviewed", "Next action documented"],
    }
    data.update(overrides)
    return WorkflowEvent.from_dict(data)


def test_on_time():
    case = make_case(state_entered_at="2026-09-28T04:00:00+05:00")
    result = evaluate_case(case, WORKFLOW, now=NOW)
    assert result.status == "ON_TIME"
    assert result.sla_hours == 24


def test_stuck():
    case = make_case(state_entered_at="2026-09-26T00:00:00+05:00")
    result = evaluate_case(case, WORKFLOW, now=NOW)
    assert result.status == "STUCK"
    assert result.sla_hours == 24


def test_sla_is_relative_to_current_state_entry():
    case = make_case(
        current_state="sample_processed",
        current_owner="Lab",
        current_owner_role="laboratory_team",
        state_entered_at="2026-09-28T05:00:00+05:00",
    )
    result = evaluate_case(case, WORKFLOW, now=NOW)
    assert result.status == "ON_TIME"
    assert result.sla_hours == 8


def test_future_timestamp_rejected():
    case = make_case(state_entered_at="2026-09-28T09:00:00+05:00")
    with pytest.raises(ValueError, match="future"):
        evaluate_case(case, WORKFLOW, now=NOW)


def test_legal_transition_updates_state_and_resets_clock():
    case = make_case(
        current_state="result_produced",
        state_entered_at="2026-09-27T20:00:00+05:00",
    )
    result = apply_event(case, event(), WORKFLOW, now=NOW)
    assert result.applied is True
    assert result.case.current_state == "result_reviewed"
    assert result.case.state_entered_at == event().occurred_at
    assert len(result.case.history) == 1
    assert result.case.history[0].state == "result_produced"
    assert result.case.history[0].completed_by == "Dr. Test"


def test_illegal_transition_is_rejected():
    case = make_case()
    bad = event(event_type="patient_informed")
    with pytest.raises(TransitionRejected, match="Illegal transition"):
        apply_event(case, bad, WORKFLOW, now=NOW)


def test_unauthorized_actor_is_rejected():
    case = make_case()
    bad = event(actor_id="Lab", actor_role="laboratory_team")
    with pytest.raises(TransitionRejected, match="not authorized"):
        apply_event(case, bad, WORKFLOW, now=NOW)


def test_duplicate_event_is_ignored():
    case = make_case()
    first = apply_event(case, event(), WORKFLOW, now=NOW)
    second = apply_event(first.case, event(), WORKFLOW, now=NOW)
    assert second.applied is False
    assert second.duplicate is True
    assert len(second.case.history) == 1


def test_backdated_event_keeps_occurred_and_received_times():
    case = make_case(state_entered_at="2026-09-28T05:00:00+05:00")
    e = event(
        occurred_at="2026-09-28T06:00:00+05:00",
        received_at="2026-09-28T07:30:00+05:00",
    )
    result = apply_event(case, e, WORKFLOW, now=NOW)
    assert result.case.state_entered_at == e.occurred_at
    assert result.history_entry.completed_at == e.occurred_at


def test_cancel_requires_reason_and_is_terminal():
    case = make_case(current_state="result_produced")
    with pytest.raises(TransitionRejected, match="Cancellation requires"):
        apply_event(case, event(event_id="EV-C", event_type="cancel", reason=None), WORKFLOW, now=NOW)

    result = apply_event(
        case,
        event(event_id="EV-C", event_type="cancel", reason="Patient did not attend"),
        WORKFLOW,
        now=NOW,
    )
    assert result.case.current_state == "cancelled"
    assert result.case.closure_status == "cancelled"
    assert result.case.history[-1].state == "result_produced"


def test_direct_close_from_result_reviewed_requires_reason():
    case = make_case(current_state="result_reviewed", current_owner="CareTeam", current_owner_role="responsible_care_team")
    with pytest.raises(TransitionRejected, match="closure reason"):
        apply_event(case, event(event_id="EV-CLOSE", event_type="close", actor_id="CareTeam", actor_role="responsible_care_team"), WORKFLOW, now=NOW)

    result = apply_event(
        case,
        event(
            event_id="EV-CLOSE",
            event_type="close",
            actor_id="CareTeam",
            actor_role="responsible_care_team",
            reason="Follow-up not clinically required",
            evidence=["Clinician documented no follow-up required"],
        ),
        WORKFLOW,
        now=NOW,
    )
    assert result.case.current_state == "closed"
    assert result.case.closure_status == "closed"


def test_force_close_requires_escalation_role_and_reason():
    case = make_case(current_state="result_produced")
    bad = event(event_id="EV-F", event_type="force_close", actor_role="responsible_clinician", reason="Duplicate order")
    with pytest.raises(TransitionRejected, match="escalation/supervisor"):
        apply_event(case, bad, WORKFLOW, now=NOW)

    good = event(
        event_id="EV-F",
        event_type="force_close",
        actor_id="Supervisor-1",
        actor_role="covering_clinician",
        reason="Duplicate order confirmed",
    )
    result = apply_event(case, good, WORKFLOW, now=NOW)
    assert result.case.current_state == "closed"
    assert result.case.closure_status == "forced_closed"
    assert result.history_entry.forced is True
    assert result.history_entry.force_close_reason == "Duplicate order confirmed"


def test_critical_escalation_clock_follows_override():
    case = make_case(
        current_state="sample_processed",
        current_owner="Lab",
        current_owner_role="laboratory_team",
        result_urgency="critical",
        state_entered_at="2026-09-28T07:30:00+05:00",
    )
    result = evaluate_case(case, WORKFLOW, now=NOW)
    assert result.status == "STUCK"
    assert result.escalate_after_hours == result.sla_hours
    assert result.escalation_due is True


def test_terminal_case_has_no_timer_or_chase_target():
    case = make_case(current_state="closed", state_entered_at="2026-08-01T00:00:00+05:00")
    result = evaluate_case(case, WORKFLOW, now=NOW)
    assert result.status == "TERMINAL"
    assert result.sla_hours is None
    assert result.escalation_due is False
    assert result.chase_target is None


def test_chase_target_switches_to_escalation_when_due():
    on_time = make_case(state_entered_at="2026-09-28T04:00:00+05:00")
    assert evaluate_case(on_time, WORKFLOW, now=NOW).chase_target == "Dr. Test"
    overdue = make_case(state_entered_at="2026-09-26T00:00:00+05:00")
    result = evaluate_case(overdue, WORKFLOW, now=NOW)
    assert result.escalation_due is True
    assert result.chase_target == "covering_clinician"


def test_roster_resolves_role_to_person():
    from carechaser_engine import resolve_role
    assert resolve_role(WORKFLOW, "lab_supervisor") == "Dr. Saleem"
    assert resolve_role(WORKFLOW, "not_a_role") == "not_a_role"
    assert resolve_role(WORKFLOW, None) is None


# ------------------------------------------- ownership across a handover
#
# A stage change often changes which role owns the case. Carrying the
# previous owner's name into the new role tells the board that a nurse owns
# a case sitting with the laboratory, and anything that reads ownership by
# name — a chase, a report of who works when — then reads it wrong.

# A distinct name: a second module-level NOW would shadow the one the tests
# above rely on, and they would quietly start measuring from the wrong moment.
HANDOVER_NOW = datetime.fromisoformat("2026-09-28T09:00:00+05:00")


def test_a_handover_moves_the_case_to_whoever_holds_the_new_role():
    case = make_case(
        current_state="ordered",
        current_owner="Nurse Ayesha",
        current_owner_role="authorized_sample_collector",
    )
    result = apply_event(
        case,
        event(event_type="sample_collected", actor_id="Nurse Ayesha",
              actor_role="authorized_sample_collector"),
        WORKFLOW, now=HANDOVER_NOW,
    )

    assert result.applied
    assert result.case.current_owner_role == "laboratory_team"
    assert result.case.current_owner == "Lab Desk", (
        "the case now belongs to the laboratory, not the nurse who sent it"
    )


def test_the_actor_keeps_the_case_when_the_role_does_not_change():
    case = make_case(
        current_state="sample_collected",
        current_owner="Lab Desk",
        current_owner_role="laboratory_team",
    )
    result = apply_event(
        case,
        event(event_type="sample_received", actor_id="Lab Desk",
              actor_role="laboratory_team"),
        WORKFLOW, now=HANDOVER_NOW,
    )
    assert result.case.current_owner == "Lab Desk"


def test_a_handover_never_invents_a_person():
    """An unassigned role falls back to its display name, never to a guess."""
    workflow = {**WORKFLOW, "roster": {}}
    case = make_case(
        current_state="ordered",
        current_owner="Nurse Ayesha",
        current_owner_role="authorized_sample_collector",
    )
    result = apply_event(
        case,
        event(event_type="sample_collected", actor_id="Nurse Ayesha",
              actor_role="authorized_sample_collector"),
        workflow, now=HANDOVER_NOW,
    )
    assert result.case.current_owner == "Laboratory Team"
