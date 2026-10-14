"""CareChaser workflow engine.

Step 3: deterministic SLA evaluation.
Step 4: auditable state transitions and append-only history.

No LLM/medical interpretation is involved in workflow decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


class WorkflowConfigError(ValueError):
    pass


class CaseValidationError(ValueError):
    pass


class TransitionRejected(ValueError):
    """Raised when an event cannot legally change a case."""


@dataclass(frozen=True)
class WorkflowEvent:
    event_id: str
    event_type: str
    actor_id: str
    actor_role: str
    occurred_at: datetime
    received_at: datetime
    evidence: tuple[str, ...] = ()
    note: Optional[str] = None
    reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowEvent":
        required = ["event_id", "event_type", "actor_id", "actor_role", "occurred_at"]
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise CaseValidationError(f"Missing required event fields: {', '.join(missing)}")

        received_at = data.get("received_at") or datetime.now(timezone.utc).isoformat()
        evidence = data.get("evidence") or []
        if not isinstance(evidence, list):
            raise CaseValidationError("event.evidence must be a list")

        return cls(
            event_id=str(data["event_id"]),
            event_type=str(data["event_type"]),
            actor_id=str(data["actor_id"]),
            actor_role=str(data["actor_role"]),
            occurred_at=_parse_datetime(data["occurred_at"]),
            received_at=_parse_datetime(received_at),
            evidence=tuple(str(item) for item in evidence),
            note=data.get("note"),
            reason=data.get("reason"),
        )


@dataclass(frozen=True)
class HistoryEntry:
    state: str
    entered_at: datetime
    completed_at: datetime
    completed_by: str
    completed_by_role: str
    evidence_recorded: tuple[str, ...] = ()
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    note: Optional[str] = None
    forced: bool = False
    force_close_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entered_at"] = self.entered_at.isoformat()
        data["completed_at"] = self.completed_at.isoformat()
        data["evidence_recorded"] = list(self.evidence_recorded)
        return data


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    patient_id_or_local_identifier: str
    test_type: str
    ordering_clinician: str
    current_owner: str
    current_owner_role: str
    current_state: str
    state_entered_at: datetime
    last_action_at: Optional[datetime] = None
    next_action: Optional[str] = None
    escalation_target: Optional[str] = None
    result_urgency: str = "routine"
    closure_status: str = "open"
    history: tuple[HistoryEntry, ...] = ()
    processed_event_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseRecord":
        required = [
            "case_id",
            "patient_id_or_local_identifier",
            "test_type",
            "ordering_clinician",
            "current_owner",
            "current_owner_role",
            "current_state",
            "state_entered_at",
        ]
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise CaseValidationError(f"Missing required case fields: {', '.join(missing)}")

        raw_history = data.get("history") or []
        if not isinstance(raw_history, list):
            raise CaseValidationError("case.history must be a list")

        history = []
        for item in raw_history:
            history.append(
                HistoryEntry(
                    state=str(item["state"]),
                    entered_at=_parse_datetime(item["entered_at"]),
                    completed_at=_parse_datetime(item["completed_at"]),
                    completed_by=str(item["completed_by"]),
                    completed_by_role=str(item["completed_by_role"]),
                    evidence_recorded=tuple(str(x) for x in item.get("evidence_recorded", [])),
                    event_id=item.get("event_id"),
                    event_type=item.get("event_type"),
                    note=item.get("note"),
                    forced=bool(item.get("forced", False)),
                    force_close_reason=item.get("force_close_reason"),
                )
            )

        return cls(
            case_id=str(data["case_id"]),
            patient_id_or_local_identifier=str(data["patient_id_or_local_identifier"]),
            test_type=str(data["test_type"]),
            ordering_clinician=str(data["ordering_clinician"]),
            current_owner=str(data["current_owner"]),
            current_owner_role=str(data["current_owner_role"]),
            current_state=str(data["current_state"]),
            state_entered_at=_parse_datetime(data["state_entered_at"]),
            last_action_at=_parse_optional_datetime(data.get("last_action_at")),
            next_action=data.get("next_action"),
            escalation_target=data.get("escalation_target"),
            result_urgency=str(data.get("result_urgency", "routine")),
            closure_status=str(data.get("closure_status", "open")),
            history=tuple(history),
            processed_event_ids=tuple(str(x) for x in data.get("processed_event_ids", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state_entered_at"] = self.state_entered_at.isoformat()
        if self.last_action_at:
            result["last_action_at"] = self.last_action_at.isoformat()
        result["history"] = [entry.to_dict() for entry in self.history]
        result["processed_event_ids"] = list(self.processed_event_ids)
        return result


@dataclass(frozen=True)
class Evaluation:
    case_id: str
    state: str
    display_name: str
    status: str  # ON_TIME | STUCK | TERMINAL
    reason: str
    elapsed_hours: float
    sla_hours: Optional[float]
    owner: str
    escalation_target: Optional[str]
    escalate_after_hours: Optional[float]
    escalation_due: bool = False
    chase_target: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionResult:
    applied: bool
    duplicate: bool
    case: CaseRecord
    history_entry: Optional[HistoryEntry]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "duplicate": self.duplicate,
            "case": self.case.to_dict(),
            "history_entry": self.history_entry.to_dict() if self.history_entry else None,
            "message": self.message,
        }


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise CaseValidationError(f"Invalid datetime: {value!r}") from exc

    if dt.tzinfo is None:
        raise CaseValidationError(f"Datetime must include a timezone offset: {value!r}")
    return dt


def _parse_optional_datetime(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    return _parse_datetime(value)


def load_workflow(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict) or not isinstance(config.get("workflow"), dict):
        raise WorkflowConfigError("workflow.yaml must contain a top-level 'workflow' mapping")

    workflow = config["workflow"]
    states = workflow.get("states")
    if not isinstance(states, list) or not states:
        raise WorkflowConfigError("workflow.states must be a non-empty list")

    seen = set()
    for state in states:
        if not isinstance(state, dict) or not state.get("state"):
            raise WorkflowConfigError("Every workflow state must define 'state'")
        name = str(state["state"])
        if name in seen:
            raise WorkflowConfigError(f"Duplicate workflow state: {name}")
        seen.add(name)
        if "sla_hours" not in state:
            raise WorkflowConfigError(f"State {name!r} has no sla_hours")
        if "owner_role" not in state:
            raise WorkflowConfigError(f"State {name!r} has no owner_role")
        if "next_states" not in state:
            raise WorkflowConfigError(f"State {name!r} has no next_states")
        for target in state.get("next_states", []):
            if target not in seen and target not in {str(x.get("state")) for x in states}:
                raise WorkflowConfigError(f"State {name!r} references unknown next_state {target!r}")

    return workflow


def _state_map(workflow: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["state"]): item for item in workflow["states"]}


def _find_urgency_override(workflow: Mapping[str, Any], case: CaseRecord) -> Optional[Mapping[str, Any]]:
    if case.result_urgency.lower() in {"critical", "urgent", "critical_result"}:
        overrides = workflow.get("urgency_overrides", {})
        if isinstance(overrides, Mapping):
            return overrides.get("critical_result")
    return None


def resolve_role(workflow: Mapping[str, Any], role_id: Optional[str]) -> Optional[str]:
    """Map a machine role id to the person currently holding it via the roster,
    else to the role's display name, else return the id unchanged.
    The engine never invents a person."""
    if not role_id:
        return None
    roster = workflow.get("roster")
    if isinstance(roster, Mapping) and roster.get(role_id):
        return str(roster[role_id])
    roles = workflow.get("roles")
    if isinstance(roles, Mapping):
        entry = roles.get(role_id)
        if isinstance(entry, Mapping) and entry.get("display_name"):
            return str(entry["display_name"])
    return str(role_id)


def evaluate_case(case: CaseRecord, workflow: Mapping[str, Any], *, now: Optional[datetime] = None) -> Evaluation:
    """Evaluate a case using only workflow configuration and timestamps."""
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        raise CaseValidationError("now must include a timezone offset")

    states = _state_map(workflow)
    state = states.get(case.current_state)
    if state is None:
        raise CaseValidationError(f"Unknown workflow state: {case.current_state!r}")

    elapsed_seconds = (now.astimezone(timezone.utc) - case.state_entered_at.astimezone(timezone.utc)).total_seconds()
    if elapsed_seconds < 0:
        raise CaseValidationError("state_entered_at cannot be in the future")
    elapsed_hours = elapsed_seconds / 3600

    display_name = str(state.get("display_name", case.current_state))
    owner = case.current_owner

    if bool(state.get("terminal", False)) or state.get("sla_hours") == 0:
        return Evaluation(
            case_id=case.case_id,
            state=case.current_state,
            display_name=display_name,
            status="TERMINAL",
            reason=f"Case is in terminal state '{display_name}'. No further SLA timer applies.",
            elapsed_hours=round(elapsed_hours, 2),
            sla_hours=None,
            owner=owner,
            escalation_target=None,
            escalate_after_hours=None,
            escalation_due=False,
            chase_target=None,
        )

    sla_hours = float(state["sla_hours"])
    escalation_target = state.get("escalate_to_role")
    escalate_after_hours = state.get("escalate_after_hours")

    override = _find_urgency_override(workflow, case)
    if override:
        notification = override.get("result_notification", {})
        if isinstance(notification, Mapping) and notification.get("sla_hours") is not None:
            sla_hours = float(notification["sla_hours"])
            # The escalation clock must follow the override; otherwise a
            # critical result is flagged stuck in minutes but not chased
            # for hours.
            escalate_after_hours = sla_hours
        escalation = override.get("escalation", {})
        if isinstance(escalation, Mapping) and escalation.get("enabled"):
            if escalation.get("role"):
                escalation_target = escalation["role"]
            if escalation.get("after_hours") is not None:
                escalate_after_hours = float(escalation["after_hours"])

    escalation_due = (
        escalate_after_hours is not None
        and elapsed_hours > float(escalate_after_hours)
    )
    chase_target = (
        escalation_target if escalation_due
        else (case.current_owner or resolve_role(workflow, state.get("owner_role")))
    )

    overdue = elapsed_hours > sla_hours
    if overdue:
        reason = (
            f"Case has remained in '{display_name}' for {elapsed_hours:.1f} hours, "
            f"exceeding the configured SLA of {sla_hours:g} hours."
        )
        status = "STUCK"
    else:
        remaining = max(sla_hours - elapsed_hours, 0)
        reason = (
            f"Case has remained in '{display_name}' for {elapsed_hours:.1f} hours; "
            f"configured SLA is {sla_hours:g} hours ({remaining:.1f} hours remaining)."
        )
        status = "ON_TIME"

    return Evaluation(
        case_id=case.case_id,
        state=case.current_state,
        display_name=display_name,
        status=status,
        reason=reason,
        elapsed_hours=round(elapsed_hours, 2),
        sla_hours=sla_hours,
        owner=owner,
        escalation_target=str(escalation_target) if escalation_target else None,
        escalate_after_hours=float(escalate_after_hours) if escalate_after_hours is not None else None,
        escalation_due=escalation_due,
        chase_target=chase_target,
    )


def _actor_is_authorized(case: CaseRecord, state: Mapping[str, Any], event: WorkflowEvent) -> bool:
    allowed_roles = {str(state["owner_role"])}
    escalation_role = state.get("escalate_to_role")
    if escalation_role:
        allowed_roles.add(str(escalation_role))
    return event.actor_role in allowed_roles


def _event_target(event_type: str, current_state: Mapping[str, Any]) -> Optional[str]:
    mapping = current_state.get("event_targets", {})
    if isinstance(mapping, Mapping):
        target = mapping.get(event_type)
        return str(target) if target else None
    return None


def apply_event(
    case: CaseRecord,
    event: WorkflowEvent,
    workflow: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
) -> TransitionResult:
    """Apply one workflow event, enforcing legal transitions and audit rules."""
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise CaseValidationError("now must include a timezone offset")

    if event.received_at > now:
        raise TransitionRejected("Event received_at cannot be in the future.")
    if event.occurred_at > now:
        raise TransitionRejected("Event occurred_at cannot be in the future.")
    if event.occurred_at < case.state_entered_at:
        raise TransitionRejected(
            "Event occurred_at cannot precede the current state's state_entered_at; "
            "that would create a negative state duration."
        )

    if event.event_id in case.processed_event_ids:
        return TransitionResult(
            applied=False,
            duplicate=True,
            case=case,
            history_entry=None,
            message=f"Event {event.event_id!r} already processed; ignored for idempotency.",
        )

    states = _state_map(workflow)
    current = states.get(case.current_state)
    if current is None:
        raise TransitionRejected(f"Unknown current state: {case.current_state!r}")

    if bool(current.get("terminal", False)):
        raise TransitionRejected(f"Case is already in terminal state {case.current_state!r}.")

    if not _actor_is_authorized(case, current, event):
        raise TransitionRejected(
            f"Actor role {event.actor_role!r} is not authorized for state {case.current_state!r}."
        )

    target_state = _event_target(event.event_type, current)
    if target_state is None:
        if event.event_type == "force_close":
            target_state = "closed"
        elif event.event_type == "cancel":
            target_state = "cancelled"
        else:
            allowed_next = {str(x) for x in current.get("next_states", [])}
            raise TransitionRejected(
                f"Illegal transition: event {event.event_type!r} has no configured target from "
                f"state {case.current_state!r}. Allowed next states: {sorted(allowed_next)}."
            )

    allowed_next = {str(x) for x in current.get("next_states", [])}
    if target_state not in allowed_next:
        raise TransitionRejected(
            f"Illegal transition: {case.current_state!r} -> {target_state!r}. "
            f"Allowed next states: {sorted(allowed_next)}."
        )

    if target_state not in states:
        raise TransitionRejected(f"Target state {target_state!r} is not defined in workflow.yaml.")

    forced = event.event_type == "force_close"
    if forced:
        if target_state != "closed":
            raise TransitionRejected("force_close must target 'closed'.")
        if not event.reason or not event.reason.strip():
            raise TransitionRejected("Forced closure requires a non-empty reason.")
        # A force-close is restricted to the current escalation/supervisor role.
        escalation_role = current.get("escalate_to_role")
        if event.actor_role != escalation_role:
            raise TransitionRejected(
                "Forced closure requires the current state's escalation/supervisor role."
            )

    if event.event_type == "cancel" and (not event.reason or not event.reason.strip()):
        raise TransitionRejected("Cancellation requires a reason.")

    if target_state == "closed" and case.current_state == "result_reviewed":
        if not event.reason or not event.reason.strip():
            raise TransitionRejected(
                "Direct closure from result_reviewed requires a closure reason, "
                "such as 'follow-up not required'."
            )

    completed = HistoryEntry(
        state=case.current_state,
        entered_at=case.state_entered_at,
        completed_at=event.occurred_at,
        completed_by=event.actor_id,
        completed_by_role=event.actor_role,
        evidence_recorded=event.evidence,
        event_id=event.event_id,
        event_type=event.event_type,
        note=event.note,
        forced=forced,
        force_close_reason=event.reason if forced else None,
    )

    target = states[target_state]
    is_terminal = bool(target.get("terminal", False))
    new_owner_role = str(target.get("owner_role", case.current_owner_role))
    new_owner = case.current_owner

    # If the actor is the target state's owner role, make them the current owner.
    if event.actor_role == new_owner_role:
        new_owner = event.actor_id
    elif target_state in {"closed", "cancelled"}:
        new_owner = event.actor_id
    elif new_owner_role != case.current_owner_role:
        # A handover: the stage now belongs to a different role, so the case
        # belongs to whoever holds that role. Keeping the previous owner's
        # name against the new role says a nurse owns a case that is sitting
        # with the laboratory, and anything reading ownership by name — a
        # board, a chase, a report of who works when — then reads it wrong.
        # resolve_role goes to the roster first and the role's display name
        # second, so no person is ever invented.
        new_owner = resolve_role(workflow, new_owner_role) or case.current_owner

    closure_status = "open"
    if target_state == "closed":
        closure_status = "forced_closed" if forced else "closed"
    elif target_state == "cancelled":
        closure_status = "cancelled"

    updated = CaseRecord(
        case_id=case.case_id,
        patient_id_or_local_identifier=case.patient_id_or_local_identifier,
        test_type=case.test_type,
        ordering_clinician=case.ordering_clinician,
        current_owner=new_owner,
        current_owner_role=new_owner_role,
        current_state=target_state,
        state_entered_at=event.occurred_at,
        last_action_at=event.occurred_at,
        next_action=None if is_terminal else target.get("next_action"),
        escalation_target=target.get("escalate_to_role"),
        result_urgency=case.result_urgency,
        closure_status=closure_status,
        history=case.history + (completed,),
        processed_event_ids=case.processed_event_ids + (event.event_id,),
    )

    return TransitionResult(
        applied=True,
        duplicate=False,
        case=updated,
        history_entry=completed,
        message=(
            f"Applied event {event.event_id!r}: {case.current_state!r} -> {target_state!r}."
        ),
    )


def evaluate_case_dict(case_data: Mapping[str, Any], workflow_path: str | Path, *, now: Optional[datetime] = None) -> dict[str, Any]:
    workflow = load_workflow(workflow_path)
    case = CaseRecord.from_dict(case_data)
    return evaluate_case(case, workflow, now=now).to_dict()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="CareChaser workflow engine")
    parser.add_argument("workflow", help="Path to workflow.yaml")
    parser.add_argument("case", help="Path to case JSON")
    args = parser.parse_args()

    with open(args.case, "r", encoding="utf-8") as handle:
        case_data = json.load(handle)

    workflow = load_workflow(args.workflow)
    case = CaseRecord.from_dict(case_data)
    result = evaluate_case(case, workflow)
    print(json.dumps(result.to_dict(), indent=2))
