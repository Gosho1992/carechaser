"""Small Step 4 demonstration for CareChaser."""
from datetime import datetime
from pathlib import Path

from carechaser_engine import CaseRecord, WorkflowEvent, apply_event, load_workflow

ROOT = Path(__file__).parent
workflow = load_workflow(ROOT / "workflow.yaml")

case = CaseRecord.from_dict({
    "case_id": "DEMO-TRANSITION-001",
    "patient_id_or_local_identifier": "DEMO-P100",
    "test_type": "CBC",
    "ordering_clinician": "Dr. Khan",
    "current_owner": "Dr. Khan",
    "current_owner_role": "responsible_clinician",
    "current_state": "result_produced",
    "state_entered_at": "2026-09-27T20:00:00+05:00",
    "result_urgency": "routine",
})

event = WorkflowEvent.from_dict({
    "event_id": "EV-DEMO-001",
    "event_type": "result_reviewed",
    "actor_id": "Dr. Khan",
    "actor_role": "responsible_clinician",
    "occurred_at": "2026-09-28T07:00:00+05:00",
    "received_at": "2026-09-28T07:01:00+05:00",
    "evidence": ["Result reviewed", "Next action documented"],
    "note": "Routine result reviewed.",
})

result = apply_event(
    case,
    event,
    workflow,
    now=datetime.fromisoformat("2026-09-28T08:00:00+05:00"),
)

print(result.message)
print("New state:", result.case.current_state)
print("New state entered at:", result.case.state_entered_at.isoformat())
print("History entry:", result.history_entry.to_dict())
