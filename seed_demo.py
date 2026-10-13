"""Generate the CareChaser demo ward.

Writes a set of cases positioned relative to a fixed demo clock, so the same
story plays out identically on every run and every take of the video.

    python seed_demo.py                 # writes carechaser_cases.json
    python seed_demo.py --out ward.json

The demo clock is 2026-09-28T08:00:00+05:00, a Monday morning ward round.
Run the servers with the same value in CARECHASER_SIMULATED_NOW.

Each case is placed by how long it has been in its current state, not by a
hard-coded date, so the ward always looks the same relative to that clock.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta

DEMO_NOW = datetime.fromisoformat("2026-09-28T08:00:00+05:00")


def ago(hours: float) -> str:
    return (DEMO_NOW - timedelta(hours=hours)).isoformat()


# hours_in_state is chosen against the SLA in workflow.yaml:
#   ordered 24, sample_collected 4, sample_processed 8,
#   result_produced 24, result_reviewed 24, patient_informed 24
WARD = [
    # The headline case: a biopsy result nobody has reviewed for three days.
    dict(
        case_id="CC-001", patient="MK-4471", test="Histopathology (biopsy)",
        state="result_produced", hours=74, owner="Dr. Khan",
        role="responsible_clinician", urgency="routine",
        next_action="Review result and document next action",
    ),
    # Critical result, minutes old, already past the 10-minute window.
    dict(
        case_id="CC-002", patient="MK-4488", test="Serum potassium",
        state="result_produced", hours=0.6, owner="Dr. Ali",
        role="responsible_clinician", urgency="critical",
        next_action="Acknowledge critical result",
    ),
    # Lab backlog: sample sitting unprocessed well past the 8 hour limit.
    dict(
        case_id="CC-003", patient="MK-4502", test="Liver function",
        state="sample_processed", hours=19, owner="Lab Desk",
        role="laboratory_team", urgency="routine",
        next_action="Complete processing and release result",
    ),
    # Reviewed, but the patient still has not been told.
    dict(
        case_id="CC-004", patient="MK-4455", test="Thyroid panel",
        state="result_reviewed", hours=31, owner="Ward 2 Care Team",
        role="responsible_care_team", urgency="routine",
        next_action="Inform patient and document follow-up",
    ),
    # Collection never happened after the order.
    dict(
        case_id="CC-005", patient="MK-4510", test="Blood culture",
        state="ordered", hours=27, owner="Nurse Ayesha",
        role="authorized_sample_collector", urgency="routine",
        next_action="Collect sample",
    ),
    # Healthy cases, so the board is not all red.
    dict(
        case_id="CC-006", patient="MK-4513", test="Full blood count",
        state="sample_processed", hours=3, owner="Lab Desk",
        role="laboratory_team", urgency="routine",
        next_action="Complete processing and release result",
    ),
    dict(
        case_id="CC-007", patient="MK-4499", test="Urea and electrolytes",
        state="result_produced", hours=5, owner="Dr. Ahmed",
        role="responsible_clinician", urgency="routine",
        next_action="Review result",
    ),
    dict(
        case_id="CC-008", patient="MK-4461", test="Chest imaging report",
        state="patient_informed", hours=2, owner="Ward 2 Care Team",
        role="responsible_care_team", urgency="routine",
        next_action="Close the case",
    ),
    # A finished case, to prove closed work disappears from the board.
    dict(
        case_id="CC-009", patient="MK-4433", test="Full blood count",
        state="closed", hours=40, owner="Ward 2 Care Team",
        role="responsible_care_team", urgency="routine",
        next_action=None, closure="closed",
    ),
]


def build() -> list[dict]:
    cases = []
    for item in WARD:
        case = {
            "case_id": item["case_id"],
            "patient_id_or_local_identifier": item["patient"],
            "test_type": item["test"],
            "ordering_clinician": item["owner"] if item["role"].endswith("clinician") else "Dr. Khan",
            "current_owner": item["owner"],
            "current_owner_role": item["role"],
            "current_state": item["state"],
            "state_entered_at": ago(item["hours"]),
            "last_action_at": ago(item["hours"]),
            "result_urgency": item["urgency"],
            "closure_status": item.get("closure", "open"),
        }
        if item.get("next_action"):
            case["next_action"] = item["next_action"]
        cases.append(case)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="carechaser_cases.json")
    args = parser.parse_args()

    cases = build()
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(cases, handle, indent=2)

    print(f"Wrote {len(cases)} cases to {args.out}")
    print(f"Demo clock: {DEMO_NOW.isoformat()}")
    print()
    print("Run the servers with:")
    print(f'  $env:CARECHASER_SIMULATED_NOW="{DEMO_NOW.isoformat()}"')
    print()
    print("Day two of the demo, same cases, clock moved on:")
    print(f'  $env:CARECHASER_SIMULATED_NOW="{(DEMO_NOW + timedelta(days=1)).isoformat()}"')


if __name__ == "__main__":
    main()
