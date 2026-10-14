"""CareChaser case store.

A small persistence layer sitting between the deterministic engine and the
MCP tool layer. Backed by a JSON file so the demo needs no database, but the
interface is deliberately narrow: swapping in Postgres or DynamoDB later
should touch this file only.

Design rules:
- The store never decides anything clinical. It loads, saves and asks the
  engine. All STUCK/ON_TIME/escalation logic stays in carechaser_engine.
- Writes are atomic (temp file + replace) so an interrupted save cannot
  leave a truncated case file behind.
- apply_event persists only when the engine accepts the transition; a
  rejected event leaves the stored case untouched.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from carechaser_engine import (
    CaseRecord,
    CaseValidationError,
    Evaluation,
    WorkflowEvent,
    apply_event,
    evaluate_case,
    load_workflow,
    resolve_role,
)


class CaseNotFound(KeyError):
    pass


class CaseStore:
    """JSON-backed store of cases plus the workflow they run under."""

    def __init__(self, cases_path: str | Path, workflow_path: str | Path) -> None:
        self.cases_path = Path(cases_path)
        self.workflow_path = Path(workflow_path)
        self.workflow = load_workflow(self.workflow_path)
        self._cases: dict[str, CaseRecord] = {}
        self.reload()

    # ---------------------------------------------------------------- load

    def reload(self) -> None:
        """Re-read both the workflow and the cases from disk."""
        self.workflow = load_workflow(self.workflow_path)
        if not self.cases_path.exists():
            self._cases = {}
            return
        with self.cases_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            raise ValueError("Case file must contain a JSON list of cases")
        self._cases = {
            str(item["case_id"]): CaseRecord.from_dict(item) for item in raw
        }

    # ---------------------------------------------------------------- save

    def _save(self) -> None:
        """Write all cases atomically."""
        payload = [case.to_dict() for case in self._cases.values()]
        self.cases_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.cases_path.parent), prefix=".cases-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.cases_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ---------------------------------------------------------------- read

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self) -> Iterator[CaseRecord]:
        return iter(self._cases.values())

    def all_cases(self) -> list[CaseRecord]:
        """Every case, for code that reads the whole history rather than
        asking about one case."""
        return list(self._cases.values())

    def get(self, case_id: str) -> CaseRecord:
        case = self._cases.get(str(case_id))
        if case is None:
            raise CaseNotFound(f"No case with id {case_id!r}")
        return case

    def exists(self, case_id: str) -> bool:
        return str(case_id) in self._cases

    def evaluate(
        self, case_id: str, *, now: Optional[datetime] = None
    ) -> Evaluation:
        return evaluate_case(self.get(case_id), self.workflow, now=now)

    def evaluate_all(
        self, *, now: Optional[datetime] = None
    ) -> list[tuple[CaseRecord, Evaluation]]:
        """Evaluate every case, skipping any that cannot exist yet.

        A case created while the demo clock was ahead has a start time in the
        future once the clock is reset. The engine rightly refuses to judge
        it, but one such case must not take the whole board down, so it is
        skipped: a case that has not started cannot be overdue.
        """
        rows: list[tuple[CaseRecord, Evaluation]] = []
        for case in self._cases.values():
            try:
                rows.append((case, evaluate_case(case, self.workflow, now=now)))
            except CaseValidationError:
                continue
        return rows

    def stuck_cases(
        self,
        *,
        now: Optional[datetime] = None,
        min_hours_overdue: float = 0.0,
        owner: Optional[str] = None,
    ) -> list[tuple[CaseRecord, Evaluation]]:
        """Cases past their SLA, worst first.

        min_hours_overdue filters on how far past the SLA a case is, not on
        raw elapsed time, so a 10-minute critical breach is not hidden by a
        filter meant for day-old routine cases.
        """
        rows: list[tuple[CaseRecord, Evaluation]] = []
        for case, evaluation in self.evaluate_all(now=now):
            if evaluation.status != "STUCK":
                continue
            if evaluation.sla_hours is None:
                continue
            overdue_by = evaluation.elapsed_hours - evaluation.sla_hours
            if overdue_by < min_hours_overdue:
                continue
            if owner and not self._matches_owner(case, evaluation, owner):
                continue
            rows.append((case, evaluation))
        # Urgency outranks lateness: a critical result minutes past its
        # acknowledgement window matters more than a routine case days late.
        def rank(row: tuple[CaseRecord, Evaluation]) -> tuple[int, float]:
            case, evaluation = row
            critical = case.result_urgency.lower() in {"critical", "urgent", "critical_result"}
            overdue_by = evaluation.elapsed_hours - (evaluation.sla_hours or 0.0)
            return (0 if critical else 1, -overdue_by)

        rows.sort(key=rank)
        return rows

    def pending_for(
        self, who: str, *, now: Optional[datetime] = None
    ) -> list[tuple[CaseRecord, Evaluation]]:
        """Open cases where this person or role is on the hook right now."""
        rows = []
        for case, evaluation in self.evaluate_all(now=now):
            if evaluation.status == "TERMINAL":
                continue
            if self._matches_owner(case, evaluation, who):
                rows.append((case, evaluation))
        rows.sort(key=lambda row: row[1].elapsed_hours, reverse=True)
        return rows

    def _matches_owner(
        self, case: CaseRecord, evaluation: Evaluation, who: str
    ) -> bool:
        """Does this case currently sit with the person or role named?

        Deliberately narrow. A role id matches cases carrying that role; a
        person's name matches only cases they personally own. Resolving a
        role back to its rostered person and treating the two as equivalent
        would make "what is waiting on Dr. Khan" quietly mean "what is
        waiting on every responsible clinician" — dangerous in a ward.

        The one extension: once escalation is due, the case really is
        waiting on the escalation target, so it shows up for them too.
        """
        needle = who.strip().lower()
        if not needle:
            return False

        if (case.current_owner or "").strip().lower() == needle:
            return True
        if (case.current_owner_role or "").strip().lower() == needle:
            return True

        if evaluation.escalation_due:
            target_role = (evaluation.escalation_target or "").strip().lower()
            if target_role and target_role == needle:
                return True
            target_person = resolve_role(self.workflow, evaluation.escalation_target)
            if target_person and target_person.strip().lower() == needle:
                return True

        return False

    # --------------------------------------------------------------- write

    def apply(
        self,
        case_id: str,
        event: WorkflowEvent | Mapping[str, Any],
        *,
        now: Optional[datetime] = None,
    ):
        """Apply an event to a stored case and persist the result.

        The engine decides whether the transition is legal. If it raises,
        nothing is written. Duplicate events are persisted as no-ops.
        """
        if not isinstance(event, WorkflowEvent):
            event = WorkflowEvent.from_dict(event)
        case = self.get(case_id)
        result = apply_event(case, event, self.workflow, now=now)
        if result.applied:
            self._cases[case.case_id] = result.case
            self._save()
        return result

    def add(self, case: CaseRecord | Mapping[str, Any]) -> CaseRecord:
        if not isinstance(case, CaseRecord):
            case = CaseRecord.from_dict(case)
        if case.case_id in self._cases:
            raise ValueError(f"Case {case.case_id!r} already exists")
        self._cases[case.case_id] = case
        self._save()
        return case

    def upsert_many(self, cases: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        for item in cases:
            record = CaseRecord.from_dict(item)
            self._cases[record.case_id] = record
            count += 1
        self._save()
        return count


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
