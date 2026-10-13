"""CareChaser follow-up log.

Records every chase and escalation so CareChaser can remember, in a later
conversation, that it already pushed on a case and nobody acted.

    Day 1: "escalate CC-001"        -> written here
    Day 2: "what should I know?"    -> read from here
            "You escalated CC-001 yesterday. Still no acknowledgement,
             now 55 hours."

Append-only JSONL, same file the LogChannel writes to, so the demo's visible
outbox and the memory behind the briefing are the same artifact. Nothing is
ever rewritten: a follow-up that happened stays on the record even after the
case closes.

The log holds actions CareChaser took. Whether the case has since moved is
answered by the engine, never by this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class FollowUp:
    case_id: str
    kind: str  # chase | escalation
    to: str
    at: datetime
    text: str = ""
    reason: Optional[str] = None
    from_owner: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FollowUp":
        raw = str(data.get("at", ""))
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        at = datetime.fromisoformat(raw) if raw else datetime.now(timezone.utc)
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        return cls(
            case_id=str(data.get("case_id", "")),
            kind=str(data.get("kind", "chase")),
            to=str(data.get("to", "")),
            at=at,
            text=str(data.get("text", "")),
            reason=data.get("reason"),
            from_owner=data.get("from_owner"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "to": self.to,
            "at": self.at.isoformat(),
            "text": self.text,
            "reason": self.reason,
            "from_owner": self.from_owner,
        }

    def hours_ago(self, now: datetime) -> float:
        delta = now.astimezone(timezone.utc) - self.at.astimezone(timezone.utc)
        return round(delta.total_seconds() / 3600, 2)


class FollowUpLog:
    """Append-only log of chases and escalations, grouped by case."""

    def __init__(self, path: str | Path = "chase_log.jsonl"):
        self.path = Path(path)

    def append(self, follow_up: FollowUp) -> FollowUp:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(follow_up.to_dict()) + "\n")
        return follow_up

    def record(
        self,
        case_id: str,
        kind: str,
        to: str,
        *,
        text: str = "",
        reason: Optional[str] = None,
        from_owner: Optional[str] = None,
        at: Optional[datetime] = None,
    ) -> FollowUp:
        return self.append(
            FollowUp(
                case_id=case_id,
                kind=kind,
                to=to,
                at=at or datetime.now(timezone.utc),
                text=text,
                reason=reason,
                from_owner=from_owner,
            )
        )

    def all(self) -> list[FollowUp]:
        if not self.path.exists():
            return []
        rows: list[FollowUp] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(FollowUp.from_dict(json.loads(line)))
                except (json.JSONDecodeError, ValueError):
                    continue  # a corrupt line must not hide the rest
        return rows

    def for_case(self, case_id: str) -> list[FollowUp]:
        return [row for row in self.all() if row.case_id == case_id]

    def last_for_case(self, case_id: str) -> Optional[FollowUp]:
        rows = self.for_case(case_id)
        return max(rows, key=lambda row: row.at) if rows else None

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per case: how many follow-ups, and the most recent one."""
        out: dict[str, dict[str, Any]] = {}
        for row in self.all():
            entry = out.setdefault(
                row.case_id, {"count": 0, "last": row, "kinds": set()}
            )
            entry["count"] += 1
            entry["kinds"].add(row.kind)
            if row.at > entry["last"].at:
                entry["last"] = row
        return out


def unresolved_followups(
    store: Any,
    log: FollowUpLog,
    *,
    now: datetime,
    min_hours_since: float = 0.0,
) -> list[dict[str, Any]]:
    """Cases CareChaser already pushed on that still have not moved.

    "Still not moved" is decided by the engine, not by this log: a case whose
    state changed after the follow-up has been acted on, even if nobody told
    CareChaser so. Terminal cases drop out.
    """
    results: list[dict[str, Any]] = []
    for case_id, entry in log.summary().items():
        # Only chases and escalations are things to follow up. Provenance
        # entries (where a case came from) live in the same append-only log
        # but are history, not outstanding work.
        if entry["kinds"] <= {"intake"}:
            continue
        if not store.exists(case_id):
            continue
        case = store.get(case_id)
        try:
            evaluation = store.evaluate(case_id, now=now)
        except Exception:
            # A case whose current stage starts after `now` cannot be judged
            # yet. That happens when the demo clock is reset below a case's
            # last transition, and it must not take the briefing down with it.
            continue
        if evaluation.status == "TERMINAL":
            continue

        last: FollowUp = entry["last"]
        # If the case entered its current state after the follow-up, somebody
        # acted. Nothing to nag about.
        # >= not >: with a frozen demo clock an acknowledgement can land on
        # the same instant as the follow-up, and that still counts as acted on.
        if case.state_entered_at.astimezone(timezone.utc) >= last.at.astimezone(timezone.utc):
            continue

        hours_since = last.hours_ago(now)
        if hours_since < min_hours_since:
            continue

        results.append(
            {
                "case_id": case_id,
                "test_type": case.test_type,
                "patient": case.patient_id_or_local_identifier,
                "state": evaluation.display_name,
                "status": evaluation.status,
                "owner": case.current_owner,
                "hours_in_state": evaluation.elapsed_hours,
                "followups": entry["count"],
                "last_kind": last.kind,
                "last_to": last.to,
                "hours_since_followup": hours_since,
            }
        )

    results.sort(key=lambda row: row["hours_since_followup"], reverse=True)
    return results


def describe_gap(hours: float) -> str:
    """Speakable time gap: 'yesterday' beats '27.4 hours ago' out loud."""
    if hours < 1:
        return "less than an hour ago"
    if hours < 20:
        return f"{round(hours)} hours ago"
    if hours < 36:
        return "yesterday"
    return f"{round(hours / 24)} days ago"
