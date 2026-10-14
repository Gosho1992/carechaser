"""CareChaser MCP server.

Exposes the deterministic workflow engine as MCP tools over Streamable HTTP,
so an agent (Alexa+, or the local web client) can ask what is stuck, who owns
it, and record acknowledgements and escalations.

Boundary that matters: every verdict in every response comes from
carechaser_engine. This layer only shapes input and output. It never decides
whether a case is stuck, never interprets a result, and never invents a
person's name.

Run:
    python carechaser_mcp.py                 # Streamable HTTP on :8000/mcp
    CARECHASER_SIMULATED_NOW=... python carechaser_mcp.py   # frozen clock
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from carechaser_engine import (
    CaseRecord,
    CaseValidationError,
    Evaluation,
    TransitionRejected,
    resolve_role,
)
from case_store import CaseNotFound, CaseStore
from chase import ChaseFacts, Chaser, build_channel, build_drafter
from democlock import DemoClock, describe as describe_clock
from followups import FollowUpLog, describe_gap, unresolved_followups
from intake import build_extractor, extract_report as run_intake

ROOT = Path(__file__).parent
CASES_PATH = Path(os.environ.get("CARECHASER_CASES", ROOT / "carechaser_cases.json"))
WORKFLOW_PATH = Path(os.environ.get("CARECHASER_WORKFLOW", ROOT / "workflow.yaml"))

store = CaseStore(CASES_PATH, WORKFLOW_PATH)
# Two separate artifacts, deliberately:
#   chase_log.jsonl   - what CareChaser remembers doing (its own clock)
#   chase_outbox.jsonl- what the log channel actually delivered
# Writing both to one file produced duplicate entries on two different clocks.
FOLLOWUP_PATH = os.environ.get("CARECHASER_CHASE_LOG", "chase_log.jsonl")
OUTBOX_PATH = os.environ.get(
    "CARECHASER_OUTBOX", str(Path(FOLLOWUP_PATH).with_name("chase_outbox.jsonl"))
)
if not any(os.environ.get(k) for k in ("CARECHASER_WEBHOOK_URL", "CARECHASER_SES_SENDER")):
    os.environ.setdefault("CARECHASER_CHASE_LOG_CHANNEL", OUTBOX_PATH)

chaser = Chaser(drafter=build_drafter(), channel=build_channel(OUTBOX_PATH))
followup_log = FollowUpLog(FOLLOWUP_PATH)
clock = DemoClock(os.environ.get("CARECHASER_CLOCK_FILE", "demo_clock.json"))
extractor = build_extractor()

mcp = MCPServer(
    name="carechaser",
    title="CareChaser",
    version="0.5.0",
    instructions=(
        "CareChaser tracks diagnostic test results through a clinical workflow "
        "and reports which cases are stuck, who owns them, and how overdue they "
        "are. It is a workflow and audit tool: it never interprets a laboratory "
        "result, never diagnoses, and never recommends treatment. When asked "
        "anything clinical, say that CareChaser only tracks workflow status and "
        "that the responsible clinician must be consulted."
    ),
)


# --------------------------------------------------------------------- utils


def _now() -> datetime:
    """Current time for every judgement this server makes.

    Delegates to the demo clock so a judge can move time forward from the
    browser without restarting anything. Reads the offset from disk on each
    call: the web client runs in a different process.
    """
    return clock.now()


def _person(role_or_name: Optional[str]) -> Optional[str]:
    """Resolve a role id to the person currently holding it, for speech."""
    if not role_or_name:
        return None
    return resolve_role(store.workflow, role_or_name)


def _overdue_by(evaluation: Evaluation) -> Optional[float]:
    if evaluation.sla_hours is None:
        return None
    return round(evaluation.elapsed_hours - evaluation.sla_hours, 2)


def _summarise(case: CaseRecord, evaluation: Evaluation) -> dict[str, Any]:
    """One case as a flat dict an agent can read aloud without post-processing."""
    return {
        "case_id": case.case_id,
        "patient": case.patient_id_or_local_identifier,
        "test_type": case.test_type,
        "state": evaluation.display_name,
        "state_id": evaluation.state,
        "status": evaluation.status,
        "urgency": case.result_urgency,
        "hours_in_state": evaluation.elapsed_hours,
        "sla_hours": evaluation.sla_hours,
        "hours_overdue": _overdue_by(evaluation),
        "owner": case.current_owner,
        "owner_role": case.current_owner_role,
        "escalation_due": evaluation.escalation_due,
        "chase": _person(evaluation.chase_target),
        "escalation_target": _person(evaluation.escalation_target),
        "reason": evaluation.reason,
    }


def _refresh() -> None:
    """Pick up edits made by other processes (the dashboard, a seed script)."""
    store.reload()


# --------------------------------------------------------------------- tools


@mcp.tool(
    title="Get stuck workflows",
    description=(
        "List diagnostic cases that have exceeded their configured time limit, "
        "worst first. Use for questions like 'what is stuck today', 'anything "
        "overdue', or 'what needs chasing'. Optionally filter to cases at least "
        "N hours past their limit, or to one person or role."
    ),
)
def get_stuck_workflows(
    min_hours_overdue: float = 0.0,
    owner: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Cases past their SLA. min_hours_overdue filters on lateness, not elapsed time."""
    _refresh()
    rows = store.stuck_cases(
        now=_now(), min_hours_overdue=min_hours_overdue, owner=owner
    )
    cases = [_summarise(case, ev) for case, ev in rows[: max(limit, 1)]]
    if not cases:
        headline = "Nothing is stuck right now."
    else:
        worst = cases[0]
        lead = (
            "Most urgent"
            if str(worst.get("urgency", "")).lower() in {"critical", "urgent", "critical_result"}
            else "Longest waiting"
        )
        hours = worst["hours_in_state"]
        spoken_hours = f"{hours * 60:.0f} minutes" if hours < 1 else f"{hours:.0f} hours"
        headline = (
            f"{len(rows)} case{'s' if len(rows) != 1 else ''} stuck. "
            f"{lead}: {worst['test_type']} for {worst['patient']}, "
            f"{spoken_hours} in {worst['state']}, owned by {worst['owner']}."
        )
    return {"headline": headline, "total_stuck": len(rows), "cases": cases}


@mcp.tool(
    title="Get case status",
    description=(
        "Status of one case by its case id: which stage it is in, who owns it, "
        "how long it has been there, whether it is overdue, and its history. "
        "Use when a specific case or patient identifier is named."
    ),
)
def get_case_status(case_id: str, include_history: bool = True) -> dict[str, Any]:
    _refresh()
    try:
        case = store.get(case_id)
    except CaseNotFound:
        return {"found": False, "message": f"No case found with id {case_id}."}

    try:
        evaluation = store.evaluate(case_id, now=_now())
    except CaseValidationError as exc:
        return {
            "found": True,
            "case_id": case_id,
            "message": (
                f"{case_id} starts later than the current demo time, so it "
                f"cannot be judged yet. Move the clock forward. ({exc})"
            ),
        }
    payload = {"found": True, **_summarise(case, evaluation)}

    if include_history:
        payload["history"] = [
            {
                "state": entry.state,
                "hours_taken": round(
                    (entry.completed_at - entry.entered_at).total_seconds() / 3600, 2
                ),
                "completed_by": entry.completed_by,
                "completed_at": entry.completed_at.isoformat(),
                "forced": entry.forced,
            }
            for entry in case.history
        ]
    return payload


@mcp.tool(
    title="Get my pending cases",
    description=(
        "Open cases currently waiting on a named person or role, most delayed "
        "first. Use for 'what is waiting on me', 'what does Dr Khan still owe', "
        "or 'what is pending for the lab'. Accepts a person's name, a role id, "
        "or the person rostered to a role."
    ),
)
def get_my_pending(who: str, limit: int = 10) -> dict[str, Any]:
    _refresh()
    rows = store.pending_for(who, now=_now())
    cases = [_summarise(case, ev) for case, ev in rows[: max(limit, 1)]]
    overdue = sum(1 for row in rows if row[1].status == "STUCK")
    if not cases:
        headline = f"Nothing is currently pending for {who}."
    else:
        headline = (
            f"{len(rows)} case{'s' if len(rows) != 1 else ''} pending for {who}"
            + (f", {overdue} overdue." if overdue else ".")
        )
    return {
        "headline": headline,
        "total_pending": len(rows),
        "total_overdue": overdue,
        "cases": cases,
    }


@mcp.tool(
    title="Acknowledge a case",
    description=(
        "Record that the responsible person has acted on a case, moving it to "
        "the next stage and stopping the chase. Use for 'mark it reviewed', "
        "'I have seen it', or 'the patient has been informed'. Requires who is "
        "acknowledging and their role. This records a workflow action only; it "
        "records no clinical judgement."
    ),
)
def acknowledge_result(
    case_id: str,
    actor_id: str,
    actor_role: str,
    event_type: str = "result_reviewed",
    reason: str | None = None,
    note: str | None = None,
    evidence: list[str] | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    _refresh()
    occurred = _now()
    # A stable default event id makes a repeated call idempotent rather than
    # writing the same acknowledgement twice.
    resolved_event_id = event_id or f"ack-{case_id}-{event_type}-{occurred.isoformat()}"
    try:
        result = store.apply(
            case_id,
            {
                "event_id": resolved_event_id,
                "event_type": event_type,
                "actor_id": actor_id,
                "actor_role": actor_role,
                "occurred_at": occurred.isoformat(),
                "received_at": occurred.isoformat(),
                "evidence": evidence or [],
                "reason": reason,
                "note": note,
            },
            now=occurred,
        )
    except CaseNotFound:
        return {"ok": False, "message": f"No case found with id {case_id}."}
    except (TransitionRejected, CaseValidationError) as exc:
        return {"ok": False, "message": str(exc)}

    if result.duplicate:
        return {
            "ok": True,
            "duplicate": True,
            "message": "That acknowledgement was already recorded.",
            "case_id": case_id,
            "state": result.case.current_state,
        }

    evaluation = store.evaluate(case_id, now=occurred)
    return {
        "ok": True,
        "duplicate": False,
        "message": (
            f"Recorded. {case_id} moved to {evaluation.display_name}"
            + (f", now with {evaluation.chase_target}." if evaluation.chase_target else ".")
        ),
        **_summarise(result.case, evaluation),
    }


@mcp.tool(
    title="Escalate a case",
    description=(
        "Hand a stuck case to the next person up the chain when the current "
        "owner has not acted. Use for 'escalate it', 'tell the supervisor', or "
        "'who covers for them'. Set confirm to false first to see who would be "
        "contacted without recording anything."
    ),
)
def escalate(case_id: str, confirm: bool = False, reason: str | None = None) -> dict[str, Any]:
    _refresh()
    try:
        case = store.get(case_id)
    except CaseNotFound:
        return {"ok": False, "message": f"No case found with id {case_id}."}

    evaluation = store.evaluate(case_id, now=_now())
    target_role = evaluation.escalation_target
    target_person = _person(target_role)

    if evaluation.status == "TERMINAL":
        return {
            "ok": False,
            "message": f"{case_id} is already closed. Nothing to escalate.",
        }
    if target_role is None:
        return {
            "ok": False,
            "message": f"No escalation target is configured for {evaluation.display_name}.",
        }

    preview = {
        "case_id": case_id,
        "current_owner": case.current_owner,
        "escalate_to": target_person,
        "escalate_to_role": target_role,
        "escalation_due": evaluation.escalation_due,
        "hours_in_state": evaluation.elapsed_hours,
        "hours_overdue": _overdue_by(evaluation),
        "state": evaluation.display_name,
    }

    if not confirm:
        due = "is due" if evaluation.escalation_due else "is not yet due"
        return {
            "ok": True,
            "recorded": False,
            "message": (
                f"Escalation for {case_id} {due}. It would go to {target_person}, "
                f"currently with {case.current_owner} for "
                f"{evaluation.elapsed_hours:.0f} hours."
            ),
            **preview,
        }

    # Escalation is a record of a chase, not a state transition: the case does
    # not advance, only the person being chased changes. Recording it is what
    # lets CareChaser remember tomorrow that it already pushed on this case.
    followup_log.record(
        case_id,
        kind="escalation",
        to=target_person or target_role,
        reason=reason,
        from_owner=case.current_owner,
        at=_now(),
    )
    return {
        "ok": True,
        "recorded": True,
        "message": (
            f"{case_id} escalated to {target_person}. The case stays in "
            f"{evaluation.display_name} until someone acts on it."
        ),
        "reason": reason,
        **preview,
    }


@mcp.tool(
    title="Send a chase message",
    description=(
        "Draft and send a follow-up message to whoever the case is currently "
        "waiting on. Use for 'chase it', 'send them a reminder', or 'let the "
        "supervisor know'. Set confirm to false first to read the draft aloud "
        "without sending it. The message states workflow facts only."
    ),
)
def send_chase(case_id: str, confirm: bool = False) -> dict[str, Any]:
    _refresh()
    try:
        case = store.get(case_id)
    except CaseNotFound:
        return {"ok": False, "message": f"No case found with id {case_id}."}

    evaluation = store.evaluate(case_id, now=_now())
    if evaluation.status == "TERMINAL":
        return {"ok": False, "message": f"{case_id} is closed. There is nobody to chase."}

    target = _person(evaluation.chase_target)
    if not target:
        return {"ok": False, "message": f"No one is assigned to {case_id} right now."}

    facts = ChaseFacts(
        case_id=case.case_id,
        test_type=case.test_type,
        patient=case.patient_id_or_local_identifier,
        state=evaluation.display_name,
        hours_in_state=evaluation.elapsed_hours,
        sla_hours=evaluation.sla_hours,
        owner=case.current_owner,
        chase_target=target,
        urgency=case.result_urgency,
        escalated=evaluation.escalation_due,
    )
    record = chaser.send(facts, dry_run=not confirm)
    if confirm and record.delivered:
        followup_log.record(
            case_id,
            kind="chase",
            to=target,
            text=record.text,
            from_owner=case.current_owner,
            at=_now(),
        )
    verb = "Draft for" if not confirm else f"Sent to {target} by {record.channel}:"
    return {
        "ok": True,
        "sent": record.delivered,
        "message": f"{verb} {record.text}",
        **record.to_dict(),
    }


@mcp.tool(
    title="Get follow-up briefing",
    description=(
        "What CareChaser already chased or escalated that still has not moved. "
        "Use to open a shift or a new conversation: 'anything I should know', "
        "'what did we chase yesterday', 'any follow-ups outstanding'. This is "
        "how CareChaser carries context between conversations."
    ),
)
def get_followups(min_hours_since: float = 0.0, limit: int = 10) -> dict[str, Any]:
    _refresh()
    now = _now()
    rows = unresolved_followups(store, followup_log, now=now, min_hours_since=min_hours_since)
    shown = rows[: max(limit, 1)]
    for row in shown:
        row["when"] = describe_gap(row["hours_since_followup"])

    if not shown:
        headline = "Nothing outstanding from earlier follow-ups."
    else:
        first = shown[0]
        verb = "escalated" if first["last_kind"] == "escalation" else "chased"
        headline = (
            f"You {verb} {first['case_id']} {first['when']} to {first['last_to']}. "
            f"It is still in {first['state']}, now {first['hours_in_state']:.0f} hours."
        )
        if len(rows) > 1:
            headline += f" {len(rows) - 1} other follow-up{'s' if len(rows) > 2 else ''} still open."
    return {"headline": headline, "total": len(rows), "cases": shown}


def _transport_security():
    """Allow the hostnames this server will actually be reached on.

    The SDK enables DNS-rebinding protection with an empty allow-list, so a
    request arriving through a tunnel or load balancer is rejected on its
    Host header alone. Listing the public hostname is the fix; disabling the
    protection entirely is not.

        CARECHASER_ALLOWED_HOSTS="abc123.trycloudflare.com,carechaser.example.com"
    """
    from mcp.server.transport_security import TransportSecuritySettings

    raw = os.environ.get("CARECHASER_ALLOWED_HOSTS", "")
    extra = [h.strip() for h in raw.split(",") if h.strip()]
    if not extra:
        return None
    hosts = ["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*"]
    origins = []
    for host in extra:
        hosts += [host, f"{host}:*"]
        origins += [f"https://{host}", f"http://{host}"]
    return TransportSecuritySettings(allowed_hosts=hosts, allowed_origins=origins)


@mcp.tool(
    title="Read a report",
    description=(
        "Read a hospital report and list which tests are on it and which are "
        "still outstanding, ready for a person to confirm. Use for 'take this "
        "report', 'what is left on this', 'two are done, what about the rest'. "
        "This only reads: no case is created until someone confirms the rows. "
        "Result values are never read."
    ),
)
def read_report(text: str) -> dict[str, Any]:
    _refresh()
    result = run_intake(text, store.workflow, extractor)
    if not result.get("ok"):
        return result

    ready = [row for row in result["rows"] if row["ok"]]
    done = [row for row in ready if row["state"] in {"result_produced", "result_reviewed", "patient_informed"}]
    outstanding = [row for row in ready if row not in done]
    result["spoken"] = (
        f"{len(result['rows'])} tests on this report. "
        f"{len(done)} {'has' if len(done) == 1 else 'have'} results, "
        f"{len(outstanding)} {'is' if len(outstanding) == 1 else 'are'} still outstanding"
        + (f", and {result['flagged']} {'needs' if result['flagged'] == 1 else 'need'} checking"
           if result["flagged"] else "")
        + ". Nothing is created until you confirm."
    )
    return result


@mcp.tool(
    title="Confirm report rows",
    description=(
        "Create cases from rows a person has reviewed after reading a report. "
        "Use only after the rows have been seen and corrected. Each row needs "
        "a test name, a stage and an owner role; rows still carrying problems "
        "are skipped rather than guessed at."
    ),
)
def confirm_report_rows(
    rows: list[dict[str, Any]], patient: str | None = None
) -> dict[str, Any]:
    _refresh()
    created, skipped = [], []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("test_name", "")).strip()
        state = row.get("state")
        if not name or not state:
            skipped.append({"test_name": name or "(unnamed)", "why": "missing test name or stage"})
            continue
        result = create_case(
            test_type=name,
            current_state=str(state),
            owner_role=row.get("owner_role"),
            hours_waiting=float(row.get("hours_waiting") or 0.0),
            patient=row.get("patient") or patient,
            urgency=str(row.get("urgency") or "routine"),
        )
        if result.get("ok"):
            created.append({"case_id": result["case_id"], "test_type": name,
                            "status": result["status"]})
            # Where this case came from. The sanitised name is recorded;
            # the raw line is NOT, because it may contain a result value and
            # the audit log is permanent. How many fragments were removed is
            # kept, so the record shows that sanitising happened without
            # storing what it removed.
            dropped = len(row.get("removed") or [])
            followup_log.record(
                result["case_id"],
                kind="intake",
                to=row.get("patient") or patient or "unknown patient",
                text=f"report line: {name}",
                reason=(
                    f"{dropped} clinical fragment{'s' if dropped != 1 else ''} removed at intake"
                    if dropped else None
                ),
                at=_now(),
            )
        else:
            skipped.append({"test_name": name, "why": result.get("message", "rejected")})

    overdue = sum(1 for c in created if c["status"] == "STUCK")
    message = (
        f"Created {len(created)} case{'s' if len(created) != 1 else ''}"
        + (f", {overdue} already overdue" if overdue else "")
        + (f". {len(skipped)} skipped" if skipped else "")
        + "."
    )
    return {"ok": bool(created), "message": message,
            "created": created, "skipped": skipped}


@mcp.tool(
    title="Get the board snapshot",
    description=(
        "Everything the command centre shows, read at one instant: the clock, "
        "the counts, the stuck queue and the open follow-ups. Use when a whole "
        "picture is wanted rather than one answer."
    ),
)
def get_board_snapshot(limit: int = 50) -> dict[str, Any]:
    """One timestamp for the whole board.

    Reading the clock, the queue and the follow-ups as three separate calls
    lets the panels disagree: a case can change state between calls and the
    screen then shows two different moments at once. For a tool whose whole
    claim is an auditable record, that is not acceptable, so every figure
    here comes from a single `now`.
    """
    _refresh()
    now = _now()
    clock_state = clock.state()

    stuck = store.stuck_cases(now=now)
    cases = [_summarise(case, ev) for case, ev in stuck[: max(limit, 1)]]

    followups = unresolved_followups(store, followup_log, now=now)
    for row in followups:
        row["when"] = describe_gap(row["hours_since_followup"])

    critical_stuck = sum(
        1 for row in cases
        if str(row.get("urgency", "")).lower() in {"critical", "urgent", "critical_result"}
    )
    return {
        "generated_at": now.isoformat(),
        "clock": clock_state.to_dict(),
        "counts": {
            "stuck": len(stuck),
            "critical_stuck": critical_stuck,
            "escalation_due": sum(1 for row in cases if row.get("escalation_due")),
            "followups": len(followups),
        },
        "cases": cases,
        "followups": followups[:20],
    }


@mcp.tool(
    title="Get a case timeline",
    description=(
        "The full story of one case: every stage it passed through, who "
        "completed it, and every chase or escalation CareChaser recorded, in "
        "order. Use for 'what happened with CC-001', 'show me the history', "
        "'has anyone chased this'."
    ),
)
def get_case_timeline(case_id: str) -> dict[str, Any]:
    _refresh()
    try:
        case = store.get(case_id)
    except CaseNotFound:
        return {"found": False, "message": f"No case found with id {case_id}."}

    now = _now()
    evaluation = store.evaluate(case_id, now=now)
    events: list[dict[str, Any]] = []

    for entry in case.history:
        hours = (entry.completed_at - entry.entered_at).total_seconds() / 3600
        events.append({
            "at": entry.completed_at.isoformat(),
            "kind": "transition",
            "title": entry.state.replace("_", " ").title() + " completed",
            "detail": f"{hours:.0f} hours, by {entry.completed_by}",
            "forced": entry.forced,
        })

    for follow_up in followup_log.for_case(case_id):
        if follow_up.kind == "intake":
            title = f"Created from a report for {follow_up.to}"
            detail = follow_up.text
            if follow_up.reason:
                detail = f"{detail} · {follow_up.reason}"
        elif follow_up.kind == "escalation":
            title = "Escalated to " + follow_up.to
            detail = follow_up.reason or follow_up.text or ""
        else:
            title = "Chased " + follow_up.to
            detail = follow_up.reason or follow_up.text or ""
        events.append({
            "at": follow_up.at.isoformat(),
            "kind": follow_up.kind,
            "title": title,
            "detail": detail,
            "when": describe_gap(follow_up.hours_ago(now)),
        })

    events.sort(key=lambda event: event["at"])
    # The "now" marker closes the timeline, so it is stamped with the current
    # time rather than when the stage began; otherwise a long-running stage
    # sorts its own marker above events that happened during it.
    events.append({
        "at": now.isoformat(),
        "kind": "current",
        "title": f"Now: {evaluation.display_name}",
        "detail": (
            f"{evaluation.elapsed_hours:.0f} hours with {case.current_owner}"
            + (f", {_overdue_by(evaluation):.0f} hours overdue"
               if evaluation.status == "STUCK" else "")
        ),
    })

    return {"found": True, "events": events, **_summarise(case, evaluation)}


@mcp.tool(
    title="Create a case",
    description=(
        "Add a diagnostic case to the board, for trying CareChaser out with "
        "your own scenario. Use for 'create a case', 'add a biopsy stuck for "
        "three days', 'make a test case'. hours_waiting says how long it has "
        "already been sitting in that stage, so a case can be created already "
        "overdue without waiting."
    ),
)
def create_case(
    test_type: str,
    current_state: str = "result_produced",
    owner: str | None = None,
    owner_role: str | None = None,
    hours_waiting: float = 0.0,
    patient: str | None = None,
    urgency: str = "routine",
    case_id: str | None = None,
) -> dict[str, Any]:
    _refresh()
    states = {s["state"]: s for s in store.workflow["states"]}
    startable = {k: v for k, v in states.items() if not v.get("terminal")}
    if current_state not in states:
        return {
            "ok": False,
            "message": (
                f"'{current_state}' is not a stage in this workflow. "
                f"Stages are: {', '.join(startable)}."
            ),
        }
    if current_state not in startable:
        # A terminal stage carries no timer, so a case created there could
        # never be stuck, acted on, or chased. Nothing to demonstrate.
        return {
            "ok": False,
            "message": (
                f"'{current_state}' is a terminal stage. Create the case at an "
                f"earlier stage and close it from there."
            ),
        }
    if hours_waiting < 0:
        return {"ok": False, "message": "hours_waiting cannot be negative."}

    state_config = states[current_state]
    role = owner_role or str(state_config.get("owner_role"))
    if role not in (store.workflow.get("roles") or {}):
        return {"ok": False, "message": f"'{role}' is not a role in this workflow."}

    # Generated ids continue the CC- series rather than colliding with it.
    if not case_id:
        used = [
            int(c.case_id.split("-")[-1])
            for c in store
            if c.case_id.startswith("CC-") and c.case_id.split("-")[-1].isdigit()
        ]
        case_id = f"CC-{max(used, default=0) + 1:03d}"
    if store.exists(case_id):
        return {"ok": False, "message": f"Case {case_id} already exists."}

    now = _now()
    entered = now - timedelta(hours=hours_waiting)
    record = {
        "case_id": case_id,
        "patient_id_or_local_identifier": patient or f"DEMO-{case_id}",
        "test_type": test_type,
        "ordering_clinician": owner or _person(role) or role,
        "current_owner": owner or _person(role) or role,
        "current_owner_role": role,
        "current_state": current_state,
        "state_entered_at": entered.isoformat(),
        "last_action_at": entered.isoformat(),
        "result_urgency": urgency,
        "closure_status": "open",
    }
    try:
        store.add(record)
    except (ValueError, CaseValidationError) as exc:
        return {"ok": False, "message": str(exc)}

    evaluation = store.evaluate(case_id, now=now)
    verdict = "already overdue" if evaluation.status == "STUCK" else "on time"
    return {
        "ok": True,
        "message": (
            f"Created {case_id}: {test_type} at {evaluation.display_name}, "
            f"waiting {evaluation.elapsed_hours:.0f} hours. It is {verdict}."
        ),
        **_summarise(store.get(case_id), evaluation),
    }


@mcp.tool(
    title="Read the ward's own history",
    description=(
        "What the audit trail shows about how this ward works: where time "
        "goes, which hours each person clears results in, and which weekday "
        "backs up. Use for 'where are we losing time', 'when is Dr Khan "
        "actually free', 'which day is worst', 'when should I chase'. Every "
        "figure is counted from recorded timestamps, and anything with too "
        "few observations is reported as thin rather than stated as fact."
    ),
)
def get_insights(limit_people: int = 10) -> dict[str, Any]:
    """Read the history back. Nothing here is a clinical judgement."""
    _refresh()
    import insights as insights_module

    result = insights_module.build(
        store.all_cases(), store.workflow, now=_now()
    ).to_dict()
    result["people"] = result["people"][:limit_people]
    result["spoken"] = result["headline"]
    result["ok"] = True
    return result


@mcp.tool(
    title="Get the demo clock",
    description=(
        "What time CareChaser currently thinks it is, and how far the demo "
        "clock has been moved forward. Use for 'what time is it', 'where are "
        "we in the demo', or before explaining why something is overdue."
    ),
)
def get_demo_clock() -> dict[str, Any]:
    state = clock.state()
    return {"message": describe_clock(state), **state.to_dict()}


@mcp.tool(
    title="Advance the demo clock",
    description=(
        "Move the demo clock forward by a number of hours so time-based "
        "behaviour can be seen without waiting. Use for 'skip ahead a day', "
        "'advance 24 hours', 'what happens tomorrow'. Time only moves "
        "forward; use reset to start over."
    ),
)
def advance_demo_clock(hours: float = 24.0) -> dict[str, Any]:
    try:
        state = clock.advance(hours)
    except ValueError as exc:
        return {"ok": False, "message": str(exc), **clock.state().to_dict()}

    stuck = store.stuck_cases(now=state.now)
    return {
        "ok": True,
        "message": (
            f"Moved forward {hours:g} hours. {describe_clock(state)} "
            f"{len(stuck)} case{'s' if len(stuck) != 1 else ''} now stuck."
        ),
        "total_stuck": len(stuck),
        **state.to_dict(),
    }


@mcp.tool(
    title="Reset the demo clock",
    description=(
        "Put the demo clock back to its starting point. Use for 'reset the "
        "clock', 'start the demo again', 'back to the beginning'."
    ),
)
def reset_demo_clock() -> dict[str, Any]:
    try:
        state = clock.reset()
    except ValueError as exc:
        return {"ok": False, "message": str(exc), **clock.state().to_dict()}
    return {"ok": True, "message": f"Clock reset. {describe_clock(state)}", **state.to_dict()}


if __name__ == "__main__":
    # AgentCore Runtime expects the container on 0.0.0.0:8000/mcp and requires
    # a STATELESS streamable-HTTP server: it isolates sessions itself and
    # injects the Mcp-Session-Id header. Locally we keep the default stateful
    # mode, which is friendlier to the MCP Inspector.
    agentcore = os.environ.get("CARECHASER_AGENTCORE") == "1"
    host = os.environ.get("CARECHASER_HOST", "0.0.0.0" if agentcore else "127.0.0.1")
    port = int(os.environ.get("CARECHASER_PORT", "8000"))
    stateless = agentcore or os.environ.get("CARECHASER_STATELESS") == "1"
    print(f"CareChaser MCP server on http://{host}:{port}/mcp")
    print(f"  cases:    {CASES_PATH}")
    print(f"  workflow: {WORKFLOW_PATH}")
    if os.environ.get("CARECHASER_SIMULATED_NOW"):
        print(f"  clock:    SIMULATED {os.environ['CARECHASER_SIMULATED_NOW']}")
    security = _transport_security()
    if security:
        print(f"  hosts:    {os.environ['CARECHASER_ALLOWED_HOSTS']}")
    if stateless:
        print("  mode:     stateless (AgentCore compatible)")
    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        stateless_http=stateless,
        transport_security=security,
    )
