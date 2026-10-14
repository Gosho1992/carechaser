"""End-to-end test of the CareChaser MCP server over Streamable HTTP.

Starts the real server in a subprocess, connects with the MCP client, and
exercises every tool the way an agent would. This is the test that proves the
hackathon requirement is met: a self-hosted MCP server, Streamable HTTP
transport, real protocol handshake.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mcp import Client

ROOT = Path(__file__).parent
FROZEN_NOW = "2026-09-28T08:00:00+05:00"

# Tests own their data: the demo ward in carechaser_cases.json is for the
# video and changes whenever seed_demo.py is rerun.
FIXTURE_CASES = [
    {
        "case_id": "CC-001",
        "patient_id_or_local_identifier": "DEMO-P001",
        "test_type": "CBC",
        "ordering_clinician": "Dr. Khan",
        "current_owner": "Dr. Khan",
        "current_owner_role": "responsible_clinician",
        "current_state": "result_produced",
        "state_entered_at": "2026-09-27T01:00:00+05:00",
        "result_urgency": "routine",
        "closure_status": "open",
    },
    {
        "case_id": "CC-002",
        "patient_id_or_local_identifier": "DEMO-P002",
        "test_type": "LFT",
        "ordering_clinician": "Dr. Ahmed",
        "current_owner": "Lab Desk",
        "current_owner_role": "laboratory_team",
        "current_state": "sample_processed",
        "state_entered_at": "2026-09-28T00:30:00+05:00",
        "result_urgency": "routine",
        "closure_status": "open",
    },
    {
        "case_id": "CC-003",
        "patient_id_or_local_identifier": "DEMO-P003",
        "test_type": "Biopsy",
        "ordering_clinician": "Dr. Ali",
        "current_owner": "Care Team",
        "current_owner_role": "responsible_care_team",
        "current_state": "result_reviewed",
        "state_entered_at": "2026-09-28T06:00:00+05:00",
        "result_urgency": "routine",
        "closure_status": "open",
    },
]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Run the real server against a throwaway copy of the case file."""
    workdir = tmp_path_factory.mktemp("mcp")
    (workdir / "cases.json").write_text(json.dumps(FIXTURE_CASES), encoding="utf-8")

    port = _free_port()
    env = {
        **os.environ,
        "CARECHASER_CASES": str(workdir / "cases.json"),
        "CARECHASER_WORKFLOW": str(ROOT / "workflow.yaml"),
        "CARECHASER_SIMULATED_NOW": FROZEN_NOW,
        "CARECHASER_PORT": str(port),
        "PYTHONPATH": str(ROOT),
        # Never let ambient AWS credentials change what the tests exercise.
        # Without this the extractor and the drafter silently switch to
        # Bedrock on a developer machine that happens to be logged in, and
        # the suite starts depending on a network service and a model's mood.
        "CARECHASER_FORCE_OFFLINE": "1",
    }
    # Logs go to a file, never to an unread pipe. Uvicorn writes a line per
    # request; with stdout=PIPE and nobody reading it, the OS buffer fills and
    # the server blocks forever on its next log write. Every later test then
    # times out, which looks like an application hang and is not one.
    log_path = workdir / "server.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "carechaser_mcp.py")],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    url = f"http://127.0.0.1:{port}/mcp"
    for _ in range(100):
        if proc.poll() is not None:
            log_file.close()
            raise RuntimeError(log_path.read_text(encoding="utf-8"))
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError("server did not start")

    yield url
    proc.terminate()
    proc.wait(timeout=10)
    log_file.close()


async def _call(url: str, tool: str, args: dict) -> dict:
    """Call a tool and parse its JSON result.

    A large result arrives as several text blocks, so they are joined before
    parsing; reading only the first block truncates the JSON mid-object.
    """
    async with Client(url) as client:
        result = await client.call_tool(tool, args, read_timeout_seconds=20)
        payload = "".join(b.text for b in result.content if getattr(b, "text", None))
        if not payload:
            raise AssertionError(
                f"{tool} returned no text. isError={getattr(result, 'isError', None)} "
                f"blocks={[type(b).__name__ for b in result.content]} "
                f"structured={str(getattr(result, 'structuredContent', None))[:300]}")
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            raise AssertionError(f"{tool} returned non-JSON: {payload[:400]!r}")


@pytest.mark.anyio
async def test_handshake_lists_every_tool(server):
    async with Client(server) as client:
        assert client.server_info.name == "carechaser"
        # The hackathon requires spec 2025-11-25 or later.
        assert client.protocol_version >= "2025-11-25"
        tools = await client.list_tools()
        names = {t.name for t in tools.tools}
        assert len(names) == 16, f"tool count changed: {sorted(names)}"
        assert names == {
            "get_stuck_workflows",
            "get_case_status",
            "get_my_pending",
            "acknowledge_result",
            "escalate",
            "send_chase",
            "get_followups",
            "get_demo_clock",
            "advance_demo_clock",
            "reset_demo_clock",
            "create_case",
            "get_case_timeline",
            "get_board_snapshot",
            "read_report",
            "confirm_report_rows",
            "get_insights",
        }


@pytest.mark.anyio
async def test_stuck_workflows_reports_the_overdue_case(server):
    data = await _call(server, "get_stuck_workflows", {})
    assert data["total_stuck"] == 1
    case = data["cases"][0]
    assert case["case_id"] == "CC-001"
    assert case["hours_overdue"] == 7.0
    assert case["chase"] == "Dr. Riaz"  # role resolved to a person


@pytest.mark.anyio
async def test_case_status_includes_history_and_unknown_is_handled(server):
    data = await _call(server, "get_case_status", {"case_id": "CC-001"})
    assert data["found"] is True
    assert data["status"] == "STUCK"
    assert "history" in data

    missing = await _call(server, "get_case_status", {"case_id": "NOPE"})
    assert missing["found"] is False


@pytest.mark.anyio
async def test_pending_accepts_person_role_or_rostered_name(server):
    by_person = await _call(server, "get_my_pending", {"who": "Dr. Khan"})
    assert by_person["total_pending"] == 1
    by_role = await _call(server, "get_my_pending", {"who": "laboratory_team"})
    assert by_role["total_pending"] == 1
    nobody = await _call(server, "get_my_pending", {"who": "Dr. Nobody"})
    assert nobody["total_pending"] == 0


@pytest.mark.anyio
async def test_escalate_preview_does_not_record(server):
    preview = await _call(server, "escalate", {"case_id": "CC-001"})
    assert preview["recorded"] is False
    assert preview["escalate_to"] == "Dr. Riaz"
    assert preview["escalation_due"] is True

    confirmed = await _call(server, "escalate", {"case_id": "CC-001", "confirm": True})
    assert confirmed["recorded"] is True
    # Escalation chases a person; it must not advance the workflow state.
    after = await _call(server, "get_case_status", {"case_id": "CC-001"})
    assert after["state_id"] == "result_produced"


@pytest.mark.anyio
async def test_unauthorized_acknowledgement_is_refused_not_crashed(server):
    data = await _call(
        server,
        "acknowledge_result",
        {
            "case_id": "CC-001",
            "actor_id": "Lab Desk",
            "actor_role": "laboratory_team",
        },
    )
    assert data["ok"] is False
    assert "not authorized" in data["message"]


@pytest.mark.anyio
async def test_acknowledgement_advances_state_and_is_idempotent(server):
    args = {
        "case_id": "CC-001",
        "actor_id": "Dr. Khan",
        "actor_role": "responsible_clinician",
        "event_id": "EV-FIXED-1",
        "evidence": ["Reviewed"],
    }
    first = await _call(server, "acknowledge_result", args)
    assert first["ok"] is True
    assert first["state_id"] == "result_reviewed"

    second = await _call(server, "acknowledge_result", args)
    assert second["duplicate"] is True

    status = await _call(server, "get_case_status", {"case_id": "CC-001"})
    assert len(status["history"]) == 1


@pytest.mark.anyio
async def test_state_persists_across_a_fresh_connection(server):
    data = await _call(server, "get_stuck_workflows", {})
    # CC-001 was acknowledged in the previous test, so nothing is stuck now.
    assert data["total_stuck"] == 0
    assert "Nothing is stuck" in data["headline"]


@pytest.mark.anyio
async def test_send_chase_drafts_then_delivers(server):
    preview = await _call(server, "send_chase", {"case_id": "CC-002"})
    assert preview["ok"] is True
    assert preview["sent"] is False          # dry run by default
    assert "CC-002" in preview["text"]

    sent = await _call(server, "send_chase", {"case_id": "CC-002", "confirm": True})
    assert sent["sent"] is True
    assert sent["to"]


@pytest.mark.anyio
async def test_chasing_a_closed_case_is_refused(server):
    await _call(server, "acknowledge_result", {
        "case_id": "CC-003", "actor_id": "Care Team",
        "actor_role": "responsible_care_team", "event_type": "close",
        "event_id": "EV-CLOSE-3", "reason": "Follow-up not clinically required",
        "evidence": ["Clinician documented no follow-up required"],
    })
    result = await _call(server, "send_chase", {"case_id": "CC-003", "confirm": True})
    assert result["ok"] is False
    assert "closed" in result["message"].lower()


@pytest.mark.anyio
async def test_escalation_is_remembered_for_the_next_conversation(server):
    # Nothing outstanding to begin with for this case.
    await _call(server, "escalate", {"case_id": "CC-002", "confirm": True})
    briefing = await _call(server, "get_followups", {})
    ids = [row["case_id"] for row in briefing["cases"]]
    assert "CC-002" in ids
    assert "escalated" in briefing["headline"] or "chased" in briefing["headline"]


@pytest.mark.anyio
async def test_advancing_the_clock_changes_what_is_stuck(server):
    before = await _call(server, "get_stuck_workflows", {})
    moved = await _call(server, "advance_demo_clock", {"hours": 48})
    assert moved["ok"] is True
    after = await _call(server, "get_stuck_workflows", {})
    # Two days later, cases that were on time have breached their SLA.
    assert after["total_stuck"] > before["total_stuck"]

    reset = await _call(server, "reset_demo_clock", {})
    assert reset["ok"] is True
    back = await _call(server, "get_stuck_workflows", {})
    assert back["total_stuck"] == before["total_stuck"]


@pytest.mark.anyio
async def test_clock_will_not_rewind(server):
    result = await _call(server, "advance_demo_clock", {"hours": -5})
    assert result["ok"] is False
    assert "forward" in result["message"]


@pytest.mark.anyio
async def test_create_case_lands_on_the_board_already_overdue(server):
    created = await _call(server, "create_case", {
        "test_type": "Chest imaging", "current_state": "result_produced",
        "hours_waiting": 72, "case_id": "JUDGE-1",
    })
    assert created["ok"] is True
    assert created["status"] == "STUCK"
    assert created["hours_in_state"] == 72.0

    board = await _call(server, "get_stuck_workflows", {"limit": 50})
    assert "JUDGE-1" in [c["case_id"] for c in board["cases"]]


@pytest.mark.anyio
async def test_create_case_rejects_an_unknown_stage(server):
    result = await _call(server, "create_case", {
        "test_type": "X", "current_state": "nowhere",
    })
    assert result["ok"] is False
    assert "not a stage" in result["message"]


@pytest.mark.anyio
async def test_timeline_merges_transitions_and_followups(server):
    await _call(server, "create_case", {
        "test_type": "Biopsy", "current_state": "result_produced",
        "hours_waiting": 90, "case_id": "JUDGE-2",
    })
    await _call(server, "escalate", {"case_id": "JUDGE-2", "confirm": True})

    timeline = await _call(server, "get_case_timeline", {"case_id": "JUDGE-2"})
    kinds = [event["kind"] for event in timeline["events"]]
    assert "escalation" in kinds
    assert kinds[-1] == "current"          # the now marker always closes it


@pytest.mark.anyio
async def test_timeline_of_an_unknown_case_is_handled(server):
    result = await _call(server, "get_case_timeline", {"case_id": "NOPE"})
    assert result["found"] is False


# ----------------------------------------------------- adversarial: create

@pytest.mark.anyio
async def test_create_case_rejects_negative_hours(server):
    result = await _call(server, "create_case", {
        "test_type": "X", "hours_waiting": -5, "case_id": "NEG-1"})
    assert result["ok"] is False
    assert "negative" in result["message"]


@pytest.mark.anyio
async def test_create_case_rejects_a_duplicate_id(server):
    first = await _call(server, "create_case", {"test_type": "A", "case_id": "DUP-1"})
    assert first["ok"] is True
    second = await _call(server, "create_case", {"test_type": "A", "case_id": "DUP-1"})
    assert second["ok"] is False
    assert "already exists" in second["message"]


@pytest.mark.anyio
async def test_create_case_rejects_a_terminal_stage(server):
    # A case cannot be born closed: closed is terminal and carries no timer.
    result = await _call(server, "create_case", {
        "test_type": "A", "current_state": "closed", "case_id": "TERM-1"})
    assert result["ok"] is False


@pytest.mark.anyio
async def test_create_case_rejects_an_unknown_role(server):
    result = await _call(server, "create_case", {
        "test_type": "A", "owner_role": "chief_wizard", "case_id": "ROLE-1"})
    assert result["ok"] is False
    assert "not a role" in result["message"]


@pytest.mark.anyio
async def test_created_case_ids_do_not_collide_with_the_ward(server):
    a = await _call(server, "create_case", {"test_type": "A"})
    b = await _call(server, "create_case", {"test_type": "B"})
    assert a["case_id"] != b["case_id"]


# ------------------------------------------------------ board coherence

@pytest.mark.anyio
async def test_board_snapshot_is_internally_consistent(server):
    board = await _call(server, "get_board_snapshot", {})
    assert board["counts"]["stuck"] >= len(board["cases"]) or board["counts"]["stuck"] == len(board["cases"])
    assert board["counts"]["critical_stuck"] == sum(
        1 for c in board["cases"] if str(c["urgency"]).lower() == "critical")
    assert board["generated_at"] == board["clock"]["now"]


@pytest.mark.anyio
async def test_board_excludes_closed_cases(server):
    await _call(server, "create_case", {
        "test_type": "Closing", "current_state": "result_reviewed",
        "hours_waiting": 50, "owner_role": "responsible_care_team", "case_id": "CLOSE-1"})
    before = await _call(server, "get_board_snapshot", {})
    assert "CLOSE-1" in [c["case_id"] for c in before["cases"]]

    await _call(server, "acknowledge_result", {
        "case_id": "CLOSE-1", "actor_id": "Care Team",
        "actor_role": "responsible_care_team", "event_type": "close",
        "event_id": "EV-CLOSE-1", "reason": "Not clinically required",
        "evidence": ["documented"]})

    after = await _call(server, "get_board_snapshot", {})
    assert "CLOSE-1" not in [c["case_id"] for c in after["cases"]]


# ------------------------------------------------- the whole loop, once

@pytest.mark.anyio
async def test_full_lifecycle_create_advance_escalate_review_close(server):
    """The demo, as a test: create, let time pass, chase, act, finish."""
    created = await _call(server, "create_case", {
        "test_type": "Histopathology", "current_state": "result_produced",
        "hours_waiting": 20, "case_id": "LIFE-1"})
    assert created["status"] == "ON_TIME"

    await _call(server, "advance_demo_clock", {"hours": 24})
    status = await _call(server, "get_case_status", {"case_id": "LIFE-1"})
    assert status["status"] == "STUCK"          # time alone made it overdue

    preview = await _call(server, "escalate", {"case_id": "LIFE-1"})
    assert preview["recorded"] is False
    confirmed = await _call(server, "escalate", {"case_id": "LIFE-1", "confirm": True})
    assert confirmed["recorded"] is True

    briefing = await _call(server, "get_followups", {})
    assert "LIFE-1" in [row["case_id"] for row in briefing["cases"]]

    await _call(server, "acknowledge_result", {
        "case_id": "LIFE-1", "actor_id": "Dr. Khan",
        "actor_role": "responsible_clinician", "event_id": "EV-LIFE-1"})

    # Acting on it clears the follow-up without anyone telling CareChaser.
    after = await _call(server, "get_followups", {})
    assert "LIFE-1" not in [row["case_id"] for row in after["cases"]]

    timeline = await _call(server, "get_case_timeline", {"case_id": "LIFE-1"})
    kinds = [event["kind"] for event in timeline["events"]]
    assert "escalation" in kinds and "transition" in kinds
    assert kinds[-1] == "current"

    await _call(server, "reset_demo_clock", {})


# ------------------------------------------------------------- report intake

REPORT = """Patient MK-9001
CBC - completed
Serum potassium 6.8 mmol/L - result available
Chest X-Ray - not completed
"""


@pytest.mark.anyio
async def test_reading_a_report_creates_nothing(server):
    before = await _call(server, "get_board_snapshot", {})
    result = await _call(server, "read_report", {"text": REPORT})
    assert result["ok"] is True
    assert result["total"] == 3
    after = await _call(server, "get_board_snapshot", {})
    assert after["counts"]["stuck"] == before["counts"]["stuck"]


@pytest.mark.anyio
async def test_result_values_never_reach_the_fields_that_become_cases(server):
    """Values appear only under `removed`, for the reviewer to see. They are
    absent from every field that would become part of a case."""
    result = await _call(server, "read_report", {"text": REPORT})
    for row in result["rows"]:
        assert "6.8" not in row["test_name"]
        assert "6.8" not in row["status_text"]
    potassium = next(
        (r for r in result["rows"] if "potassium" in r["test_name"].lower()), None)
    assert potassium is not None, f"no potassium row in {[r['test_name'] for r in result['rows']]}"
    assert "6.8" in potassium["removed"]


@pytest.mark.anyio
async def test_a_created_case_carries_no_result_value_anywhere(server):
    """The permanent record is the real test: once a case exists, nothing
    in it or its audit trail contains the value from the report."""
    read = await _call(server, "read_report", {"text": REPORT})
    rows = [row for row in read["rows"] if row["ok"]]
    confirmed = await _call(server, "confirm_report_rows",
                            {"rows": rows, "patient": read["patient"]})
    for created in confirmed["created"]:
        timeline = await _call(server, "get_case_timeline", {"case_id": created["case_id"]})
        blob = json.dumps(timeline)
        assert "6.8" not in blob
        assert "mmol" not in blob.lower()


@pytest.mark.anyio
async def test_a_case_remembers_which_report_it_came_from(server):
    read = await _call(server, "read_report", {"text": REPORT})
    rows = [row for row in read["rows"] if row["ok"]][:1]
    confirmed = await _call(server, "confirm_report_rows",
                            {"rows": rows, "patient": read["patient"]})
    case_id = confirmed["created"][0]["case_id"]
    timeline = await _call(server, "get_case_timeline", {"case_id": case_id})
    first = timeline["events"][0]
    assert first["kind"] == "intake"
    assert "from a report" in first["title"]


@pytest.mark.anyio
async def test_provenance_does_not_show_up_as_outstanding_work(server):
    """Where a case came from is history, not something to chase."""
    read = await _call(server, "read_report", {"text": REPORT})
    rows = [row for row in read["rows"] if row["ok"]][:1]
    confirmed = await _call(server, "confirm_report_rows",
                            {"rows": rows, "patient": read["patient"]})
    case_id = confirmed["created"][0]["case_id"]
    briefing = await _call(server, "get_followups", {})
    assert case_id not in [row["case_id"] for row in briefing["cases"]]


@pytest.mark.anyio
async def test_confirming_rows_creates_the_cases(server):
    read = await _call(server, "read_report", {"text": REPORT})
    rows = [row for row in read["rows"] if row["ok"]]
    for row in rows:
        row["hours_waiting"] = 40

    confirmed = await _call(server, "confirm_report_rows",
                            {"rows": rows, "patient": read["patient"]})
    assert confirmed["ok"] is True
    assert len(confirmed["created"]) == len(rows)
    assert all(c["status"] == "STUCK" for c in confirmed["created"])

    board = await _call(server, "get_board_snapshot", {})
    created_ids = {c["case_id"] for c in confirmed["created"]}
    assert created_ids <= {c["case_id"] for c in board["cases"]}


@pytest.mark.anyio
async def test_flagged_rows_are_skipped_not_guessed(server):
    confirmed = await _call(server, "confirm_report_rows", {
        "rows": [{"test_name": "Mystery", "state": None, "owner_role": None}]})
    assert confirmed["ok"] is False
    assert confirmed["skipped"][0]["why"]
