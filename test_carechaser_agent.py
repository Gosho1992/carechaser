"""Tests for the agent and web layer.

These run against a real MCP server subprocess with a frozen clock and the
offline brain, so they assert routing and wording, not a model's output.
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
from fastapi.testclient import TestClient

from carechaser_agent import CareChaserAgent, OfflineBrain, call_tool, list_tool_schemas

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
def mcp_url(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("agent")
    (workdir / "cases.json").write_text(json.dumps(FIXTURE_CASES), encoding="utf-8")
    port = _free_port()
    env = {
        **os.environ,
        "CARECHASER_CASES": str(workdir / "cases.json"),
        "CARECHASER_WORKFLOW": str(ROOT / "workflow.yaml"),
        "CARECHASER_SIMULATED_NOW": FROZEN_NOW,
        "CARECHASER_PORT": str(port),
        "PYTHONPATH": str(ROOT),
        # Same reason as the MCP fixture: ambient AWS credentials must not
        # change which extractor or drafter the tests exercise.
        "CARECHASER_FORCE_OFFLINE": "1",
    }
    # See the note in the other fixture: an unread stdout pipe deadlocks the
    # server once its buffer fills.
    log_path = workdir / "server.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "carechaser_mcp.py")],
        env=env, stdout=log_file, stderr=subprocess.STDOUT,
    )
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
    yield f"http://127.0.0.1:{port}/mcp"
    proc.terminate()
    proc.wait(timeout=10)
    log_file.close()


# ------------------------------------------------------------------- routing


@pytest.mark.parametrize(
    "said,expected",
    [
        ("what's stuck today", "get_stuck_workflows"),
        ("anything overdue", "get_stuck_workflows"),
        ("what's waiting on me", "get_my_pending"),
        ("what does Dr Ahmed still owe", "get_my_pending"),
        ("what's the status of CC 001", "get_case_status"),
        ("escalate CC 001", "escalate"),
        ("mark CC 001 reviewed", "acknowledge_result"),
        ("chase CC 001", "send_chase"),
        ("send CC 001 a reminder", "send_chase"),
    ],
)
def test_offline_brain_routes_speech_to_the_right_tool(said, expected):
    plan = OfflineBrain().plan(said, {"speaker": "Dr. Khan"})
    assert plan[0][0] == expected


def test_spoken_case_ids_are_normalised():
    brain = OfflineBrain()
    for spoken in ["status of CC 001", "status of cc-001", "status of CC001"]:
        assert brain.plan(spoken, {})[0][1]["case_id"] == "CC-001"


def test_chase_defaults_to_draft_and_confirms_on_yes():
    assert OfflineBrain().plan("chase CC 001", {})[0][1]["confirm"] is False
    assert OfflineBrain().plan("yes send it CC 001", {})[0][1]["confirm"] is True


def test_escalation_defaults_to_preview_not_action():
    args = OfflineBrain().plan("escalate CC 001", {})[0][1]
    assert args["confirm"] is False
    confirmed = OfflineBrain().plan("yes escalate CC 001", {})[0][1]
    assert confirmed["confirm"] is True


# ------------------------------------------------------------- against a server


@pytest.mark.anyio
async def test_tool_schemas_are_fetched_from_the_server(mcp_url):
    schemas = await list_tool_schemas(mcp_url)
    names = {s["toolSpec"]["name"] for s in schemas}
    assert "get_stuck_workflows" in names
    assert all(s["toolSpec"]["description"] for s in schemas)


@pytest.mark.anyio
async def test_agent_answers_what_is_stuck_with_engine_numbers(mcp_url):
    agent = CareChaserAgent(mcp_url=mcp_url)
    turn = await agent.ask("what is stuck today")
    assert turn.tool_calls[0]["tool"] == "get_stuck_workflows"
    assert "31 hours" in turn.reply
    assert "Dr. Khan" in turn.reply


@pytest.mark.anyio
async def test_agent_keeps_a_thread_across_turns(mcp_url):
    agent = CareChaserAgent(mcp_url=mcp_url)
    await agent.ask("what is stuck today")
    second = await agent.ask("what is the status of CC 001")
    assert second.tool_calls[0]["arguments"]["case_id"] == "CC-001"
    assert len(agent.history) == 4


@pytest.mark.anyio
async def test_refusal_is_spoken_not_swallowed(mcp_url):
    result = await call_tool(
        "acknowledge_result",
        {"case_id": "CC-001", "actor_id": "Lab", "actor_role": "laboratory_team"},
        mcp_url,
    )
    assert result["ok"] is False
    reply = OfflineBrain().speak("mark it reviewed", [{"result": result}])
    assert "not authorized" in reply


# ------------------------------------------------------------------ web layer


@pytest.fixture
def web(mcp_url, monkeypatch):
    monkeypatch.setenv("CARECHASER_MCP_URL", mcp_url)
    monkeypatch.setenv("CARECHASER_FORCE_OFFLINE", "1")
    import importlib

    import carechaser_web

    importlib.reload(carechaser_web)
    return TestClient(carechaser_web.app)


def test_health_reports_the_brain_in_use(web):
    body = web.get("/api/health").json()
    assert body["ok"] is True
    assert body["brain"] == "offline"


def test_ask_endpoint_returns_reply_and_tool_trace(web):
    body = web.post("/api/ask", json={"said": "what is stuck today"}).json()
    assert body["ok"] is True
    assert body["tool_calls"][0]["tool"] == "get_stuck_workflows"
    assert "stuck" in body["reply"].lower()


def test_page_is_served(web):
    page = web.get("/")
    assert page.status_code == 200
    assert "CareChaser" in page.text
