"""CareChaser web client.

Serves the simulated Alexa+ experience: a browser page that listens, sends
the transcript here, runs the agent, and speaks the answer back. The page
also shows every tool call, so a viewer can see that the numbers come from
the engine rather than from a model.

Run (with the MCP server already running on :8000):
    python carechaser_web.py
    open http://127.0.0.1:8080
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from carechaser_agent import DEFAULT_MCP_URL, CareChaserAgent, build_brain, call_tool

ROOT = Path(__file__).parent
STATIC = ROOT / "static"

app = FastAPI(title="CareChaser voice client")

MCP_URL = os.environ.get("CARECHASER_MCP_URL", DEFAULT_MCP_URL)
BRAIN = build_brain()

# One agent per browser session id, so a conversation keeps its thread across
# turns. This is what makes "escalate it" work after "what's stuck today".
_agents: dict[str, CareChaserAgent] = {}


def agent_for(session_id: str) -> CareChaserAgent:
    if session_id not in _agents:
        _agents[session_id] = CareChaserAgent(mcp_url=MCP_URL, brain=BRAIN)
    return _agents[session_id]


class Ask(BaseModel):
    said: str
    session_id: str = "default"
    speaker: str | None = None
    speaker_role: str | None = None


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "brain": BRAIN.name, "mcp_url": MCP_URL}


@app.post("/api/ask")
async def ask(body: Ask) -> JSONResponse:
    context = {
        k: v
        for k, v in {"speaker": body.speaker, "speaker_role": body.speaker_role}.items()
        if v
    }
    try:
        turn = await agent_for(body.session_id).ask(body.said, context)
    except Exception as exc:  # surfaced in the UI rather than a blank page
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "reply": "I could not reach the CareChaser server.",
                "error": str(exc),
                "tool_calls": [],
            },
        )
    return JSONResponse(
        content={
            "ok": True,
            "reply": turn.reply,
            "tool_calls": turn.tool_calls,
            "brain": turn.brain,
        }
    )


@app.post("/api/reset")
async def reset(body: Ask) -> dict[str, Any]:
    _agents.pop(body.session_id, None)
    return {"ok": True}


class ReportText(BaseModel):
    text: str


class ConfirmRows(BaseModel):
    rows: list[dict[str, Any]]
    patient: str | None = None


class NewCase(BaseModel):
    test_type: str
    current_state: str = "result_produced"
    hours_waiting: float = 0.0
    owner: str | None = None
    owner_role: str | None = None
    patient: str | None = None
    urgency: str = "routine"


class ClockMove(BaseModel):
    hours: float = 24.0


@app.get("/api/clock")
async def clock() -> dict[str, Any]:
    """Read the demo clock through the MCP server, so the browser and the
    engine can never disagree about what time it is."""
    return await call_tool("get_demo_clock", {}, MCP_URL)


@app.post("/api/clock/advance")
async def clock_advance(body: ClockMove) -> dict[str, Any]:
    return await call_tool("advance_demo_clock", {"hours": body.hours}, MCP_URL)


@app.post("/api/clock/reset")
async def clock_reset() -> dict[str, Any]:
    return await call_tool("reset_demo_clock", {}, MCP_URL)


@app.post("/api/intake/read")
async def intake_read(body: ReportText) -> dict[str, Any]:
    """Read a report into reviewable rows. Creates nothing."""
    return await call_tool("read_report", {"text": body.text}, MCP_URL)


@app.post("/api/intake/confirm")
async def intake_confirm(body: ConfirmRows) -> dict[str, Any]:
    """Create cases from rows a human has reviewed."""
    return await call_tool(
        "confirm_report_rows",
        {"rows": body.rows, "patient": body.patient},
        MCP_URL,
    )


@app.get("/api/board")
async def board() -> dict[str, Any]:
    """The whole command centre, from one snapshot taken at one instant."""
    return await call_tool("get_board_snapshot", {"limit": 50}, MCP_URL)


@app.get("/api/case/{case_id}")
async def case_detail(case_id: str) -> dict[str, Any]:
    return await call_tool("get_case_timeline", {"case_id": case_id}, MCP_URL)


@app.post("/api/case")
async def case_create(body: NewCase) -> dict[str, Any]:
    return await call_tool("create_case", body.model_dump(exclude_none=True), MCP_URL)


@app.post("/api/case/{case_id}/escalate")
async def case_escalate(case_id: str, confirm: bool = False) -> dict[str, Any]:
    return await call_tool("escalate", {"case_id": case_id, "confirm": confirm}, MCP_URL)


@app.get("/api/workflow")
async def workflow_shape() -> dict[str, Any]:
    """Stages and roles, so the create form offers real options rather than
    hard-coded strings that drift from workflow.yaml."""
    from carechaser_engine import load_workflow

    config = load_workflow(os.environ.get("CARECHASER_WORKFLOW", "workflow.yaml"))
    return {
        "states": [
            {"id": s["state"], "name": s.get("display_name", s["state"]),
             "owner_role": s.get("owner_role"), "sla_hours": s.get("sla_hours")}
            for s in config["states"] if not s.get("terminal")
        ],
        "roles": [
            {"id": rid, "name": (meta or {}).get("display_name", rid),
             "person": (config.get("roster") or {}).get(rid)}
            for rid, meta in (config.get("roles") or {}).items()
        ],
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("CARECHASER_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("CARECHASER_WEB_PORT", "8080"))
    print(f"CareChaser voice client on http://{host}:{port}")
    print(f"  brain: {BRAIN.name}")
    print(f"  mcp:   {MCP_URL}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
