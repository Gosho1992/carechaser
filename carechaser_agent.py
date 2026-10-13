"""CareChaser agent layer.

Sits between the voice interface and the MCP server:

    speech -> agent -> (Bedrock chooses a tool) -> MCP server -> engine
                    <- spoken answer + tool trace <-

Two brains are available:

- BedrockBrain: Amazon Bedrock Converse with tool use. This is the real path
  and the one that satisfies the AWS Builder mini challenge.
- OfflineBrain: a deterministic intent router used when no AWS credentials
  are present, and in tests. It picks tools by keyword. It is not a language
  model and is never presented as one.

Either way the tools themselves are the MCP server's, so every number spoken
comes from the deterministic engine, not from a model.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from mcp import Client

DEFAULT_MCP_URL = os.environ.get("CARECHASER_MCP_URL", "http://127.0.0.1:8000/mcp")
DEFAULT_MODEL_ID = os.environ.get(
    "CARECHASER_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")

SYSTEM_PROMPT = """You are CareChaser, a voice assistant for hospital staff.

You track diagnostic test results through a clinical workflow and report which
cases are stuck, who owns them, and how overdue they are.

Rules you must follow:
- Every fact you state must come from a tool result. Never estimate hours,
  invent a case id, or guess who owns something.
- You are a workflow tool. You never interpret a laboratory result, never
  diagnose, and never suggest treatment. If asked anything clinical, say that
  CareChaser only tracks workflow status and the responsible clinician must be
  consulted.
- Before escalating, call escalate with confirm false and tell the user who
  would be contacted. Only call it again with confirm true if they agree.
- You are speaking aloud. Keep answers to one or two short sentences. Say
  "thirty one hours", not "31.0". Lead with the single most urgent thing.
- If a tool returns ok false, say plainly what was refused and why."""


# --------------------------------------------------------------------- tools


async def list_tool_schemas(mcp_url: str = DEFAULT_MCP_URL) -> list[dict[str, Any]]:
    """Fetch tool definitions from the MCP server, in Bedrock toolSpec shape."""
    async with Client(mcp_url) as client:
        listed = await client.list_tools()
        return [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": {"json": tool.input_schema},
                }
            }
            for tool in listed.tools
        ]


async def call_tool(
    name: str, arguments: dict[str, Any], mcp_url: str = DEFAULT_MCP_URL
) -> dict[str, Any]:
    """Call one MCP tool and return its parsed result."""
    async with Client(mcp_url) as client:
        result = await client.call_tool(name, arguments)
        if not result.content:
            return {}
        text = result.content[0].text
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return {"text": text}


# -------------------------------------------------------------------- brains


@dataclass
class Turn:
    """One exchange, including what the agent did to answer it."""

    said: str
    reply: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    brain: str = "offline"


class OfflineBrain:
    """Keyword intent router. Used without AWS credentials and in tests.

    Deterministic by design: the same sentence always picks the same tool, so
    tests do not depend on a model's mood.
    """

    name = "offline"

    def __init__(self, default_actor: str = "Dr. Khan", default_role: str = "responsible_clinician"):
        self.default_actor = default_actor
        self.default_role = default_role

    @staticmethod
    def _case_id(text: str) -> Optional[str]:
        # Spoken ids arrive as "C C 001", "cc 001", "case 001".
        match = re.search(r"\b([A-Za-z]{2,4})[\s\-]?(\d{2,4})\b", text)
        if match:
            return f"{match.group(1).upper().replace(' ', '')}-{match.group(2)}"
        return None

    @staticmethod
    def _person(text: str) -> Optional[str]:
        match = re.search(r"\b(?:dr\.?|doctor)\s+([A-Za-z]+)", text, re.I)
        if match:
            return f"Dr. {match.group(1).title()}"
        if re.search(r"\blab\b", text, re.I):
            return "laboratory_team"
        if re.search(r"\b(me|my|mine|i)\b", text, re.I):
            return None  # caller identity, filled in by the web layer
        return None

    def plan(self, said: str, context: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        text = said.lower().strip()
        case_id = self._case_id(said)
        who = self._person(said) or context.get("speaker")

        if re.search(r"\b(advance|skip ahead|move (?:the )?clock|fast forward|next day|tomorrow|a day later)\b", text):
            hours = 24.0
            number = re.search(r"\b(\d+(?:\.\d+)?)\s*(hour|day)", text)
            if number:
                hours = float(number.group(1)) * (24 if number.group(2) == "day" else 1)
            return [("advance_demo_clock", {"hours": hours})]

        if re.search(r"\b(reset (?:the )?clock|start the demo again|back to the beginning)\b", text):
            return [("reset_demo_clock", {})]

        if re.search(r"\b(what time|demo clock|where are we in the demo)\b", text):
            return [("get_demo_clock", {})]

        if re.search(r"\b(should know|anything outstanding|follow[- ]?ups?|yesterday|still waiting|what did we chase|briefing|catch me up)\b", text):
            return [("get_followups", {})]

        if re.search(r"\b(chase|remind|reminder|notify|send it|send the message|message (?:them|him|her))\b", text):
            if case_id:
                confirm = bool(re.search(r"\b(yes|confirm|do it|send it|go ahead)\b", text))
                return [("send_chase", {"case_id": case_id, "confirm": confirm})]

        if re.search(r"\b(escalate|supervisor|covering|chase up)\b", text):
            if case_id:
                confirm = bool(re.search(r"\b(yes|confirm|do it|go ahead)\b", text))
                return [("escalate", {"case_id": case_id, "confirm": confirm})]

        if re.search(r"\b(acknowledge|reviewed|i have seen|mark it|seen it|informed)\b", text):
            if case_id:
                event = (
                    "patient_informed"
                    if "informed" in text or "patient" in text
                    else "result_reviewed"
                )
                return [
                    (
                        "acknowledge_result",
                        {
                            "case_id": case_id,
                            "actor_id": context.get("speaker", self.default_actor),
                            "actor_role": context.get("speaker_role", self.default_role),
                            "event_type": event,
                        },
                    )
                ]

        if case_id and re.search(r"\b(status|where|what about|how long|history)\b", text):
            return [("get_case_status", {"case_id": case_id})]

        if re.search(r"\b(waiting on|pending|owe|assigned to|mine|my cases)\b", text):
            return [("get_my_pending", {"who": who or self.default_actor})]

        if case_id:
            return [("get_case_status", {"case_id": case_id})]

        return [("get_stuck_workflows", {})]

    def speak(self, said: str, results: list[dict[str, Any]]) -> str:
        """Build a spoken reply from tool output only."""
        if not results:
            return "I could not find anything for that."
        data = results[-1]["result"]
        if isinstance(data, dict):
            if data.get("message"):
                return str(data["message"])
            if data.get("headline"):
                headline = str(data["headline"])
                cases = data.get("cases") or []
                if len(cases) > 1:
                    second = cases[1]
                    headline += (
                        f" Next is {second['test_type']} for {second['patient']}, "
                        f"{second['hours_in_state']:.0f} hours with {second['owner']}."
                    )
                return headline
            if data.get("found") is True:
                return (
                    f"{data['case_id']} is in {data['state']}, "
                    f"{data['hours_in_state']:.0f} hours, owned by {data['owner']}. "
                    f"{'It is overdue.' if data['status'] == 'STUCK' else 'It is on time.'}"
                )
        return "I have the result but could not summarise it."


class BedrockBrain:
    """Amazon Bedrock Converse with tool use."""

    name = "bedrock"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, region: str = DEFAULT_REGION):
        import boto3  # imported here so the offline path needs no AWS SDK

        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def converse(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        return self.client.converse(
            modelId=self.model_id,
            messages=messages,
            system=[{"text": SYSTEM_PROMPT}],
            toolConfig={"tools": tools},
            inferenceConfig={"maxTokens": 512, "temperature": 0.0},
        )


# --------------------------------------------------------------------- agent


class CareChaserAgent:
    """Runs one turn: hear, choose tools, call MCP, answer."""

    def __init__(
        self,
        mcp_url: str = DEFAULT_MCP_URL,
        brain: Any | None = None,
        max_tool_rounds: int = 4,
    ):
        self.mcp_url = mcp_url
        self.brain = brain or OfflineBrain()
        self.max_tool_rounds = max_tool_rounds
        self.history: list[dict[str, Any]] = []

    async def ask(self, said: str, context: dict[str, Any] | None = None) -> Turn:
        context = context or {}
        if isinstance(self.brain, BedrockBrain):
            return await self._ask_bedrock(said, context)
        return await self._ask_offline(said, context)

    # -- offline ---------------------------------------------------------

    async def _ask_offline(self, said: str, context: dict[str, Any]) -> Turn:
        plan = self.brain.plan(said, context)
        calls = []
        for name, arguments in plan[: self.max_tool_rounds]:
            result = await call_tool(name, arguments, self.mcp_url)
            calls.append({"tool": name, "arguments": arguments, "result": result})
        reply = self.brain.speak(said, calls)
        self.history.append({"role": "user", "content": [{"text": said}]})
        self.history.append({"role": "assistant", "content": [{"text": reply}]})
        return Turn(said=said, reply=reply, tool_calls=calls, brain=self.brain.name)

    # -- bedrock ---------------------------------------------------------

    async def _ask_bedrock(self, said: str, context: dict[str, Any]) -> Turn:
        tools = await list_tool_schemas(self.mcp_url)
        speaker = context.get("speaker")
        opening = said if not speaker else f"[speaker: {speaker}] {said}"
        self.history.append({"role": "user", "content": [{"text": opening}]})

        calls: list[dict[str, Any]] = []
        for _ in range(self.max_tool_rounds):
            response = self.brain.converse(self.history, tools)
            message = response["output"]["message"]
            self.history.append(message)

            uses = [b["toolUse"] for b in message["content"] if "toolUse" in b]
            if not uses:
                text = " ".join(b["text"] for b in message["content"] if "text" in b)
                return Turn(said=said, reply=text.strip(), tool_calls=calls, brain="bedrock")

            results_block = []
            for use in uses:
                result = await call_tool(use["name"], use.get("input", {}), self.mcp_url)
                calls.append(
                    {"tool": use["name"], "arguments": use.get("input", {}), "result": result}
                )
                results_block.append(
                    {
                        "toolResult": {
                            "toolUseId": use["toolUseId"],
                            "content": [{"json": result}],
                        }
                    }
                )
            self.history.append({"role": "user", "content": results_block})

        return Turn(
            said=said,
            reply="I needed too many steps for that. Could you narrow it down?",
            tool_calls=calls,
            brain="bedrock",
        )


def build_brain() -> Any:
    """Bedrock when credentials exist, offline router otherwise."""
    if os.environ.get("CARECHASER_FORCE_OFFLINE"):
        return OfflineBrain()
    try:
        import boto3

        session = boto3.Session()
        if session.get_credentials() is None:
            return OfflineBrain()
        return BedrockBrain()
    except Exception:
        return OfflineBrain()
