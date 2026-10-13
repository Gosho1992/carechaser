"""The two-day demo: CareChaser remembers what it chased.

Day one, a clinician escalates a stuck biopsy. Nobody acts. Day two, in a new
conversation with a new agent session, CareChaser opens by saying so.

    python demo_continuity.py

Starts both servers on free ports, runs day one at the demo clock, restarts
with the clock moved on 24 hours, and runs day two. Nothing is hard-coded:
day two reads the real follow-up log written on day one.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
DAY_ONE = datetime.fromisoformat("2026-09-28T08:00:00+05:00")
DAY_TWO = DAY_ONE + timedelta(days=1)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait(port: int, proc: subprocess.Popen) -> None:
    for _ in range(150):
        if proc.poll() is not None:
            raise RuntimeError("server exited early")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"nothing listening on {port}")


class Stack:
    """MCP server plus web client at one point in time."""

    def __init__(self, workdir: Path, when: datetime, log: Path):
        self.mcp_port = free_port()
        self.web_port = free_port()
        env = {
            **os.environ,
            "CARECHASER_CASES": str(workdir / "cases.json"),
            "CARECHASER_WORKFLOW": str(ROOT / "workflow.yaml"),
            "CARECHASER_CHASE_LOG": str(log),
            "CARECHASER_SIMULATED_NOW": when.isoformat(),
            "CARECHASER_FORCE_OFFLINE": "1",
            "PYTHONPATH": str(ROOT),
            "CARECHASER_PORT": str(self.mcp_port),
        }
        self.mcp = subprocess.Popen(
            [sys.executable, str(ROOT / "carechaser_mcp.py")],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        wait(self.mcp_port, self.mcp)

        web_env = {
            **env,
            "CARECHASER_MCP_URL": f"http://127.0.0.1:{self.mcp_port}/mcp",
            "CARECHASER_WEB_PORT": str(self.web_port),
        }
        self.web = subprocess.Popen(
            [sys.executable, str(ROOT / "carechaser_web.py")],
            env=web_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        wait(self.web_port, self.web)

    def ask(self, said: str, session: str) -> None:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.web_port}/api/ask",
            data=json.dumps(
                {
                    "said": said,
                    "session_id": session,
                    "speaker": "Dr. Khan",
                    "speaker_role": "responsible_clinician",
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        data = json.loads(urllib.request.urlopen(request).read())
        tools = ", ".join(call["tool"] for call in data["tool_calls"])
        print(f'  "{said}"')
        print(f"     [{tools}]")
        print(f"     {data['reply']}\n")

    def stop(self) -> None:
        for proc in (self.web, self.mcp):
            proc.terminate()
            proc.wait(timeout=10)


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="carechaser-demo-"))
    log = workdir / "chase_log.jsonl"
    subprocess.run(
        [sys.executable, str(ROOT / "seed_demo.py"), "--out", str(workdir / "cases.json")],
        check=True, stdout=subprocess.DEVNULL,
    )

    print(f"\n=== DAY ONE — {DAY_ONE:%A %d %B, %H:%M} ===\n")
    day_one = Stack(workdir, DAY_ONE, log)
    try:
        day_one.ask("what is stuck today", "monday")
        day_one.ask("what is the status of CC 001", "monday")
        day_one.ask("escalate CC 001", "monday")
        day_one.ask("yes escalate CC 001", "monday")
    finally:
        day_one.stop()

    print("   ... overnight, nobody acts on CC-001 ...\n")

    print(f"=== DAY TWO — {DAY_TWO:%A %d %B, %H:%M} — new conversation ===\n")
    day_two = Stack(workdir, DAY_TWO, log)
    try:
        day_two.ask("anything I should know", "tuesday")
        day_two.ask("chase CC 001", "tuesday")
        day_two.ask("yes send it CC 001", "tuesday")
        day_two.ask("mark CC 001 reviewed", "tuesday")
        day_two.ask("anything I should know", "tuesday")
    finally:
        day_two.stop()

    print("=== follow-up log written across both days ===\n")
    for line in log.read_text(encoding="utf-8").strip().splitlines():
        row = json.loads(line)
        print(f"  {row['at'][:16]}  {row['kind']:10} {row['case_id']} -> {row['to']}")
    print()


if __name__ == "__main__":
    main()
