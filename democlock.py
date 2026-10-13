"""CareChaser demo clock.

Lets a judge move time forward instead of waiting a day to see an SLA breach.

    now = base + offset

base    is CARECHASER_SIMULATED_NOW if set, otherwise the real clock.
offset  is a number of hours stored in a small file, so the MCP server, the
        web client and any script all read the same value without restarting.

Why a file rather than a variable: the servers are separate processes. A
judge pressing "+24 hours" in the browser has to move the clock the MCP
server reads, and the two do not share memory.

Real time still advances underneath. With a one-day offset, an hour of real
waiting is an hour of demo time on top of that day, which is what you want:
the demo moves forward, it does not freeze.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_OFFSET_PATH = os.environ.get("CARECHASER_CLOCK_FILE", "demo_clock.json")


def _parse(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("simulated time needs a timezone offset")
    return parsed


@dataclass
class ClockState:
    base: Optional[datetime]
    offset_hours: float
    now: datetime

    @property
    def simulated(self) -> bool:
        return self.base is not None or self.offset_hours != 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "now": self.now.isoformat(),
            "base": self.base.isoformat() if self.base else None,
            "offset_hours": self.offset_hours,
            "simulated": self.simulated,
        }


class DemoClock:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or DEFAULT_OFFSET_PATH)

    # ------------------------------------------------------------ reading

    def base(self) -> Optional[datetime]:
        raw = os.environ.get("CARECHASER_SIMULATED_NOW")
        return _parse(raw) if raw else None

    def offset_hours(self) -> float:
        """Read from disk every time: another process may have changed it."""
        if not self.path.exists():
            return 0.0
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                return float(json.load(handle).get("offset_hours", 0.0))
        except (json.JSONDecodeError, ValueError, OSError):
            return 0.0

    def now(self) -> datetime:
        base = self.base() or datetime.now(timezone.utc)
        return base + timedelta(hours=self.offset_hours())

    def state(self) -> ClockState:
        return ClockState(base=self.base(), offset_hours=self.offset_hours(), now=self.now())

    # ------------------------------------------------------------ writing

    def _write(self, offset_hours: float) -> ClockState:
        payload = {
            "offset_hours": round(float(offset_hours), 4),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".clock-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp, self.path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return self.state()

    def advance(self, hours: float) -> ClockState:
        """Move time forward. Refuses to go backwards.

        Rewinding would put recorded follow-ups and state changes in the
        future, which the engine rejects as impossible. A judge who wants an
        earlier time resets instead.
        """
        if hours <= 0:
            raise ValueError("The demo clock only moves forward. Use reset to start over.")
        return self._write(self.offset_hours() + hours)

    def reset(self) -> ClockState:
        return self._write(0.0)


def describe(state: ClockState) -> str:
    """One speakable line about where the demo clock is."""
    stamp = state.now.strftime("%A %d %B, %H:%M")
    if not state.simulated:
        return f"Using the real clock: {stamp}."
    if state.offset_hours == 0:
        return f"Demo time is {stamp}."
    days, hours = divmod(state.offset_hours, 24)
    moved = []
    if days:
        moved.append(f"{int(days)} day{'s' if days != 1 else ''}")
    if hours:
        moved.append(f"{hours:g} hour{'s' if hours != 1 else ''}")
    return f"Demo time is {stamp}, moved forward {' and '.join(moved)}."
