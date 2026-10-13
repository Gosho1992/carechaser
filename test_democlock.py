"""Tests for the runtime demo clock."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from democlock import DemoClock, describe

BASE = "2026-09-28T08:00:00+05:00"


@pytest.fixture
def clock(tmp_path, monkeypatch):
    monkeypatch.setenv("CARECHASER_SIMULATED_NOW", BASE)
    return DemoClock(tmp_path / "demo_clock.json")


def test_starts_at_the_base_time(clock):
    assert clock.now() == datetime.fromisoformat(BASE)
    assert clock.offset_hours() == 0.0


def test_advance_moves_time_forward(clock):
    clock.advance(24)
    assert clock.now() == datetime.fromisoformat("2026-09-29T08:00:00+05:00")


def test_advances_accumulate(clock):
    clock.advance(24)
    clock.advance(3)
    assert clock.offset_hours() == 27.0
    assert clock.now() == datetime.fromisoformat("2026-09-29T11:00:00+05:00")


def test_fractional_hours_work(clock):
    clock.advance(0.5)
    assert clock.now() == datetime.fromisoformat("2026-09-28T08:30:00+05:00")


def test_clock_refuses_to_go_backwards(clock):
    # Rewinding would put recorded follow-ups in the future, which the
    # engine treats as impossible.
    with pytest.raises(ValueError, match="only moves forward"):
        clock.advance(-5)
    with pytest.raises(ValueError):
        clock.advance(0)


def test_reset_returns_to_the_base(clock):
    clock.advance(72)
    clock.reset()
    assert clock.now() == datetime.fromisoformat(BASE)
    assert clock.offset_hours() == 0.0


def test_offset_is_shared_through_the_file(tmp_path, monkeypatch):
    """Two processes, one clock: the browser moves it, the server sees it."""
    monkeypatch.setenv("CARECHASER_SIMULATED_NOW", BASE)
    path = tmp_path / "demo_clock.json"
    writer = DemoClock(path)
    reader = DemoClock(path)
    writer.advance(12)
    assert reader.offset_hours() == 12.0
    assert reader.now() == datetime.fromisoformat("2026-09-28T20:00:00+05:00")


def test_real_clock_when_no_base_is_set(tmp_path, monkeypatch):
    monkeypatch.delenv("CARECHASER_SIMULATED_NOW", raising=False)
    clock = DemoClock(tmp_path / "c.json")
    assert clock.state().base is None
    assert clock.state().simulated is False
    before = datetime.now(timezone.utc)
    assert (clock.now() - before).total_seconds() < 5


def test_offset_applies_to_the_real_clock_too(tmp_path, monkeypatch):
    monkeypatch.delenv("CARECHASER_SIMULATED_NOW", raising=False)
    clock = DemoClock(tmp_path / "c.json")
    clock.advance(24)
    gap = (clock.now() - datetime.now(timezone.utc)).total_seconds() / 3600
    assert 23.9 < gap < 24.1
    assert clock.state().simulated is True


def test_corrupt_clock_file_falls_back_to_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("CARECHASER_SIMULATED_NOW", BASE)
    path = tmp_path / "c.json"
    path.write_text("not json", encoding="utf-8")
    clock = DemoClock(path)
    assert clock.offset_hours() == 0.0
    assert clock.now() == datetime.fromisoformat(BASE)


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path, monkeypatch):
    monkeypatch.setenv("CARECHASER_SIMULATED_NOW", BASE)
    path = tmp_path / "c.json"
    clock = DemoClock(path)
    clock.advance(1)
    assert json.loads(path.read_text(encoding="utf-8"))["offset_hours"] == 1.0
    assert [p.name for p in tmp_path.iterdir()] == ["c.json"]


def test_description_is_speakable(clock):
    assert "Monday 28 September" in describe(clock.state())
    clock.advance(24)
    assert "moved forward 1 day" in describe(clock.state())
    clock.advance(3)
    assert "1 day and 3 hours" in describe(clock.state())
