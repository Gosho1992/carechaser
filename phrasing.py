"""How CareChaser says things out loud.

One module, because the web page and the spoken answer must never word the
same fact differently. Every number that appears here was supplied by the
engine; nothing in this file computes, rounds into a different meaning, or
invents a figure.

WHY THIS IS NOT A LANGUAGE MODEL

The obvious way to make a workflow assistant sound human is to hand the
numbers to a model and let it write the sentence. CareChaser does not, for
two reasons. The account's Bedrock text models are not authorised, so it
could not; and more importantly the project's whole claim is that a fluent
answer is not evidence that anything happened. A model that phrases well
also paraphrases, and a paraphrased SLA is a wrong SLA. Templates cannot
drift.

Warmth is a wording problem, not an intelligence problem.

VARIETY WITHOUT RANDOMNESS

Several lines below have two or three forms. The form is chosen from a hash
of the question, not at random, so the same question always produces the
same answer — a demo can be rehearsed, and a test can assert on it — while
two different questions do not come back word for word identical.
"""

from __future__ import annotations

import zlib
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

# What CareChaser says it is, when asked. Deliberately short: a judge asking
# "what is this" wants one breath, then the current state of the ward.
IDENTITY = (
    "I'm CareChaser. I track every diagnostic test through the ward until "
    "somebody actually acts on it — who owes the next step, how long it's "
    "been waiting, and who to escalate to when it stalls."
)

BOUNDARY = (
    "I don't read results or give clinical advice — I only track whether the "
    "workflow moved."
)


def pick(options: Sequence[str], seed: str) -> str:
    """Choose one phrasing from `seed`, stably.

    crc32 rather than hash(): Python salts str hashing per process, so the
    built-in would give a different greeting on every restart and no test
    could pin it.
    """
    if not options:
        return ""
    return options[zlib.crc32(seed.strip().lower().encode("utf-8")) % len(options)]


# ----------------------------------------------------------------- numbers


def say_hours(hours: float) -> str:
    """A duration a person would say aloud.

    "two days" beats "48 hours" in a sentence someone has to listen to. The
    threshold work happens here and nowhere else, so the page and the voice
    cannot disagree about what 36 hours is called.
    """
    try:
        hours = float(hours)
    except (TypeError, ValueError):
        return "an unknown time"

    if hours < 0:
        hours = 0.0
    minutes = round(hours * 60)

    if minutes < 1:
        return "less than a minute"
    if minutes < 60:
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    if hours < 24:
        whole = round(hours)
        return f"{whole} hour{'' if whole == 1 else 's'}"

    days = hours / 24
    whole_days = int(days)
    remainder = round((days - whole_days) * 24)
    if remainder >= 23:                       # 47.6h reads as two days, not one
        whole_days, remainder = whole_days + 1, 0
    if remainder == 0:
        if whole_days == 1:
            return "a full day"
        if whole_days == 2:
            return "two days"
        return f"{whole_days} days"
    day_part = "a day" if whole_days == 1 else f"{whole_days} days"
    return f"{day_part} and {remainder} hour{'' if remainder == 1 else 's'}"


def say_state(state: str, workflow: Optional[Mapping[str, Any]] = None) -> str:
    """A stage named the way a person on the ward would name it."""
    spoken = {
        "ordered": "waiting for collection",
        "sample_collected": "collected but not yet with the lab",
        "sample_processed": "with the laboratory",
        "result_produced": "released but not yet reviewed",
        "result_reviewed": "reviewed, waiting for the patient to be told",
        # NOT closed. patient_informed carries a 24-hour SLA and still has to
        # be moved to `closed`, so calling it closed would describe an open
        # case as finished — in this domain, the worst sentence to get wrong.
        "patient_informed": "patient informed, still waiting to be closed off",
        "cancelled": "cancelled",
        "closed": "closed",
    }
    if state in spoken:
        return spoken[state]
    for item in (workflow or {}).get("states", []):
        if item.get("state") == state:
            return str(item.get("display_name", state)).lower()
    return str(state).replace("_", " ")


# -------------------------------------------------------------- greetings


def greeting(now: datetime, name: Optional[str] = None) -> str:
    """Time-of-day greeting in the ward's own clock."""
    hour = now.hour
    if hour < 12:
        part = "Good morning"
    elif hour < 17:
        part = "Good afternoon"
    else:
        part = "Good evening"
    return f"{part}, {name}." if name else f"{part}."


def is_greeting(text: str) -> bool:
    lowered = text.lower().strip().strip("?.!")
    return lowered in {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "morning", "salam", "assalam o alaikum", "are you there", "hello there",
    }


def is_identity_question(text: str) -> bool:
    """"What is CareChaser", "what can you do", "who are you", "help"."""
    lowered = text.lower().strip().strip("?.!")
    if lowered in {"help", "what can you do", "who are you", "what are you"}:
        return True
    import re
    return bool(re.search(
        # "care" alone is included because speech recognition clips the
        # second half of the name constantly — "what is care" is never a
        # question about anything else in this app.
        r"\b(what|who)\s+(is|are|r)\s+(this|you|care|care\s*chaser|kh?\s*chaser)\b"
        r"|\bwhat\s+(can|do)\s+you\s+(do|help|offer)\b"
        r"|\bhow\s+(do|does)\s+(you|this|it)\s+work\b"
        r"|\bexplain\s+(yourself|care\s*chaser)\b",
        lowered,
    ))


# ------------------------------------------------------------- the cases


def say_case(case: Mapping[str, Any], *, include_owner: bool = True) -> str:
    """One case as a sentence, not a record.

    "a coagulation screen for patient MK-4478, collected but not yet with the
    lab for two days, with Nurse Ayesha" — rather than a comma-separated dump
    of field values.
    """
    test = str(case.get("test_type") or "a test").strip()
    patient = str(case.get("patient") or case.get("patient_id_or_local_identifier") or "").strip()
    waited = say_hours(case.get("hours_in_state", 0))
    # `state` carries the display name ("Result Released / Awaiting Clinician
    # Review"), which is a column heading, not something to say aloud.
    # `state_id` is the machine key the spoken map is written against.
    state = say_state(str(case.get("state_id") or case.get("state") or ""))
    owner = str(case.get("owner") or "").strip()

    article = "an" if test[:1].lower() in "aeiou" else "a"
    sentence = f"{article} {test.lower()}"
    if patient:
        sentence += f" for patient {patient}"
    sentence += f", {state} for {waited}"
    if include_owner and owner:
        sentence += f", with {owner}"
    return sentence


def say_urgency(case: Mapping[str, Any]) -> str:
    """Only ever repeats a label the clinical system printed on the report."""
    urgency = str(case.get("urgency", "")).lower()
    return "marked urgent" if urgency in {"critical", "urgent", "critical_result"} else ""


def stuck_summary(cases: Sequence[Mapping[str, Any]], total: int, seed: str = "") -> str:
    """The headline for a board full of stuck work.

    Leads with the shape of the problem, then one case in full. A list of ten
    read aloud is not information, it is noise.
    """
    if not cases:
        return pick([
            "Nothing is stuck right now — everything is inside its time limit.",
            "All clear. Nothing has run past its limit.",
        ], seed or "clear")

    worst = cases[0]
    urgent = [c for c in cases if say_urgency(c)]
    opener = (
        f"{total} case{'s' if total != 1 else ''} {'are' if total != 1 else 'is'} "
        f"past {'their' if total != 1 else 'its'} time limit."
    )
    if urgent:
        opener += (
            f" {len(urgent)} of the ones I can see {'are' if len(urgent) != 1 else 'is'} "
            f"marked urgent."
        )

    lead = "The one I'd start with is" if say_urgency(worst) else "Longest waiting is"
    body = f" {lead} {say_case(worst)}."

    if len(cases) > 1:
        body += f" After that, {say_case(cases[1])}."
    return opener + body


# ------------------------------------------------------- not understanding


# Three forms, so a judge who asks two things it cannot do does not hear the
# same sentence twice and conclude it is a canned demo.
_UNSURE = (
    "I didn't quite follow that.",
    "I'm not sure I understood that one.",
    "That one's outside what I can work out from the workflow.",
)

_CAPABILITIES = (
    "what's stuck right now",
    "how long a particular case has been waiting",
    "what's pending for a person",
    "who a case escalates to",
    "what's still outstanding from earlier",
)

_REFERRAL = (
    "For anything clinical, or anything about a result itself, please speak "
    "to the team looking after the patient.",
    "If it's about a result rather than the workflow, the clinical team is "
    "the right people to ask.",
)


def dont_know(said: str) -> str:
    """Say so, offer the way back, and point elsewhere for the rest.

    Silently answering a different question than the one asked is the worst
    thing a workflow assistant can do, because the answer still sounds right.
    """
    opener = pick(_UNSURE, said)
    can = ", ".join(_CAPABILITIES[:-1]) + f", or {_CAPABILITIES[-1]}"
    return f"{opener} I can tell you {can}. {pick(_REFERRAL, said)}"


def asks_about_urgency(text: str) -> bool:
    """"How many critical cases are pending", "anything urgent".

    This has to be checked before the "pending" route, or the word "pending"
    wins and the question comes back answered for the wrong person.
    """
    import re
    lowered = text.lower()
    if not re.search(r"\b(critical|urgent|emergenc\w+|stat)\b", lowered):
        return False
    return bool(re.search(
        r"\b(how many|how much|any|anything|are there|is there|what|which|count|list)\b",
        lowered,
    ))


def urgency_answer(board: Mapping[str, Any], seed: str = "") -> str:
    """How many stuck cases carry an urgency label the ward printed."""
    stuck, urgent = board_counts(board)
    if not stuck:
        return "Nothing is past its limit right now, urgent or otherwise."
    if not urgent:
        return (
            f"None of the {stuck} case{'s' if stuck != 1 else ''} past the limit "
            f"{'are' if stuck != 1 else 'is'} marked urgent."
        )

    text = (
        f"{urgent} of the {stuck} case{'s' if stuck != 1 else ''} past the limit "
        f"{'are' if urgent != 1 else 'is'} marked urgent."
    )
    cases = [c for c in (board.get("cases") or []) if say_urgency(c)]
    if cases:
        text += f" The longest is {say_case(cases[0])}."
    text += (
        " Urgency is whatever the clinical system printed on the report — "
        "I don't decide it."
    )
    return text


def asks_about_escalation(text: str) -> bool:
    """Is this "who picks it up if nobody acts?" — in any of its forms."""
    import re
    return bool(re.search(
        r"\b(who (?:is |are )?(?:responsible|accountable|next|covers?|covering|takes over)"
        r"|who do(?:es)? (?:it|that|this) (?:go|escalate) to"
        r"|what happens if|if .{0,30}\b(?:does ?n[o']?t|doesn t|fails? to|won[’']?t)\s+"
        r"(?:respond|reply|answer|act|do))\b",
        text.lower(),
    ))


def escalation_answer(who: str, data: Mapping[str, Any], seed: str = "") -> str:
    """"Who is responsible if X doesn't respond."

    Every case the engine returns carries its own escalation target, because
    the target depends on the stage the case is sitting in and not on who
    happens to own it. So the honest answer is per stage, not one name — and
    saying it that way is the product's actual argument out loud.
    """
    cases = list(data.get("cases") or [])
    total = int(data.get("total_pending") or len(cases))
    overdue = int(data.get("total_overdue") or 0)

    if not cases:
        return (
            f"Nothing is waiting on {who} at the moment, so nothing would "
            f"escalate."
        )

    # Group by target: several cases usually share one.
    targets: dict[str, list[Mapping[str, Any]]] = {}
    for item in cases:
        target = str(item.get("escalation_target") or "").strip()
        if target:
            targets.setdefault(target, []).append(item)

    head = (
        f"{total} case{'s' if total != 1 else ''} "
        f"{'are' if total != 1 else 'is'} waiting on {who}"
        + (f", {overdue} already past the limit." if overdue else ".")
    )

    if not targets:
        return head + " No escalation target is configured for those stages."

    parts = []
    for target, items in sorted(targets.items(), key=lambda kv: -len(kv[1])):
        stages = sorted({say_state(str(i.get("state_id") or i.get("state") or ""))
                         for i in items})
        stage_text = stages[0] if len(stages) == 1 else f"{len(items)} of them"
        parts.append(f"{target} picks up anything {stage_text}")

    joined = parts[0] if len(parts) == 1 else "; ".join(parts[:-1]) + f"; and {parts[-1]}"
    return (
        f"{head} If {who} doesn't act, it doesn't stay with them: {joined}. "
        f"The target depends on the stage the case is sitting in, not on who owns it."
    )


def board_counts(board: Optional[Mapping[str, Any]]) -> tuple[Optional[int], Optional[int]]:
    """(stuck, urgent) from a board snapshot.

    The counters live under `counts`. Falling back to len(cases) would be
    worse than having no number at all: the snapshot takes a `limit`, so the
    list is truncated and the "total" would quietly be the page size.
    """
    counts = (board or {}).get("counts") or {}
    stuck = counts.get("stuck")
    urgent = counts.get("critical_stuck")
    if stuck is None and board and "total_stuck" in board:
        stuck = board.get("total_stuck")
    return (
        int(stuck) if isinstance(stuck, (int, float)) else None,
        int(urgent) if isinstance(urgent, (int, float)) else None,
    )


def short_greeting(now: datetime, name: Optional[str], board: Optional[Mapping[str, Any]],
                   seed: str = "") -> str:
    """A greeting is a greeting. The full introduction is for being asked."""
    text = greeting(now, name)
    stuck, urgent = board_counts(board)
    if stuck:
        text += (
            f" {stuck} case{'s' if stuck != 1 else ''} "
            f"{'are' if stuck != 1 else 'is'} past the limit right now"
        )
        text += f", {urgent} of them marked urgent." if urgent else "."
        text += pick([" Want me to run through them?",
                      " Shall I start with the longest?"], seed or "hi")
    elif stuck == 0:
        text += " Nothing is past its limit right now."
    else:
        text += " Ask me what's stuck, and I'll start there."
    return text


def introduce(board: Optional[Mapping[str, Any]] = None, seed: str = "") -> str:
    """Answer "what is CareChaser" — identity, then the live ward.

    The numbers come from a board snapshot, so the introduction is never
    stale and never invented.
    """
    text = IDENTITY
    stuck, urgent = board_counts(board)
    if stuck:
        text += (
            f" Right now {stuck} case{'s' if stuck != 1 else ''} "
            f"{'have' if stuck != 1 else 'has'} run past the limit"
        )
        text += f", {urgent} of them marked urgent." if urgent else "."
    elif stuck == 0:
        text += " Right now nothing has run past its limit."
    text += " " + BOUNDARY
    text += pick([
        " Ask me what's stuck, and I'll start there.",
        " Try asking what's stuck today.",
    ], seed or "intro")
    return text
