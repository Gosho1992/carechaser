"""Tests for how CareChaser says things.

Two things are being protected here. That the wording stays human — a judge
hears sentences, not field dumps. And, more importantly, that making it
human did not make it loose: every number spoken must still be the engine's
number, and a question it cannot answer must be refused out loud rather than
quietly answered with a different one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import phrasing
from phrasing import (
    dont_know,
    greeting,
    introduce,
    is_greeting,
    is_identity_question,
    pick,
    say_case,
    say_hours,
    say_state,
    stuck_summary,
)

KARACHI = timezone(timedelta(hours=5))


def case(**over):
    base = {
        "test_type": "Coagulation screen",
        "patient": "MK-4478",
        "state_id": "sample_collected",
        "state": "Sample Collected",
        "hours_in_state": 48.0,
        "owner": "Nurse Ayesha",
        "urgency": "routine",
    }
    base.update(over)
    return base


# ------------------------------------------------------------- durations


@pytest.mark.parametrize("hours,expected", [
    (0.5, "30 minutes"),
    (1 / 60, "1 minute"),
    (1.0, "1 hour"),
    (3.4, "3 hours"),
    (24.0, "a full day"),
    (48.0, "two days"),
    (31.0, "a day and 7 hours"),
    (72.0, "3 days"),
])
def test_durations_are_spoken_the_way_a_person_would(hours, expected):
    assert say_hours(hours) == expected


def test_a_duration_just_short_of_a_day_boundary_rounds_up():
    """47.6 hours is two days to a listener, not "a day and 24 hours"."""
    assert say_hours(47.6) == "two days"


def test_a_nonsense_duration_does_not_crash_the_sentence():
    assert say_hours(None) == "an unknown time"
    assert say_hours(-5) == "less than a minute"


# ----------------------------------------------------------- the sentence


def test_a_case_reads_as_a_sentence_not_a_record():
    spoken = say_case(case())
    assert spoken == (
        "a coagulation screen for patient MK-4478, collected but not yet "
        "with the lab for two days, with Nurse Ayesha"
    )


def test_the_stage_is_named_the_way_the_ward_would_name_it():
    assert say_state("result_produced") == "released but not yet reviewed"
    assert say_state("ordered") == "waiting for collection"


def test_the_display_name_is_not_read_aloud_when_a_state_id_exists():
    """"Result Released / Awaiting Clinician Review" is a column heading."""
    spoken = say_case(case(state_id="result_produced",
                           state="Result Released / Awaiting Clinician Review"))
    assert "awaiting clinician review" not in spoken.lower()
    assert "released but not yet reviewed" in spoken


def test_a_vowel_test_name_gets_the_right_article():
    assert say_case(case(test_type="Arterial blood gas")).startswith("an arterial")


# -------------------------------------------------------------- the board


def test_the_summary_leads_with_the_shape_then_one_case():
    spoken = stuck_summary([case(), case(patient="MK-4458")], total=63)
    assert spoken.startswith("63 cases are past their time limit.")
    assert "MK-4478" in spoken
    assert "MK-4458" in spoken


def test_an_urgent_case_is_counted_but_never_interpreted():
    """Urgency is a label the clinical system printed. The wording may repeat
    it; nothing here may decide it."""
    spoken = stuck_summary([case(urgency="critical")], total=1)
    assert "marked urgent" in spoken
    assert "The one I'd start with is" in spoken


def test_a_routine_case_is_not_described_as_urgent():
    assert "urgent" not in stuck_summary([case()], total=1)


def test_an_empty_board_is_good_news_not_an_error():
    spoken = stuck_summary([], total=0).lower()
    assert any(word in spoken for word in ("nothing", "all clear"))
    assert "past" not in spoken          # nothing is reported as overdue


def test_only_two_cases_are_read_aloud_however_many_are_stuck():
    """Sixty-three read out is noise, not information."""
    spoken = stuck_summary([case(patient=f"MK-{n}") for n in range(4400, 4420)],
                           total=63)
    named = [n for n in range(4400, 4420) if f"MK-{n}" in spoken]
    assert len(named) == 2


# ----------------------------------------------------- not understanding


def test_a_question_it_cannot_answer_is_refused_out_loud():
    said = "what should I prescribe for this patient"
    spoken = dont_know(said)
    assert any(word in spoken.lower() for word in ("didn't", "not sure", "outside"))
    assert "what's stuck right now" in spoken


def test_the_refusal_points_at_a_human_for_clinical_questions():
    spoken = dont_know("is 6.8 potassium dangerous")
    assert "clinical" in spoken.lower() or "team" in spoken.lower()


def test_two_different_questions_do_not_get_word_for_word_the_same_refusal():
    """A judge who asks two things it cannot do should not conclude the
    whole thing is one canned string."""
    variants = {dont_know(q) for q in (
        "what should I prescribe", "who is the president",
        "tell me a joke", "what is the weather", "how is the patient doing",
    )}
    assert len(variants) > 1


def test_the_same_question_always_gets_the_same_refusal():
    """Variety must not cost determinism: a demo has to be rehearsable."""
    assert dont_know("tell me a joke") == dont_know("tell me a joke")
    assert dont_know("Tell Me A Joke  ") == dont_know("tell me a joke")


def test_phrasing_choice_survives_a_restart():
    """crc32, not the built-in hash, which Python salts per process."""
    assert pick(["a", "b", "c"], "same question") == pick(["a", "b", "c"], "same question")
    assert pick([], "anything") == ""


# ---------------------------------------------------------- introductions


@pytest.mark.parametrize("hour,expected", [
    (9, "Good morning"), (14, "Good afternoon"), (20, "Good evening"),
])
def test_the_greeting_follows_the_hour_on_the_ward(hour, expected):
    said = greeting(datetime(2026, 10, 16, hour, tzinfo=KARACHI), "Dr. Khan")
    assert said == f"{expected}, Dr. Khan."


def test_identity_questions_are_recognised_in_the_forms_people_use():
    for question in ("what is carechaser", "what is care chaser?", "who are you",
                     "what can you do", "how does this work", "help"):
        assert is_identity_question(question), question


def test_a_greeting_is_not_mistaken_for_a_question():
    assert is_greeting("hello")
    assert is_greeting("Good morning")
    assert not is_greeting("what is stuck today")


def test_the_introduction_carries_live_numbers_not_a_canned_paragraph():
    spoken = introduce({"counts": {"stuck": 63, "critical_stuck": 12}})
    assert "63 cases" in spoken
    assert "12 of them marked urgent" in spoken


def test_a_truncated_case_list_is_never_read_as_a_total():
    """get_board_snapshot takes a limit, so len(cases) is the page size.
    Reporting it as the total would understate the ward."""
    import re
    spoken = introduce({"counts": {"stuck": 63}, "cases": [{}, {}, {}]})
    assert "63 cases" in spoken
    assert not re.search(r"\b3\b", spoken)      # the page size, never spoken


def test_the_introduction_states_the_boundary():
    """The one claim that must never be dropped for brevity."""
    spoken = introduce({"counts": {"stuck": 5}})
    assert "don't read results" in spoken or "clinical advice" in spoken


def test_the_introduction_works_before_any_data_exists():
    spoken = introduce(None)
    assert "CareChaser" in spoken
    assert "tracking" not in spoken        # no invented count


# ------------------------------------------------------------ escalation


def test_the_escalation_answer_names_the_target_per_stage():
    """The target depends on the stage, not on who owns the case — so the
    answer must not collapse to a single name."""
    data = {
        "total_pending": 2, "total_overdue": 1,
        "cases": [
            {"state_id": "ordered", "escalation_target": "Dr. Iqbal"},
            {"state_id": "sample_collected", "escalation_target": "Sr. Nurse Fatima"},
        ],
    }
    spoken = phrasing.escalation_answer("Nurse Ayesha", data)
    assert "Dr. Iqbal" in spoken
    assert "Sr. Nurse Fatima" in spoken
    assert "depends on the stage" in spoken


def test_nobody_waiting_means_nothing_would_escalate():
    spoken = phrasing.escalation_answer("Dr. Ali", {"total_pending": 0, "cases": []})
    assert "nothing would escalate" in spoken.lower()


def test_a_stage_with_no_configured_target_is_said_plainly():
    """Silence here would imply there is an escalation path when there isn't."""
    data = {"total_pending": 1, "cases": [{"state_id": "ordered", "escalation_target": ""}]}
    spoken = phrasing.escalation_answer("Lab Desk", data)
    assert "no escalation target" in spoken.lower()


def test_escalation_questions_are_recognised_in_the_forms_people_ask():
    for question in (
        "if nurse Ayesha does not respond who is responsible",
        "who covers for Dr Khan",
        "who is next if nobody acts",
        "what happens if the lab doesn't respond",
        "who does that escalate to",
    ):
        assert phrasing.asks_about_escalation(question), question


def test_the_greeting_leads_with_the_count_when_work_is_late():
    spoken = phrasing.short_greeting(
        datetime(2026, 10, 16, 9, tzinfo=KARACHI), "Dr. Khan",
        {"counts": {"stuck": 8, "critical_stuck": 1}})
    assert spoken.startswith("Good morning, Dr. Khan.")
    assert "8 cases are past the limit" in spoken
    assert "1 of them marked urgent" in spoken


def test_a_quiet_ward_is_greeted_as_quiet():
    spoken = phrasing.short_greeting(
        datetime(2026, 10, 16, 9, tzinfo=KARACHI), "Dr. Khan", {"counts": {"stuck": 0}})
    assert "Nothing is past its limit" in spoken


# --------------------------------------------- what the states are called


def test_patient_informed_is_not_described_as_closed():
    """It carries a 24-hour SLA and still has to be moved to `closed`.
    Calling an open case closed is the worst sentence to get wrong here."""
    spoken = say_state("patient_informed")
    assert "closed with" not in spoken
    assert "still waiting to be closed" in spoken


def test_a_stuck_patient_informed_case_does_not_read_as_finished():
    spoken = say_case(case(state_id="patient_informed", hours_in_state=50))
    assert "informed" in spoken
    assert not spoken.startswith("a coagulation screen for patient MK-4478, closed")


# ---------------------------------------------------------- urgency asks


def test_urgency_questions_are_recognised_before_the_pending_route():
    for question in ("how many critical cases are pending?",
                     "anything urgent today", "are there any urgent cases",
                     "how many urgent"):
        assert phrasing.asks_about_urgency(question), question


def test_an_ordinary_pending_question_is_not_an_urgency_question():
    assert not phrasing.asks_about_urgency("what is pending for Dr Khan")
    assert not phrasing.asks_about_urgency("what is stuck today")


def test_the_urgency_answer_is_scoped_to_the_whole_queue():
    spoken = phrasing.urgency_answer({"counts": {"stuck": 69, "critical_stuck": 10}})
    assert "10 of the 69 cases" in spoken


def test_the_urgency_answer_never_claims_to_decide_urgency():
    spoken = phrasing.urgency_answer({"counts": {"stuck": 69, "critical_stuck": 10}})
    assert "don't decide it" in spoken


def test_no_urgent_cases_is_said_plainly_not_omitted():
    spoken = phrasing.urgency_answer({"counts": {"stuck": 12, "critical_stuck": 0}})
    assert "None of the 12" in spoken


def test_a_clipped_product_name_is_still_recognised():
    """Speech recognition drops the second half of the name constantly."""
    assert is_identity_question("what is care")
