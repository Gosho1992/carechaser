"""CareChaser chase layer.

Turns a stuck case into a message that actually reaches the person who can
unstick it:

    engine says STUCK -> facts -> drafter writes prose -> channel delivers

Two boundaries matter here and are enforced in code, not just in prose:

1. The model never supplies facts. Case id, hours, names and the stage come
   from the engine and are passed in. The drafter only phrases them.
2. Every draft is checked against those facts before it can be sent. If the
   model drops the case id or invents a different number of hours, the draft
   is rejected and the deterministic template is sent instead.

Drafters:
    BedrockDrafter  - Amazon Bedrock Converse. The real path; covers the
                      AWS Builder mini challenge.
    TemplateDrafter - deterministic string template. Used with no AWS
                      credentials and in tests. Not a language model.

Channels:
    LogChannel      - appends to chase_log.jsonl. Default, and what the demo
                      shows on screen.
    EmailChannel    - Amazon SES.
    WebhookChannel  - POST to any URL (WhatsApp gateway, Slack, hospital IT).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

DEFAULT_MODEL_ID = os.environ.get(
    "CARECHASER_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")

DRAFT_SYSTEM = """You write short follow-up messages for hospital staff.

A diagnostic case has been waiting longer than the agreed time limit. Write a
message to the person who can move it forward.

Rules:
- Use ONLY the facts given to you. Never add a number, a name, a date or a
  clinical detail that is not in the facts.
- Never interpret the test result, never suggest a diagnosis, never imply
  what the result might show, and never state any clinical urgency beyond
  the urgency label given.
- No blame. The point is to unstick the case, not to criticise anyone.
- Two sentences maximum. Say what is waiting, how long, and what is needed.
- Always include the case id exactly as given.
- Plain text, no greeting line, no signature."""


# ------------------------------------------------------------- saying times
#
# A critical result has a limit measured in minutes, not hours. Rounding
# everything to whole hours made a potassium result that was fifty minutes
# past a ten minute limit read as "0 hours past the 0.17 hour limit" — the
# most urgent case on the board describing itself as not late at all.


def say_duration(hours: float) -> str:
    """A length of time as a person would say it: '50 minutes', '1 hour'."""
    minutes = round(hours * 60)
    if minutes < 60:
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    whole = round(hours)
    return f"{whole} hour{'' if whole == 1 else 's'}"


def say_limit(hours: float) -> str:
    """The same length used as a label: 'the 10 minute limit'. Singular."""
    minutes = round(hours * 60)
    if minutes < 60:
        return f"{minutes} minute"
    return f"{round(hours)} hour"


def duration_numbers(hours: float) -> set[int]:
    """Every number a correct sentence about this duration may contain."""
    return {round(hours), int(hours), round(hours * 60)}


@dataclass
class ChaseFacts:
    """Everything the message may mention. Nothing else is allowed in."""

    case_id: str
    test_type: str
    patient: str
    state: str
    hours_in_state: float
    sla_hours: Optional[float]
    owner: str
    chase_target: str
    urgency: str = "routine"
    escalated: bool = False

    @property
    def overdue_exact(self) -> Optional[float]:
        """Hours past the limit, unrounded. Rounding here loses a critical
        case's whole breach, so the rounding happens where it is displayed."""
        if self.sla_hours is None:
            return None
        return self.hours_in_state - self.sla_hours

    @property
    def hours_overdue(self) -> Optional[float]:
        if self.overdue_exact is None:
            return None
        return round(self.overdue_exact, 1)

    def to_prompt(self) -> str:
        lines = [
            f"case id: {self.case_id}",
            f"test: {self.test_type}",
            f"patient reference: {self.patient}",
            f"current stage: {self.state}",
            f"time waiting in this stage: {say_duration(self.hours_in_state)}",
            "agreed limit for this stage: "
            + (say_duration(self.sla_hours) if self.sla_hours is not None else "none"),
            "past the limit by: "
            + (
                say_duration(self.overdue_exact)
                if self.overdue_exact is not None and self.overdue_exact > 0
                else "not past the limit"
            ),
            f"currently with: {self.owner}",
            f"message is addressed to: {self.chase_target}",
            f"urgency label: {self.urgency}",
        ]
        if self.escalated:
            lines.append(
                "note: this has been escalated because the original owner did not act"
            )
        return "\n".join(lines)


@dataclass
class Draft:
    text: str
    source: str  # bedrock | template | template-after-rejection
    rejected_reason: Optional[str] = None


@dataclass
class ChaseRecord:
    case_id: str
    to: str
    text: str
    source: str
    channel: str
    sent_at: str
    delivered: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ drafters


class Drafter(Protocol):
    name: str

    def draft(self, facts: ChaseFacts) -> str: ...


class TemplateDrafter:
    """Deterministic fallback. Always factually correct, never eloquent."""

    name = "template"

    def draft(self, facts: ChaseFacts) -> str:
        overdue = facts.overdue_exact
        if overdue is not None and overdue > 0:
            tail = (
                f", which is {say_duration(overdue)} past the "
                f"{say_limit(facts.sla_hours)} limit"
            )
        else:
            tail = ""
        lead = "Escalated: " if facts.escalated else ""
        return (
            f"{lead}{facts.case_id} ({facts.test_type}, patient {facts.patient}) "
            f"has been waiting at {facts.state} for "
            f"{say_duration(facts.hours_in_state)}"
            f"{tail}. It is currently with {facts.owner} and needs the next step "
            f"recorded."
        )


class BedrockDrafter:
    """Amazon Bedrock Converse. Phrases the facts; never supplies them."""

    name = "bedrock"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, region: str = DEFAULT_REGION):
        import boto3

        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def draft(self, facts: ChaseFacts) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": DRAFT_SYSTEM}],
            messages=[{"role": "user", "content": [{"text": facts.to_prompt()}]}],
            inferenceConfig={"maxTokens": 220, "temperature": 0.2},
        )
        blocks = response["output"]["message"]["content"]
        return " ".join(b["text"] for b in blocks if "text" in b).strip()


# ----------------------------------------------------------------- guardrail


def check_draft(text: str, facts: ChaseFacts) -> Optional[str]:
    """Return a rejection reason, or None if the draft is safe to send.

    This is the check that lets a language model near a clinical workflow:
    a draft that drops the case id, invents numbers, or strays into clinical
    language never reaches anyone.
    """
    if not text or len(text.strip()) < 20:
        return "draft was empty or too short"
    if len(text) > 600:
        return "draft was too long to be a chase message"
    if facts.case_id.lower() not in text.lower():
        return "draft did not contain the case id"

    # A duration may be stated in hours or in minutes, so both readings of
    # each fact are allowed. Nothing else is.
    allowed = set(duration_numbers(facts.hours_in_state))
    if facts.sla_hours is not None:
        allowed |= duration_numbers(facts.sla_hours)
    if facts.overdue_exact is not None:
        allowed |= duration_numbers(facts.overdue_exact)
        allowed.add(round(facts.hours_overdue))
    # Numbers inside the case id or patient reference are not claims.
    scrubbed = text.lower()
    for token in (facts.case_id, facts.patient, facts.test_type):
        scrubbed = scrubbed.replace(token.lower(), " ")
    for found in re.findall(r"\b\d+(?:\.\d+)?\b", scrubbed):
        value = round(float(found))
        if value not in allowed:
            return f"draft contained an unsupported number: {found}"

    banned = [
        "diagnos", "malignan", "cancer", "tumour", "tumor", "abnormal result",
        "likely", "suggests", "prescrib", "treatment", "dose", "mg ",
    ]
    lowered = text.lower()
    for word in banned:
        if word in lowered:
            return f"draft used clinical language: {word.strip()}"
    return None


# ------------------------------------------------------------------ channels


class Channel(Protocol):
    name: str

    def send(self, to: str, text: str, facts: ChaseFacts) -> dict[str, Any]: ...


class LogChannel:
    """Append-only JSONL log. The demo's visible outbox."""

    name = "log"

    def __init__(self, path: str | Path = "chase_log.jsonl"):
        self.path = Path(path)

    def send(self, to: str, text: str, facts: ChaseFacts) -> dict[str, Any]:
        entry = {
            "case_id": facts.case_id,
            "kind": "chase",
            "to": to,
            "text": text,
            "from_owner": facts.owner,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        return {"ok": True, "path": str(self.path)}


class EmailChannel:
    """Amazon SES."""

    name = "email"

    def __init__(self, sender: str, recipients: dict[str, str], region: str = DEFAULT_REGION):
        import boto3

        self.sender = sender
        self.recipients = recipients  # person or role -> email address
        self.client = boto3.client("ses", region_name=region)

    def send(self, to: str, text: str, facts: ChaseFacts) -> dict[str, Any]:
        address = self.recipients.get(to)
        if not address:
            return {"ok": False, "error": f"no email address on file for {to}"}
        response = self.client.send_email(
            Source=self.sender,
            Destination={"ToAddresses": [address]},
            Message={
                "Subject": {"Data": f"CareChaser: {facts.case_id} needs a next step"},
                "Body": {"Text": {"Data": text}},
            },
        )
        return {"ok": True, "message_id": response.get("MessageId"), "to": address}


class WebhookChannel:
    """POST the message anywhere: a WhatsApp gateway, Slack, hospital IT."""

    name = "webhook"

    def __init__(self, url: str, timeout: float = 10.0):
        self.url = url
        self.timeout = timeout

    def send(self, to: str, text: str, facts: ChaseFacts) -> dict[str, Any]:
        import httpx

        try:
            response = httpx.post(
                self.url,
                json={"to": to, "case_id": facts.case_id, "text": text},
                timeout=self.timeout,
            )
            return {"ok": response.is_success, "status": response.status_code}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


# -------------------------------------------------------------------- runner


class Chaser:
    """Draft, check, deliver."""

    def __init__(self, drafter: Drafter | None = None, channel: Channel | None = None):
        self.drafter = drafter or build_drafter()
        self.channel = channel or LogChannel()

    def compose(self, facts: ChaseFacts) -> Draft:
        fallback = TemplateDrafter()
        if isinstance(self.drafter, TemplateDrafter):
            return Draft(text=fallback.draft(facts), source="template")
        try:
            text = self.drafter.draft(facts)
        except Exception as exc:
            return Draft(
                text=fallback.draft(facts),
                source="template-after-rejection",
                rejected_reason=f"drafter failed: {exc}",
            )
        reason = check_draft(text, facts)
        if reason:
            return Draft(
                text=fallback.draft(facts),
                source="template-after-rejection",
                rejected_reason=reason,
            )
        return Draft(text=text, source=self.drafter.name)

    def send(self, facts: ChaseFacts, *, dry_run: bool = False) -> ChaseRecord:
        draft = self.compose(facts)
        detail: dict[str, Any] = {}
        if draft.rejected_reason:
            detail["draft_rejected"] = draft.rejected_reason
        if dry_run:
            return ChaseRecord(
                case_id=facts.case_id,
                to=facts.chase_target,
                text=draft.text,
                source=draft.source,
                channel=self.channel.name,
                sent_at=datetime.now(timezone.utc).isoformat(),
                delivered=False,
                detail={**detail, "dry_run": True},
            )
        outcome = self.channel.send(facts.chase_target, draft.text, facts)
        return ChaseRecord(
            case_id=facts.case_id,
            to=facts.chase_target,
            text=draft.text,
            source=draft.source,
            channel=self.channel.name,
            sent_at=datetime.now(timezone.utc).isoformat(),
            delivered=bool(outcome.get("ok")),
            detail={**detail, **outcome},
        )


def build_drafter() -> Drafter:
    """Bedrock when credentials exist, deterministic template otherwise."""
    if os.environ.get("CARECHASER_FORCE_OFFLINE"):
        return TemplateDrafter()
    try:
        import boto3

        if boto3.Session().get_credentials() is None:
            return TemplateDrafter()
        return BedrockDrafter()
    except Exception:
        return TemplateDrafter()


def build_channel(log_path: str | Path | None = None) -> Channel:
    """Chosen by environment so the demo needs no code change."""
    url = os.environ.get("CARECHASER_WEBHOOK_URL")
    if url:
        return WebhookChannel(url)
    sender = os.environ.get("CARECHASER_SES_SENDER")
    if sender:
        recipients = json.loads(os.environ.get("CARECHASER_SES_RECIPIENTS", "{}"))
        return EmailChannel(sender, recipients)
    return LogChannel(log_path or os.environ.get("CARECHASER_OUTBOX", "chase_outbox.jsonl"))
