"""CareChaser report intake.

Turns the document a nurse already has into tracked work:

    report text -> extraction -> validation -> review -> cases

The boundary that makes this safe to build at all:

    CareChaser extracts WHICH tests are on the report and WHETHER each one
    has been done. It never extracts a result value, never decides anything
    is abnormal, and never infers urgency from clinical content.

    Urgency is read only where the laboratory has written it as a label —
    "CRITICAL", "URGENT", "STAT" — because that is the clinical system
    speaking, not CareChaser forming a view. A report with no such label
    yields a routine case, whatever numbers are on it.

So "Serum potassium 6.8 mmol/L — result available" yields:

    {"test_name": "Serum potassium", "status": "result_available"}

and the 6.8 is discarded before it reaches the engine. A stripping pass
enforces that even if a model tries to include it.

Two extractors:

    BedrockExtractor - Amazon Bedrock, strict JSON out. The real path.
    RuleExtractor    - deterministic line and CSV parser. Used with no AWS
                       credentials and in tests. Not a language model.

Nothing here writes to the case store. Extraction produces rows for a human
to review; creating the cases is a separate, deliberate step.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

DEFAULT_MODEL_ID = os.environ.get(
    "CARECHASER_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Report wording -> the workflow stage the case should start in.
# Deliberately explicit: a status the hospital uses that is not listed here
# is flagged for a human rather than guessed at.
STATUS_TO_STATE: dict[str, str] = {
    "ordered": "ordered",
    "requested": "ordered",
    "not collected": "ordered",
    "collected": "sample_collected",
    "sample collected": "sample_collected",
    "in lab": "sample_processed",
    "processing": "sample_processed",
    "in progress": "sample_processed",
    "pending": "sample_processed",
    "not completed": "sample_processed",
    "awaiting result": "sample_processed",
    "completed": "result_produced",
    "released": "result_produced",
    "result available": "result_produced",
    "result released": "result_produced",
    "resulted": "result_produced",
    "reviewed": "result_reviewed",
    "seen by doctor": "result_reviewed",
    "patient informed": "patient_informed",
}

EXTRACT_SYSTEM = """You read hospital test reports and list what is on them.

Return ONLY a JSON object, no prose, in this exact shape:

{"patient": "<identifier or null>",
 "items": [{"test_name": "<name>", "status": "<status as written>"}]}

Rules:
- List every test mentioned, once each.
- status must be the status as the document states it, lowercased, such as
  "completed", "pending", "result available", "not collected".
- NEVER include a result value, measurement, unit, reference range, or any
  clinical finding. You are listing which tests exist and whether they are
  done. The numbers are not your concern and must not appear anywhere.
- Never infer urgency, never say anything is abnormal, never add a test that
  is not written in the document.
- If the patient identifier is not stated, use null."""

# A number is a RESULT unless it is part of a word (B12, T3, HbA1c, COVID-19)
# or a duration (24 hour urine). Everything else goes: a bare number next to a
# test name is a measurement, and measurements must never enter the workflow.
MEASUREMENT_PATTERNS = [
    # Ranges first: otherwise the unit pattern eats "145 mmol/L" out of
    # "135-145 mmol/L" and leaves the orphaned 135 behind.
    re.compile(r"\b\d+(?:\.\d+)?\s*[-–/]\s*\d+(?:\.\d+)?\b"),          # ranges, 135-145
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:mmol/l|mg/dl|g/dl|iu/l|u/l|mcg|µg|ng/ml|ml|%|cells?/)\S*", re.I),
    re.compile(r"\b(?:high|low|elevated|abnormal|critical value|out of range|positive|negative)\b", re.I),
    re.compile(r"\b(?:result|value|level|reading)s?\s*[:=]?\s*\d+(?:\.\d+)?", re.I),
]

# Urgency labels a laboratory writes on a report. Matched as whole words so
# "understated" cannot contain "stat". Nothing is inferred from a value: a
# potassium of 6.8 is routine here unless the report says otherwise.
URGENCY_MARKER = re.compile(r"\b(critical|urgent|stat)\b", re.I)

TIME_WORDS = {"hour", "hours", "hr", "hrs", "h", "day", "days", "minute", "minutes", "min"}

# Units left stranded once their number is removed are not part of a name.
ORPHAN_UNITS = re.compile(
    r"\b(?:mmol/l|mg/dl|g/dl|iu/l|u/l|miu/l|ng/l|ng/ml|pmol/l|umol/l|µmol/l|"
    r"x10\^9/l|kpa|mcg|µg|ml|mmol|mg|inr|ref(?:erence)?(?:\s+range)?)\b",
    re.I,
)


def _loose_state(status_text: str) -> Optional[str]:
    """Map a status that is not an exact phrase, without ever guessing.

    A written status often carries more than one known word: "reviewed,
    follow-up pending" holds both "reviewed" and "pending", which point at
    different stages. Taking whichever came first in the dictionary made the
    answer depend on how the table happened to be typed, and "pending" sits
    above "reviewed" in it, so every reviewed result was filed as still
    processing.

    The word the writer led with is the one that describes the case. Ties go
    to the longer phrase, since "result released" is more specific than
    "released".
    """
    hits = [
        (status_text.find(phrase), -len(phrase), mapped)
        for phrase, mapped in STATUS_TO_STATE.items()
        if phrase in status_text
    ]
    if not hits:
        return None
    return min(hits)[2]


def _is_bare_number(token: str) -> bool:
    """A token that is only digits, decimals or a numeric range."""
    return bool(re.fullmatch(r"\d+(?:[.,]\d+)?(?:\s*[-–/]\s*\d+(?:[.,]\d+)?)?", token))


def strip_values(text: str) -> str:
    """Remove measurements, ranges and clinical judgements from a fragment.

    Kept: numbers bound into a word (B12, T3, HbA1c, COVID-19, 25-OH) and
    durations (24 hour urine), because those are parts of a test's name.
    Dropped: every standalone number, with or without a unit. A test name
    mangled by this rule is visible in the review table and a human fixes it;
    a result value smuggled into the workflow is not recoverable.
    """
    cleaned = text
    for pattern in MEASUREMENT_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    tokens = cleaned.split()
    kept: list[str] = []
    for index, token in enumerate(tokens):
        stripped = token.strip(".,:;()[]")
        if _is_bare_number(stripped):
            following = tokens[index + 1].strip(".,:;").lower() if index + 1 < len(tokens) else ""
            if following in TIME_WORDS:
                kept.append(token)          # "24 hour urine"
            continue                         # otherwise it is a measurement
        kept.append(token)

    cleaned = ORPHAN_UNITS.sub(" ", " ".join(kept))
    return re.sub(r"\s{2,}", " ", cleaned).strip(" .,:;-/")


@dataclass
class IntakeRow:
    """One test from a report, ready for a human to approve or correct."""

    test_name: str
    status_text: str
    raw_text: str = ""
    removed: list[str] = field(default_factory=list)
    state: Optional[str] = None
    owner_role: Optional[str] = None
    patient: Optional[str] = None
    hours_waiting: float = 0.0
    urgency: str = "routine"
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues and bool(self.state and self.owner_role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "status_text": self.status_text,
            "raw_text": self.raw_text,
            "removed": list(self.removed),
            "state": self.state,
            "owner_role": self.owner_role,
            "patient": self.patient,
            "hours_waiting": self.hours_waiting,
            "urgency": self.urgency,
            "issues": list(self.issues),
            "ok": self.ok,
        }


# ---------------------------------------------------------------- extractors


class RuleExtractor:
    """Deterministic parser for 'Test name — status' lines and CSV.

    Not a language model. Used when no AWS credentials are present, and in
    every test, so extraction behaviour can be asserted exactly.
    """

    name = "rules"

    SEPARATORS = ["—", "–", " - ", ":", "\t", "|", ","]

    def extract(self, text: str) -> dict[str, Any]:
        rows: list[dict[str, str]] = []
        patient = None

        csv_rows = self._try_csv(text)
        if csv_rows is not None:
            return {"patient": self._find_patient(text), "items": csv_rows}

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            found = self._patient_in(line)
            if found and patient is None:
                patient = found
                continue
            pair = self._split(line)
            if pair:
                rows.append({"test_name": pair[0], "status": pair[1].lower()})
        return {"patient": patient, "items": rows}

    def _try_csv(self, text: str) -> Optional[list[dict[str, str]]]:
        if "," not in text or "\n" not in text:
            return None
        try:
            reader = csv.DictReader(io.StringIO(text))
            fields = [f.lower().strip() for f in (reader.fieldnames or [])]
            if not {"test", "status"} <= set(fields) and not {"test_name", "status"} <= set(fields):
                return None
            out = []
            for row in reader:
                lowered = {k.lower().strip(): (v or "").strip() for k, v in row.items() if k}
                name = lowered.get("test") or lowered.get("test_name")
                status = lowered.get("status")
                if name and status:
                    out.append({"test_name": name, "status": status.lower()})
            return out or None
        except csv.Error:
            return None

    # "Patient ref: MK-4471", "Patient ID MK-4471", "Patient: MK-4471".
    # The optional label has to be consumed, or the reference comes back as
    # the word "ref" and every case carries the same useless identifier.
    PATIENT = re.compile(
        r"\bpatient\b\s*(?:ref(?:erence)?|id(?:entifier)?|no\.?|number)?"
        # A reference must carry a digit, so prose ("Patient was seen") is
        # not mistaken for one.
        r"\s*[:#-]?\s*([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)",
        re.I,
    )

    @classmethod
    def _patient_in(cls, line: str) -> Optional[str]:
        match = cls.PATIENT.search(line)
        return match.group(1) if match else None

    def _find_patient(self, text: str) -> Optional[str]:
        for line in text.splitlines():
            found = self._patient_in(line)
            if found:
                return found
        return None

    # A printed lab report leads the eye from the test to its result with a
    # row of dots. That run of dots is the real separator, and it has to be
    # tried first: a comma inside "released, not yet acknowledged" would
    # otherwise split the line in the middle of the status, leaving the value
    # attached to the test name.
    DOTTED = re.compile(r"^(.{2,60}?)\s*\.{3,}\s*(.+)$")

    # OCR does not return the dots. Textract reads a printed row as separate
    # blocks and scan_intake rejoins them with spaces, so on a scanned report
    # the gap between columns is what survives, not the leader. This has to be
    # tried before the comma, or "reviewed, follow-up pending" splits the line
    # in the middle of its own status.
    COLUMNS = re.compile(r"^(.{2,60}?)\s{2,}(.+)$")

    def _split(self, line: str) -> Optional[tuple[str, str]]:
        for pattern in (self.DOTTED, self.COLUMNS):
            match = pattern.match(line)
            if match:
                left, right = match.group(1).strip(" .-—–:\t|"), match.group(2).strip()
                if left and right:
                    return left, right
        for sep in self.SEPARATORS:
            if sep in line:
                left, _, right = line.partition(sep)
                left, right = left.strip(" -—–:\t|"), right.strip()
                if left and right and len(left) < 80:
                    return left, right
        return None


class BedrockExtractor:
    """Amazon Bedrock. Reads the document; never supplies clinical facts."""

    name = "bedrock"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, region: str = DEFAULT_REGION):
        import boto3

        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def extract(self, text: str) -> dict[str, Any]:
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": EXTRACT_SYSTEM}],
            messages=[{"role": "user", "content": [{"text": text}]}],
            inferenceConfig={"maxTokens": 900, "temperature": 0.0},
        )
        blocks = response["output"]["message"]["content"]
        raw = " ".join(b["text"] for b in blocks if "text" in b).strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
            raise ValueError("extractor did not return the expected shape")
        return parsed


def build_extractor():
    if os.environ.get("CARECHASER_FORCE_OFFLINE"):
        return RuleExtractor()
    try:
        import boto3

        if boto3.Session().get_credentials() is None:
            return RuleExtractor()
        return BedrockExtractor()
    except Exception:
        return RuleExtractor()


# ---------------------------------------------------------------- validation


def validate(
    extracted: Mapping[str, Any], workflow: Mapping[str, Any]
) -> list[IntakeRow]:
    """Turn raw extraction into rows checked against the real workflow.

    A row is only `ok` if its status maps to a stage that exists and that
    stage's owner role exists. Anything else is flagged for the human, never
    guessed: the model must not be able to invent a stage or a role, because
    that would let it edit the workflow configuration.
    """
    states = {s["state"]: s for s in workflow.get("states", [])}
    startable = {k: v for k, v in states.items() if not v.get("terminal")}
    roles = set(workflow.get("roles") or {})
    patient = extracted.get("patient")

    rows: list[IntakeRow] = []
    seen: set[str] = set()

    for item in extracted.get("items", []):
        if not isinstance(item, Mapping):
            continue
        raw_name = str(item.get("test_name", "")).strip()
        raw_status = str(item.get("status", "")).strip().lower()
        name = strip_values(raw_name)
        status_text = strip_values(raw_status)
        if not name:
            continue

        row = IntakeRow(
            test_name=name,
            status_text=status_text,
            raw_text=" — ".join(x for x in (raw_name, raw_status) if x),
            removed=removed_fragments(raw_name, name) + removed_fragments(raw_status, status_text),
            patient=patient,
            # Read from the label the laboratory printed, not from any value.
            urgency=("critical"
                     if URGENCY_MARKER.search(f"{raw_name} {raw_status}")
                     else "routine"),
        )

        key = name.lower()
        if key in seen:
            row.issues.append("duplicate test on this report")
        seen.add(key)

        if not status_text:
            row.issues.append("no status given")
        else:
            state = STATUS_TO_STATE.get(status_text)
            if state is None:
                state = _loose_state(status_text)
            if state is None:
                row.issues.append(f"status '{status_text}' is not one this workflow knows")
            elif state not in startable:
                row.issues.append(f"'{state}' is not a stage a case can start in")
            else:
                row.state = state
                owner_role = str(startable[state].get("owner_role", ""))
                if owner_role in roles:
                    row.owner_role = owner_role
                else:
                    row.issues.append(f"stage '{state}' has no valid owner role configured")

        rows.append(row)

    return rows


def removed_fragments(raw: str, cleaned: str) -> list[str]:
    """What the sanitiser took out, as a human-readable list.

    Shown in the review table so a person can see exactly which clinical
    content CareChaser refused to carry, rather than trusting that it did.
    """
    kept = cleaned.lower().split()
    out: list[str] = []
    for token in raw.split():
        bare = token.strip(".,:;()[]")
        if bare and bare.lower() not in kept and bare.lower() not in " ".join(kept):
            out.append(bare)
    return out


def extract_report(
    text: str, workflow: Mapping[str, Any], extractor: Any | None = None
) -> dict[str, Any]:
    """Full intake pass: extract, strip values, validate. Writes nothing."""
    if not text or not text.strip():
        return {"ok": False, "message": "The report was empty.", "rows": [], "source": None}

    extractor = extractor or build_extractor()
    try:
        extracted = extractor.extract(text)
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Could not read that report: {exc}",
            "rows": [],
            "source": extractor.name,
        }

    rows = validate(extracted, workflow)
    if not rows:
        return {
            "ok": False,
            "message": "No tests were found in that report.",
            "rows": [],
            "source": extractor.name,
            "patient": extracted.get("patient"),
        }

    ready = [r for r in rows if r.ok]
    flagged = [r for r in rows if not r.ok]
    message = f"Found {len(rows)} test{'s' if len(rows) != 1 else ''}"
    if flagged:
        message += f", {len(flagged)} needing a look"
    message += ". Nothing has been created yet."

    return {
        "ok": True,
        "message": message,
        "patient": extracted.get("patient"),
        "source": extractor.name,
        "total": len(rows),
        "ready": len(ready),
        "flagged": len(flagged),
        "rows": [row.to_dict() for row in rows],
    }
