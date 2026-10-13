"""Exercise the two live Bedrock paths and show exactly what came back.

Everything in the test suite runs against the deterministic fallbacks, so
these two branches have never actually executed. They are the ones that run
on the deployed server, which makes this the gap worth closing before a demo.

    python bedrock_check.py

Needs AWS credentials and Bedrock model access in your region. It makes two
model calls (a few cents at most) and writes nothing to the case store.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

from carechaser_engine import load_workflow

REPORT = """Musafah Surgical Hospital
Patient MK-4471

CBC — completed
Liver function — completed
Serum potassium 6.8 mmol/L — result available
Vitamin B12 — not completed
Chest X-Ray — not completed
"""


def rule(title: str) -> None:
    print("\n" + title)
    print("─" * len(title))


def check_extractor(workflow) -> bool:
    from intake import BedrockExtractor, extract_report, validate

    rule("1. EXTRACTOR — can Bedrock read a report?")
    try:
        extractor = BedrockExtractor()
        print(f"   model: {extractor.model_id}")
    except Exception as exc:
        print(f"   FAILED to create: {exc}")
        return False

    try:
        raw = extractor.extract(REPORT)
    except Exception as exc:
        print(f"   FAILED to call: {type(exc).__name__}: {exc}")
        print("\n   Most likely: no model access in this region, or the model id")
        print("   in CARECHASER_MODEL_ID is not one your account can use.")
        return False

    print("\n   raw model output:")
    print("   " + json.dumps(raw, indent=2).replace("\n", "\n   "))

    rows = validate(raw, workflow)
    print(f"\n   after validation: {len(rows)} rows")
    ok = True
    for row in rows:
        mark = "ok  " if row.ok else "FLAG"
        removed = (" | removed: " + ", ".join(row.removed)) if row.removed else ""
        print(f"   {mark} {row.test_name:22} -> {row.state or '—'}{removed}")

    blob = json.dumps([r.to_dict() for r in rows])
    for value in ["6.8", "mmol"]:
        in_fields = any(
            value.lower() in (r.test_name + r.status_text).lower() for r in rows
        )
        if in_fields:
            print(f"\n   PROBLEM: '{value}' survived into a field that becomes a case.")
            ok = False
    if ok:
        print("\n   No result value reached a case field. Boundary held.")

    full = extract_report(REPORT, workflow, extractor)
    print(f"   read_report would return ok={full['ok']}: {full['message']}")
    return ok and full["ok"]


def check_drafter() -> bool:
    from chase import BedrockDrafter, ChaseFacts, Chaser, LogChannel, check_draft

    rule("2. DRAFTER — can Bedrock write a chase message?")
    facts = ChaseFacts(
        case_id="CC-001",
        test_type="Histopathology (biopsy)",
        patient="MK-4471",
        state="Result Released / Awaiting Clinician Review",
        hours_in_state=74.0,
        sla_hours=24.0,
        owner="Dr. Khan",
        chase_target="Dr. Riaz",
        urgency="routine",
        escalated=True,
    )
    try:
        drafter = BedrockDrafter()
        print(f"   model: {drafter.model_id}")
        text = drafter.draft(facts)
    except Exception as exc:
        print(f"   FAILED: {type(exc).__name__}: {exc}")
        return False

    print("\n   model wrote:")
    print(f"   {text}")

    reason = check_draft(text, facts)
    if reason:
        print(f"\n   GUARDRAIL REJECTED IT: {reason}")
        print("   The deterministic template would be sent instead, so nothing")
        print("   breaks — but the model's wording needs a look.")
    else:
        print("\n   Guardrail accepted it: no invented numbers, no clinical language.")

    chaser = Chaser(drafter=drafter, channel=LogChannel(os.devnull))
    draft = chaser.compose(facts)
    print(f"   Chaser would send the {draft.source} version.")
    return True


def main() -> None:
    try:
        import boto3

        identity = boto3.client("sts").get_caller_identity()
        print(f"AWS account: {identity['Account']}")
        print(f"region:      {os.environ.get('AWS_REGION', 'us-east-1')}")
    except Exception as exc:
        print(f"No usable AWS credentials: {exc}")
        sys.exit(1)

    workflow = load_workflow("workflow.yaml")
    results = []
    for check in (lambda: check_extractor(workflow), check_drafter):
        try:
            results.append(check())
        except Exception:
            traceback.print_exc()
            results.append(False)

    rule("VERDICT")
    print(f"   extractor: {'works' if results[0] else 'needs attention'}")
    print(f"   drafter:   {'works' if results[1] else 'needs attention'}")
    if all(results):
        print("\n   Both live paths work. The deployed server can use Bedrock.")
    else:
        print("\n   At least one path failed. The deterministic fallbacks still")
        print("   work, so the demo is safe either way — but say so honestly")
        print("   rather than claiming Bedrock is in use.")


if __name__ == "__main__":
    main()
