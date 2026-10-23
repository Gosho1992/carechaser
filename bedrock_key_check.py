"""Does a Bedrock API key actually get us a text model?

`bedrock_check.py` exercises CareChaser's two live Bedrock paths, but it
authenticates the ordinary way and starts with `sts.get_caller_identity()`,
which a bearer token cannot do. This is the smaller question asked first:
can this credential invoke any Claude model at all?

    setx AWS_BEARER_TOKEN_BEDROCK "<key>"     # new window afterwards
    python bedrock_key_check.py

Nothing is written, and each successful call costs a fraction of a cent —
the prompt is four words and the reply is capped at twenty tokens.

WHAT IT IS LOOKING FOR

Three different failures look identical from a distance, and the fix is
different for each:

    AccessDeniedException        the key is fine, the account is not
                                 approved for this model
    ValidationException          the model id is wrong for this account or
                                 region — usually means an inference profile
                                 id is required instead of the bare id
    ResourceNotFoundException    no such model in this region

So the script tries the bare model id and the cross-region inference
profile id for each model, and prints the exception class rather than
flattening everything to "it didn't work".
"""

from __future__ import annotations

import os
import sys

# Bare ids and their US inference-profile forms. Newer Claude models on
# Bedrock are only reachable through a profile, and calling the bare id
# returns a validation error that reads like the model does not exist.
CANDIDATES = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
]

PROMPT = "Reply with the single word: ok"

# Error codes whose meaning is already stated in the verdict, so printing
# the raw message under them would just be noise.
KNOWN = {
    "AccessDeniedException",          # not approved for this model
    "ValidationException",            # wrong id form, often needs a profile
    "ResourceNotFoundException",      # not in this region
    "UnrecognizedClientException",    # credential itself is not valid
}

EXPLAIN = {
    "AccessDeniedException": "account not approved for this model",
    "ValidationException": "id wrong for this account/region, or needs a profile",
    "ResourceNotFoundException": "no such model here — not a refusal",
    "UnrecognizedClientException": "the credential itself was rejected",
    "ThrottlingException": "rate limited; try again shortly",
}


def main() -> None:
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

    print(f"region: {region}")
    print(f"bearer token present: {'yes' if token else 'no'}")
    if not token:
        print("\nNo AWS_BEARER_TOKEN_BEDROCK set. If you meant to use ordinary")
        print("AWS credentials instead, that works too — boto3 will pick them up.")

    try:
        import boto3
    except ImportError:
        print("boto3 is not installed: pip install boto3")
        sys.exit(1)

    print(f"boto3: {boto3.__version__}")

    # The listing needs the control-plane client and broader permissions than
    # an API key usually carries, so a failure here is not fatal.
    print("\n1. WHAT THIS ACCOUNT CAN SEE")
    listed: list[str] = []
    try:
        control = boto3.client("bedrock", region_name=region)
        models = control.list_foundation_models()["modelSummaries"]
        text = sorted(
            m["modelId"] for m in models
            if "TEXT" in (m.get("outputModalities") or [])
        )
        print(f"   {len(text)} text models listed")
        for model_id in text:
            if model_id.startswith(("anthropic.", "us.anthropic.")):
                print(f"     {model_id}")
                listed.append(model_id)
        if not listed:
            print("     (no Anthropic models listed for this account/region)")
    except Exception as exc:
        print(f"   could not list: {type(exc).__name__}: {exc}")
        print("   (an API key often cannot call the control plane — not fatal)")

    # Try what the account actually reports before falling back to guesses.
    # Hardcoding model ids is how the first run of this script concluded
    # "no access" when every failure was really "no such model here".
    targets = listed or CANDIDATES
    if listed:
        print(f"\n   Trying the {len(listed)} Anthropic models this account lists,")
        print("   rather than a hardcoded guess.")

    print("\n2. WHAT THIS CREDENTIAL CAN ACTUALLY INVOKE")
    runtime = boto3.client("bedrock-runtime", region_name=region)
    working: list[str] = []
    codes: dict[str, str] = {}
    def attempt(model_id: str) -> tuple[bool, str, str]:
        try:
            reply = runtime.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": PROMPT}]}],
                inferenceConfig={"maxTokens": 20, "temperature": 0},
            )
            return True, reply["output"]["message"]["content"][0]["text"].strip(), ""
        except Exception as exc:
            # botocore raises ClientError for all of these; the code that
            # tells them apart is inside the response, not the class name.
            code = getattr(exc, "response", {}).get("Error", {}).get("Code") \
                or type(exc).__name__
            return False, str(exc), code

    for model_id in targets:
        ok, said, code = attempt(model_id)

        # Many current models are only reachable through a cross-region
        # inference profile; called by their bare id they report a validation
        # error that reads like the model is missing. One retry settles it.
        if not ok and code in {"ValidationException", "ResourceNotFoundException"} \
                and not model_id.startswith("us."):
            ok_profile, said_profile, code_profile = attempt("us." + model_id)
            if ok_profile:
                print(f"   WORKS   us.{model_id}  ->  {said_profile!r}   (needs profile id)")
                working.append("us." + model_id)
                continue
            code = f"{code} / {code_profile} via profile"

        if ok:
            print(f"   WORKS   {model_id}  ->  {said!r}")
            working.append(model_id)
        else:
            codes[model_id] = code.split(" /")[0]
            print(f"   {code:34} {model_id}")
            if codes[model_id] not in KNOWN:
                print(f"       {said.split(':')[-1].strip()[:90]}")

    print("\nVERDICT")
    seen = set(codes.values())
    if not working and seen == {"UnrecognizedClientException"}:
        print("   The credential itself was rejected on every model, which is")
        print("   not the same as being refused the models. Check the key was")
        print("   copied whole, that AWS_BEARER_TOKEN_BEDROCK is set in THIS")
        print("   window, and that the region matches where the key was made.")
        return
    if not working and seen == {"ValidationException"}:
        print("   Every call was a validation error rather than a refusal, so")
        print("   the credential works and the model ids are the problem — the")
        print("   account likely needs the inference-profile form of the id.")
        return
    if working:
        print(f"   {len(working)} model(s) usable. The best one to set:")
        print(f"       CARECHASER_MODEL_ID={working[0]}")
        print("   Next: run bedrock_check.py to exercise CareChaser's own paths.")
    else:
        # Say which failure actually happened. The first version of this
        # verdict announced "AccessDenied on everything" regardless of the
        # codes, and so reported a model-id mistake as an account-level
        # refusal — the wrong conclusion, stated confidently.
        print("   No text model answered. What came back:")
        for code in sorted(seen):
            count = sum(1 for value in codes.values() if value == code)
            print(f"     {count:>3}x {code} — {EXPLAIN.get(code, 'unexpected')}")
        if "AccessDeniedException" in seen:
            print("\n   AccessDenied is the account-level one: model access is")
            print("   granted per model in the Bedrock console under Model")
            print("   access, and a new API key cannot change it.")
        if seen <= {"ResourceNotFoundException", "ValidationException"}:
            print("\n   None of these is a refusal. The credential works and the")
            print("   ids are wrong — check section 1 for what this account")
            print("   really has, and try one of those.")
        print("\n   Either way the deterministic router already covers the demo,")
        print("   so this is a nice-to-have, not a blocker.")


if __name__ == "__main__":
    main()
