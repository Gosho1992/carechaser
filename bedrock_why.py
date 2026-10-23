"""Print the whole error, not a summary of it.

bedrock_key_check.py classifies failures by error code, which is useful for
a sweep and useless here: every Anthropic model returns ValidationException
and the code alone does not say why. The message does, and it is the only
thing worth reading now.

Two variables changed at once between the voice path that works and the
text path that does not — a different API (bidirectional streaming vs
Converse) and a different credential (an IAM user vs a Bedrock API key).
This changes them back one at a time.

    python bedrock_why.py

Reads AWS_BEARER_TOKEN_BEDROCK if set, and ordinary AWS credentials if
they are configured. Prints neither.
"""

from __future__ import annotations

import os

import boto3

REGION = os.environ.get("AWS_REGION") or "us-east-1"
PROMPT = [{"role": "user", "content": [{"text": "Reply with the single word: ok"}]}]

# A spread rather than a list: one dated id, one short alias, one explicitly
# profile-prefixed. If they fail differently, the shape of the id is the
# problem; if identically, it is not.
MODELS = [
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-sonnet-5",
]


def show(label: str, exc: Exception) -> None:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
    message = getattr(exc, "response", {}).get("Error", {}).get("Message", str(exc))
    print(f"   {label}")
    print(f"      {code}: {message}")


def try_converse(runtime, model_id: str) -> bool:
    try:
        reply = runtime.converse(
            modelId=model_id, messages=PROMPT,
            inferenceConfig={"maxTokens": 20, "temperature": 0},
        )
        said = reply["output"]["message"]["content"][0]["text"].strip()
        print(f"   WORKS  converse      {model_id}  ->  {said!r}")
        return True
    except Exception as exc:
        show(f"converse      {model_id}", exc)
        return False


def try_invoke(runtime, model_id: str) -> bool:
    """The older InvokeModel path, in case Converse is the thing refused."""
    import json
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
    })
    try:
        reply = runtime.invoke_model(modelId=model_id, body=body)
        payload = json.loads(reply["body"].read())
        print(f"   WORKS  invoke_model  {model_id}  ->  {str(payload)[:80]}")
        return True
    except Exception as exc:
        show(f"invoke_model  {model_id}", exc)
        return False


def main() -> None:
    print(f"region: {REGION}")
    has_key = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
    print(f"API key set: {'yes' if has_key else 'no'}")

    print("\n1. WHAT THE VALIDATION ERROR ACTUALLY SAYS")
    runtime = boto3.client("bedrock-runtime", region_name=REGION)
    for model_id in MODELS:
        try_converse(runtime, model_id)

    print("\n2. THE OLDER INVOKE PATH, SAME CREDENTIAL")
    try_invoke(runtime, MODELS[0])

    # Changing the credential back to the one Nova Sonic uses. A Bedrock API
    # key and an IAM user are not interchangeable: the key carries
    # AmazonBedrockLimitedAccess, the user carries whatever is attached to it.
    print("\n3. THE SAME CALL WITH ORDINARY IAM CREDENTIALS")
    if not has_key:
        print("   (no API key was set, so section 1 already used these)")
        return
    try:
        session = boto3.Session()
        creds = session.get_credentials()
        if creds is None:
            print("   No ordinary AWS credentials configured in this window.")
            print("   If `aws configure` has been run, open a new window without")
            print("   AWS_BEARER_TOKEN_BEDROCK set and run this again.")
            return
        sts = session.client("sts", region_name=REGION)
        print(f"   account: {sts.get_caller_identity()['Account']}")
        # A fresh client built from the session, with the bearer token taken
        # out of the environment so it cannot win over the IAM credentials.
        saved = os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        try:
            iam_runtime = session.client("bedrock-runtime", region_name=REGION)
            for model_id in MODELS[:2]:
                try_converse(iam_runtime, model_id)
        finally:
            if saved:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = saved
    except Exception as exc:
        print(f"   could not use IAM credentials: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
