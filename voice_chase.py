"""CareChaser voice chase — Nova Sonic on one side, the live MCP server on the other.

What it does, in order:

  1. asks the deployed CareChaser MCP server which case is stuck and what the
     chase message says                                    (MCP, over HTTPS)
  2. speaks that chase to Amazon Nova 2 Sonic and plays back the clinician's
     recorded reply                                        (Bedrock, bidirectional stream)
  3. decides from the clinician's own words whether that was an acknowledgement
                                                           (deterministic, no model)
  4. records the acknowledgement back through the MCP server, so the board and
     the audit trail update                                (MCP again)

The boundary is the same one the whole project rests on: Sonic carries the
voice, the deterministic engine decides what happened. A model never marks a
case reviewed. The keyword check in `classify()` does, from the transcript of
what the human actually said, and anything ambiguous is left alone.

Run from the Nova samples folder, with the sonic venv active:

    python voice_chase.py --case CC-001
    python voice_chase.py --case CC-001 --reply "not yet, I will look at it tonight"
    python voice_chase.py --case CC-001 --dry-run     # no write-back

Needs: AWS credentials in the environment, nova_sonic_simple.py beside this
file, and the CareChaser server reachable.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import wave
from pathlib import Path

MCP_URL = os.environ.get("CARECHASER_MCP_URL", "https://100-31-163-198.sslip.io/mcp")
HERE = Path(__file__).parent
REPLY_PCM = HERE / "clinician_reply.pcm"
REPLY_WAV = HERE / "sonic_reply.wav"

DEFAULT_REPLY = "Yes, I have reviewed that biopsy result and documented the next step."

# Words a clinician actually uses. Checked against what the human said, never
# against anything a model produced. Anything that matches neither list is
# left as "unclear" and the case is not touched.
ACK_PATTERNS = [
    r"\b(i have|i've|just) (reviewed|seen|looked at|signed off)",
    r"\b(reviewed|signed off|done|completed) (it|that|the result)?\b",
    r"\byes,? (i|it)('ve| have)? (reviewed|seen|done)",
    r"\bit'?s (been )?(reviewed|done|handled)",
]
DECLINE_PATTERNS = [
    r"\bnot yet\b",
    r"\bhaven'?t\b",
    r"\bhave not\b",
    r"\bwill (look|review|get to)\b",
    r"\blater\b",
    r"\btonight\b",
    r"\btomorrow\b",
    r"\bno\b",
]


# --------------------------------------------------------------------- MCP


async def mcp_call(tool: str, args: dict) -> dict:
    from mcp import Client

    async with Client(MCP_URL) as client:
        result = await client.call_tool(tool, args, read_timeout_seconds=30)
        payload = "".join(b.text for b in result.content if getattr(b, "text", None))
        return json.loads(payload) if payload else {}


async def fetch_case(case_id: str | None) -> tuple[dict, str]:
    """The case to chase, and the message CareChaser drafted for it."""
    if not case_id:
        board = await mcp_call("get_board_snapshot", {"limit": 5})
        if not board.get("cases"):
            raise SystemExit("Nothing is stuck. Advance the demo clock or create a case.")
        case_id = board["cases"][0]["case_id"]
        print(f"   picked the most urgent stuck case: {case_id}")

    status = await mcp_call("get_case_status", {"case_id": case_id, "include_history": False})
    if not status.get("found"):
        raise SystemExit(f"No case {case_id} on the server.")

    draft = await mcp_call("send_chase", {"case_id": case_id, "confirm": False})
    text = draft.get("text") or draft.get("message", "")
    return status, text


# ------------------------------------------------------------------ speech


def synthesize(text: str, path: Path) -> None:
    """Polly makes the clinician's side of the call: 16 kHz mono PCM, which is
    exactly the format Sonic expects, so no conversion step is needed.

    Called in-process rather than through `aws polly synthesize-speech`: passed
    on a Windows command line, a reply that opens with a word and a comma
    ("Yes, I have reviewed...") was parsed into the region, and the CLI tried to
    reach polly.yes.amazonaws.com. The client takes the region as an argument,
    so there is nothing left to misread.
    """
    import boto3

    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    polly = boto3.client("polly", region_name=region)
    spoken = polly.synthesize_speech(
        Text=text, OutputFormat="pcm", SampleRate="16000", VoiceId="Matthew"
    )
    path.write_bytes(spoken["AudioStream"].read())


async def talk_to_sonic(reply_text: str) -> tuple[list[str], int]:
    """Send the clinician's spoken reply to Sonic, collect transcripts and audio.

    Subclasses the sample client the same way save_reply.py does, because the
    stock one only plays audio and prints nothing.
    """
    sys.path.insert(0, str(HERE))
    from nova_sonic_simple import CHUNK_SIZE, SimpleNovaSonic

    transcripts: list[str] = []
    audio = bytearray()

    class Capturing(SimpleNovaSonic):
        async def _process_responses(self):
            try:
                while self.is_active:
                    output = await self.stream.await_output()
                    result = await output[1].receive()
                    if not (result.value and result.value.bytes_):
                        continue
                    event = json.loads(result.value.bytes_.decode("utf-8")).get("event", {})
                    if "audioOutput" in event:
                        audio.extend(base64.b64decode(event["audioOutput"]["content"]))
                    elif "textOutput" in event:
                        said = event["textOutput"]["content"]
                        transcripts.append(said)
                        print(f"   heard: {said.strip()}")
            except Exception as exc:          # a dropped stream must not lose the run
                print(f"   (stream ended: {exc!r})")

    synthesize(reply_text, REPLY_PCM)
    raw = REPLY_PCM.read_bytes() + b"\x00" * 32000      # trailing silence = "I'm done"

    client = Capturing()
    await client.start_session()
    await client.start_audio_input()

    step = CHUNK_SIZE * 2
    for index in range(0, len(raw), step):
        await client.send_audio_chunk(raw[index:index + step])
        await asyncio.sleep(CHUNK_SIZE / 16000.0)        # real time, or Sonic waits
    await client.end_audio_input()
    await asyncio.sleep(20)

    # Let the reader loop finish on its own rather than cancelling it. A cancel
    # lands mid-chunk and the CRT layer prints an InvalidStateError over the
    # top of the result, which looks like a failure and is not one.
    client.is_active = False
    await client.end_session()
    if client.response and not client.response.done():
        try:
            await asyncio.wait_for(asyncio.shield(client.response), timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            pass
    await asyncio.sleep(0.5)        # give the stream a moment to settle

    with wave.open(str(REPLY_WAV), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(bytes(audio))

    return transcripts, len(audio)


# ------------------------------------------------------------- the decision


def say_duration(hours: float) -> str:
    """Mirrors chase.py, so this script and the server never disagree on
    screen about how long a case has been waiting."""
    minutes = round(hours * 60)
    if minutes < 60:
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    whole = round(hours)
    return f"{whole} hour{'' if whole == 1 else 's'}"


def classify(transcripts: list[str]) -> tuple[str, str]:
    """Did the clinician say they had acted? Keywords only, on their words.

    Sonic's own reply is ignored: it is a conversational pleasantry, not
    evidence. Only the first transcript, which is what the human said, counts.
    """
    if not transcripts:
        return "unclear", ""
    said = transcripts[0].strip()
    lowered = said.lower()

    for pattern in DECLINE_PATTERNS:
        if re.search(pattern, lowered):
            return "declined", said
    for pattern in ACK_PATTERNS:
        if re.search(pattern, lowered):
            return "acknowledged", said
    return "unclear", said


# -------------------------------------------------------------------- main


async def run(case_id: str | None, reply_text: str, dry_run: bool) -> None:
    print("\n1. ASK CARECHASER  (MCP over HTTPS)")
    case, chase_text = await fetch_case(case_id)
    case_id = case["case_id"]
    print(f"   {case_id}: {case['test_type']} — {case['state']}")
    print(
        f"   {say_duration(case['hours_in_state'])} with {case['owner']}, "
        f"status {case['status']}"
    )
    print(f"\n   CareChaser would say:\n   \"{chase_text}\"")

    print("\n2. THE CALL  (Amazon Nova 2 Sonic, bidirectional stream)")
    print(f"   clinician says: \"{reply_text}\"")
    transcripts, audio_bytes = await talk_to_sonic(reply_text)
    print(f"   Sonic replied with {audio_bytes} bytes of audio -> {REPLY_WAV.name}")

    print("\n3. DECIDE  (deterministic, no model)")
    verdict, heard = classify(transcripts)
    print(f"   transcript: \"{heard}\"")
    print(f"   verdict:    {verdict}")

    print("\n4. RECORD  (MCP write-back)")
    if verdict != "acknowledged":
        print("   Nothing recorded. The clinician did not say they had acted,")
        print("   so the case stays open and keeps being chased.")
        return
    if dry_run:
        print("   Dry run: would mark the case reviewed, but nothing was written.")
        return

    result = await mcp_call("acknowledge_result", {
        "case_id": case_id,
        "actor_id": case["owner"],
        "actor_role": case["owner_role"],
        "event_type": "result_reviewed",
        "note": f"Acknowledged by voice: {heard}",
        "event_id": f"voice-{case_id}-{heard[:24]}",
    })
    print(f"   {result.get('message')}")

    board = await mcp_call("get_board_snapshot", {"limit": 50})
    still_stuck = [row["case_id"] for row in board["cases"]]
    print(f"\n   board now: {board['counts']['stuck']} stuck")
    print(f"   {case_id} still listed: {case_id in still_stuck}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=None, help="case id; default is the most urgent")
    parser.add_argument("--reply", default=DEFAULT_REPLY, help="what the clinician says")
    parser.add_argument("--dry-run", action="store_true", help="do not write back")
    args = parser.parse_args()

    print("CareChaser voice chase")
    print(f"server: {MCP_URL}")
    asyncio.run(run(args.case, args.reply, args.dry_run))


if __name__ == "__main__":
    main()
