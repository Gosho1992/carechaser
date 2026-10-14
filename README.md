# CareChaser

**A self-hosted MCP server that chases diagnostic results until somebody acts on them.**

A lab result that nobody reviews has the same outcome for the patient as a
result that was never produced. CareChaser tracks each diagnostic test through
seven workflow stages, notices when one stops moving, names the person who owes
the next step, drafts the follow-up, and escalates when nobody acts.

The conversation ends. The workflow doesn't — state lives in files on the
server, so a case chased on Monday is still being chased on Wednesday.

---

## Try it

| | |
| --- | --- |
| Live app | `https://100-31-163-198.sslip.io` |
| MCP endpoint | `https://100-31-163-198.sslip.io/mcp` |
| Credentials | in the Devpost submission notes |

Nothing to install. All data is synthetic.

Fastest way to see that the numbers are real: use **Create a case**, set
*Hours already waiting* above the stage limit, and watch the engine judge a
case that did not exist thirty seconds ago.

---

## The one rule

**CareChaser never carries a clinical value.** It does not store `6.8 mmol/L`,
does not interpret a result, does not diagnose, and does not suggest treatment.
It tracks whether the workflow moved and who owes the next step.

That rule is enforced in three places, each with its own tests:

- `workflow.yaml` states it as an explicit can / must-not list
- `intake.py` strips every measurement, unit and judgement word out of a pasted
  report before a case is created, and shows back what it removed
- `chase.py::check_draft` rejects any outgoing message containing clinical
  language or a number the engine did not supply

---

## The MCP server

`carechaser_mcp.py` is a **self-hosted MCP server**, served over
**Streamable HTTP**, built on the MCP Python SDK 2.2. It negotiates protocol
version **2026-07-28**.

Sixteen tools, grouped:

- **Reading** — `get_stuck_workflows`, `get_case_status`, `get_my_pending`,
  `get_followups`, `get_board_snapshot`, `get_case_timeline`
- **Acting** — `acknowledge_result`, `escalate`, `send_chase`, `create_case`
- **Intake** — `read_report`, `confirm_report_rows`
- **Reading the history** — `get_insights`
- **Demo clock** — `get_demo_clock`, `advance_demo_clock`, `reset_demo_clock`

This is the only way in. The web page does not read the case files and neither
does the voice script — both call MCP tools, and the tools call the engine. A
second code path is the usual way a dashboard and an assistant come to disagree
about the same case, and there isn't one here.

`escalate` and `send_chase` take `confirm`. Called with `confirm` false they
name who *would* be contacted and write nothing. A voice interface that
mishears a case id gets one chance to be corrected before anyone is contacted.

---

## AWS

| Service | Used for |
| --- | --- |
| **Amazon Bedrock** — `amazon.nova-2-sonic-v1:0` | Speech understanding and spoken reply in the voice chase, over `InvokeModelWithBidirectionalStream` |
| **Amazon Polly** | Renders the clinician's side of the call as 16 kHz mono PCM, exactly Sonic's input format |
| **Amazon Textract** | Reads scanned ward printouts — image PDFs with no text layer — so a report can be dropped in as it comes off the scanner |
| **Amazon EC2** | `t3.micro`, Ubuntu 24.04, Elastic IP, Caddy for TLS and basic auth. Textract is reached through an IAM instance role, so no AWS keys live on the server |

`voice_chase.py` closes the loop by voice: it asks the MCP server which case is
stuck, speaks the chase to Nova 2 Sonic, hears the clinician's reply, decides
deterministically whether that was an acknowledgement, and writes back through
MCP only if it was.

Run the refusal first — it is the better demonstration:

```bash
python voice_chase.py --case CC-001 --reply "not yet, I will look at it tonight"
```

The write-back is armed and nothing is recorded. Sonic replies warmly; the
engine is unmoved. A fluent response is not proof that an action occurred.

---

## Verification

```bash
python -m pytest -q
```

**287 tests**, passing on Windows and Linux. They cover the engine, the case
store, follow-ups, the chase layer and its guardrail, the demo clock, report
intake, scanned-document reading, the ward simulator, the insights layer, the
agent, and the MCP server end to end over real HTTP.

Worth running rather than taking on trust. A few that make the point:

| Test | Proves |
| --- | --- |
| `test_invented_number_is_rejected` | A model cannot put a figure the engine did not supply into a message |
| `test_a_critical_breach_is_never_reported_as_zero` | A ten-minute limit breached by fifty minutes reads as fifty minutes, not "0 hours" |
| `test_real_test_names_with_numbers_survive` | `Potassium 6.8` becomes `Potassium`, while `Vitamin B12`, `HbA1c` and `COVID-19 PCR` survive intact |
| `test_a_created_case_carries_no_result_value_anywhere` | No clinical value reaches the case, its history, or the audit log |
| `test_create_case_rejects_a_terminal_stage` | A case cannot be created somewhere it could never be stuck |
| `test_board_snapshot_is_internally_consistent` | The counters cannot disagree with the queue beneath them |
| `test_clock_will_not_rewind` | The demo clock is forward-only, so a case cannot get younger |

---

## Seven days of operation

CareChaser has been running continuously since **14 October 2026**, on real
elapsed time, posting a batch of scanned reports each morning and letting a
synthetic ward clear them through the day.

| | |
| --- | --- |
| `ward_sim.py generate` | writes the day's reports as scanned PDFs, 07:30 local |
| `daily_ingest.py` | Textract reads them, intake strips every value, clean rows become cases |
| `ward_sim.py tick` | every 30 minutes, each clinician may clear work they own |
| `insights.py` | reads the audit trail back |

Two systemd timers run it; `systemd/` in this repo holds all six unit files.

**The clock is locked.** `CARECHASER_CLOCK_LOCKED=1` closes the demo controls
while the run is accumulating evidence: one press of "+1 day" would date every
later entry into a future that never happened, and nothing afterwards could
tell which timestamps were real.

### What is simulated, and what is not

The reports are generated and the clinicians are scripted. The arrival volume,
the clearing hours and the weekday pattern come from per-person profiles in
`ward_sim.py`, written to `ward_profiles.json`.

Everything downstream is real: real Textract calls on real image PDFs, real
MCP tool calls, real timestamps, real SLA evaluation, a real append-only audit
trail accumulating over real days.

**`insights.py` never reads those profiles.** It recovers the pattern from the
audit trail alone, exactly as it would on a ward where no such file exists, and
a test fails if the word `ward_profiles`, `ward_sim` or `PROFILES` appears
anywhere in it. The Insights panel can therefore be checked against the
generator's actual parameters: the system is told nothing and recovers the
working hours anyway.

That is the claim worth testing, and it is the reason the run exists rather
than a seeded fixture.

## Running it locally

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
python seed_demo.py

python carechaser_mcp.py      # MCP server on :8000
python carechaser_web.py      # web client on :8080, in a second terminal
```

Then open `http://127.0.0.1:8080`.

For the voice chase you also need `aws-sdk-bedrock-runtime[awscrt]`, AWS
credentials in the environment, and an IAM user carrying **both**
`AmazonBedrockFullAccess` and `AmazonPollyFullAccess`.

---

## Repository map

| File | What it is |
| --- | --- |
| `workflow.yaml` | The rules: seven stages, owners, time limits, escalation targets, roster, boundaries, and the fourteen sources the model was generalised from |
| `carechaser_engine.py` | State machine, SLA evaluation, legal transitions, append-only history |
| `case_store.py` | JSON-backed store with atomic writes |
| `carechaser_mcp.py` | The MCP server and its sixteen tools |
| `carechaser_web.py` + `static/` | FastAPI web client and the command-centre UI |
| `carechaser_agent.py` | Agent layer: deterministic router, plus a Bedrock Converse path behind a flag |
| `chase.py` | Drafting, the guardrail, and delivery channels |
| `followups.py` | Append-only follow-up log — the memory behind the cross-session briefing |
| `intake.py` | Report → rows, with every clinical value stripped |
| `scan_intake.py` | Scanned PDFs through Textract, columns rejoined by position on the page |
| `insights.py` | Reads the audit trail back: where time goes, when people work, which day backs up |
| `ward_sim.py` | The synthetic ward, and the profiles the insights must recover without seeing |
| `daily_ingest.py` | The morning job: reports in, tracked cases out |
| `systemd/` | The six unit files the deployment runs on |
| `democlock.py` | Simulated clock shared across processes |
| `voice_chase.py` | The Nova 2 Sonic voice loop |
| `seed_demo.py` | Nine-case synthetic ward |
| `USING_CARECHASER.md` | Screen-by-screen user guide |
| `FRICTION.md` | Friction log: what cost time, and what would have prevented it |

---

## Where AI is used, and where it deliberately is not

| Job | What runs |
| --- | --- |
| Voice understanding and speech | Amazon Nova 2 Sonic, on Bedrock |
| Spoken audio for the chase | Amazon Polly |
| Did the clinician acknowledge? | Deterministic keyword check |
| Is a case overdue? | Engine, timestamps against `workflow.yaml` |
| Who is the escalation target? | Roster lookup — the engine never invents a person |
| Is this transition legal? | State machine |
| Reading a scanned page | **Amazon Textract** — OCR is a model's job, and it only turns pixels into characters |
| Report extraction | Regex, so it can be proven that no clinical value survives |
| Where time goes, who works when | Counted from the audit trail. Nothing predicted, nothing fitted |

**Honest caveat:** the Bedrock *text* path (`BedrockBrain`, `BedrockDrafter`)
is implemented and tested but disabled in the deployment. The AWS account
returns `authorizationStatus: NOT_AUTHORIZED` for every text model; a support
case went unanswered. Nova 2 Sonic is the Bedrock model that runs. The "Ask
CareChaser" box therefore uses the deterministic router, with
`CARECHASER_FORCE_OFFLINE=1` set on the service.

---

## Limitations

- The workflow model is generic, synthesised from published guidance. The time
  limits are engineering defaults, not clinical standards, and a real
  deployment replaces them.
- Storage is JSON files, not a database. Fine for one ward, not for a hospital.
- Delivery is a log file by default; SES and webhook channels exist but are not
  wired to a real recipient.
- `voice_chase.py` runs from a terminal and is not part of the web app.
- CareChaser is not a medical device.

---

## Licence

MIT. See [LICENSE](LICENSE).
