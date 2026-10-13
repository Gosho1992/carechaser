# Friction log

Where CareChaser lost time, what the tool said, and what would have prevented
it. Written as it happened, not cleaned up afterwards. Roughly in the order
the time was lost.

Everything here is about AWS and the MCP Python SDK — the two toolchains the
project is built on.

---

## 1. Bedrock AgentCore: abandoned after four undeclared prerequisites and a zero quota

**What I was doing:** deploying the MCP server to Bedrock AgentCore Runtime,
which looked like the intended hosting path for an MCP server on AWS.

**What happened:** the prerequisites surfaced one at a time, each as a failure
rather than a check. First the AWS CLI was missing. Then `uv`, which the
documentation did not list. Then GNU `zip` — and after installing it through
winget, it was still not on `PATH`, so the same command failed a second time
with the same message.

Having cleared all four, the build ran, uploaded, and then failed:

```
ServiceQuotaExceededException: maxAgents limit exceeded
```

The account's `maxAgents` quota was **zero**. Nothing in the console, the
documentation or the CLI said so before the upload.

**What it cost:** most of a day, and the deployment target. The project moved
to plain EC2 with Caddy, which took about ninety minutes and has worked since.

**What would have prevented it:** a preflight command — `agentcore doctor` —
that checks the prerequisites *and* the account's `maxAgents` quota before
building anything. A quota of zero should be reported at `configure` time, not
after an upload. The prerequisite list in the documentation should also be
complete, and `uv` and `zip` should be on it.

---

## 2. "Operation not allowed" hid the real reason: the account was not authorised

**What I was doing:** calling Bedrock Converse with Claude and Nova text
models.

**What happened:** every text model returned the same thing:

```
ValidationException: Operation not allowed
```

That message describes an *operation* problem, so I spent time on the request:
wrong model id, wrong region, inference profile instead of model id, Converse
instead of InvokeModel. The Bedrock console Playground returned the same error,
which at least ruled out my code.

The actual cause only appeared from a command I had no reason to try:

```
aws bedrock get-foundation-model-availability --model-id ...
  → authorizationStatus: NOT_AUTHORIZED
```

Everything else in that response said `AVAILABLE`. A support case was filed on
5 October and had not been answered by 13 October.

**What it cost:** about two days across several sessions, and ultimately a
feature: `BedrockBrain` and `BedrockDrafter` are implemented and tested but
shipped disabled.

**What would have prevented it:** make the error say what is wrong.
`AccessDeniedException: this account is not authorized for model X` is a
one-line change and would have saved all of it. Related: the console's **Model
access** page has been retired, so the obvious place to check authorisation no
longer exists, and nothing replaced it that a first-time user would find.

---

## 3. `list-inference-profiles` returns models that are end-of-life

**What I was doing:** choosing a working model from what the account offered.

**What happened:** `aws bedrock list-inference-profiles` listed profiles
including `us.anthropic.claude-3-haiku` and `us.amazon.nova-premier`. Both
returned end-of-life errors when invoked. The listing gives no lifecycle
status, so a reasonable "pick one from the list" loop walks straight into dead
models.

**What would have prevented it:** a `lifecycleStatus` field on each profile, or
simply not listing retired models.

---

## 4. `amazon.nova-sonic-v1:0` fails silently; the working id is `amazon.nova-2-sonic-v1:0`

**What I was doing:** the first bidirectional-stream call to Nova Sonic.

**What happened:** with `amazon.nova-sonic-v1:0` the stream opened, audio was
accepted, and **nothing came back**. No error, no exception, no empty-response
event. The session simply sat there until the timeout.

Switching the model id to `amazon.nova-2-sonic-v1:0` made it work first time.

**What it cost:** a long debugging session aimed at the audio pipeline —
sample rate, chunk size, encoding — when the audio was fine all along.

**What would have prevented it:** a validation error on an unsupported or
retired model id. Silence is the worst possible failure mode for a streaming
API, because every other suspect is cheaper to test than the model id.

---

## 5. Nova Sonic needs real-time pacing and trailing silence, and says so nowhere

**What happened:** sending the audio as fast as the socket accepted it produced
no response. Two undocumented requirements:

- chunks must be sent at roughly real time — `sleep(CHUNK_SIZE / 16000)`
  between them
- the stream needs trailing silence (about 32,000 bytes of zeroes) before Sonic
  treats the turn as finished

Neither is obvious from an API whose whole point is streaming, and both produce
the same symptom as problem 4: nothing happens.

**What would have prevented it:** one paragraph in the Nova Sonic docs, or an
explicit end-of-turn event that does not depend on silence.

---

## 6. `aws-sdk-bedrock-runtime` is labelled "evaluation and testing only"

The only Python SDK that supports `InvokeModelWithBidirectionalStream` carries
an AWS notice saying it is for evaluation and testing, not production. There is
no production path in Python for Nova Sonic that I could find. That is a
reasonable thing to be told up front rather than discovered after building on
it.

---

## 7. The AWS CLI parsed part of my text as the region

**What I was doing:** calling Polly from a Python subprocess on Windows:

```
aws polly synthesize-speech --output-format pcm --sample-rate 16000 \
  --voice-id Matthew --text "Yes, I have reviewed that biopsy result." out.pcm
```

**What happened:**

```
Could not connect to the endpoint URL: "https://polly.yes.amazonaws.com/v1/speech"
```

The region had become **`yes`** — the first word of the `--text` value, up to
the comma. `AWS_DEFAULT_REGION` was correctly set to `us-east-1`.

**Fix:** stop using the CLI. `boto3.client('polly', region_name=...)` takes the
region as an argument, so there is nothing left to misparse. That is the right
answer anyway, but the failure mode is alarming: a value I passed as text
silently became configuration.

**What would have prevented it:** the CLI refusing to treat an argument value
as a region, or an error naming which argument supplied `yes`.

---

## 8. Bedrock permissions do not include Polly

Obvious in hindsight, costly in the moment. An IAM user with
`AmazonBedrockFullAccess` running a Bedrock sample that also uses Polly fails
at the first speech call:

```
AccessDeniedException: ... not authorized to perform: polly:SynthesizeSpeech
```

The AWS speech-to-speech samples use Polly without saying that a second policy
is needed. One line in the sample's README would fix it.

---

## 9. MCP Python SDK 2.x renamed the things every tutorial uses

Every example I could find online was written against 1.x. In 2.2:

| Was | Now |
| --- | --- |
| `FastMCP` | `MCPServer` |
| `streamablehttp_client` + `ClientSession` | `Client` |
| `Tool.inputSchema` | `Tool.input_schema` |

None of these fail at import with a message that names the replacement, so the
first hour is spent assuming your installation is broken rather than that the
tutorial is old.

**What would have prevented it:** deprecation shims that raise with the new
name in the message, and a dated "if you are reading a 1.x example" note at the
top of the docs.

---

## 10. MCP's DNS-rebinding protection ships on, with an empty allow-list

**What happened:** the server worked on `127.0.0.1` and returned
`421 Misdirected Request` the moment it sat behind Caddy — or any tunnel, or
any proxy.

The cause is `TransportSecuritySettings`: DNS-rebinding protection is enabled
by default and `allowed_hosts` defaults to empty, so every `Host` header that
is not localhost is rejected. The 421 status gives no hint that this is a
configuration choice rather than a routing problem.

**Fix:** an explicit allow-list, wired to an environment variable
(`CARECHASER_ALLOWED_HOSTS`). Verified afterwards that the real hostname
returns 200 while `evil.example.com` still returns 421 — the protection is
doing its job, it just needed telling what the server is called.

**What would have prevented it:** a log line at startup —
*"DNS rebinding protection is on and no hosts are allowed; requests with a Host
header other than localhost will be rejected"* — and a 421 body saying so.

---

## 11. Testing on Windows: a blocked pipe looked like a performance problem

Not an AWS or MCP issue, but the environment made it expensive, so it belongs
here.

The MCP server fixture in the test suite started `uvicorn` with
`stdout=subprocess.PIPE` and never read it. Around the forty-fifth request the
pipe buffer filled, the server blocked on its own logging, and the suite
degraded into read timeouts: **22 failures in 6,612 seconds** — one hour fifty
minutes — where the same suite takes fourteen seconds now.

The symptom looks like a slow machine or a flaky network, which is where the
first hour of investigation went.

**Fix:** redirect the subprocess output to a file and set an explicit
`read_timeout_seconds`.

**Worth knowing:** this is a general trap with any long-running subprocess in a
test fixture, and nothing in the error output points at the pipe.

---

## What went well

- **Caddy.** Automatic TLS worked first time, including the certificate for an
  `sslip.io` hostname built from the Elastic IP. No domain purchase, no
  certbot, no renewal cron.
- **sslip.io.** Having a real hostname for a bare IP removed the whole class of
  self-signed-certificate problems.
- **Nova 2 Sonic itself**, once the model id was right. Transcription of
  clinical phrasing was accurate, latency was workable, and the audio format
  lines up exactly with Polly's `pcm` output at 16 kHz — the two services fit
  together with no conversion step.
- **The MCP Streamable HTTP transport**, once the host allow-list was set,
  has not needed attention since.
