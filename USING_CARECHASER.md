# Using CareChaser

A short guide to what this is, what you are looking at, and what to try.

---

## What CareChaser is for

In a hospital, a test result existing is not the same as somebody acting on it.
A biopsy result can sit for three days because the doctor who ordered it was in
theatre, nobody owned the result, and no system noticed.

CareChaser tracks every test from the moment it is ordered to the moment the
case is closed. It knows which stage each case is in, who it is waiting on, and
how long it has been there. When something sits too long, CareChaser says so,
offers to chase the right person, and remembers that it did — so tomorrow it can
tell you nobody acted.

**What it does not do.** CareChaser never interprets a result, never decides
anything is abnormal, and never suggests treatment. Urgency comes from the
clinical system. Every number it reports is calculated by a deterministic
engine, not written by a language model.

---

## Opening it

```
https://100-31-163-198.sslip.io
username: judge
password: (as supplied)
```

Chrome or Edge. The voice button needs a microphone; everything also works by
typing.

---

## What is on the screen

### Top bar

The hospital name, and three badges: the **demo user** you are acting as, a
reminder that the data is **synthetic**, and which **engine** is answering.

### Demo clock

Cases become overdue with time, and nobody wants to wait a day to see that.

- **+1 hour / +1 day / +3 days** move time forward. The board updates straight away.
- **Reset** puts the clock back to the start.

Time only moves forward. Reset is how you start over.

### The four counters

| Counter | Meaning |
|---|---|
| **Stuck** | Cases past the time limit for their current stage |
| **Critical stuck** | Of those, how many are marked critical |
| **Escalation due** | Cases where the next person up should now be told |
| **Open follow-ups** | Cases CareChaser already chased that still have not moved |

---

## Left column

### Needs attention

The work queue, worst first. Critical cases rank above routine ones, however
long the routine one has been waiting — a critical result 30 minutes past its
window matters more than a routine case three days late.

Each row shows the case ID, the test and patient, the stage, who owns it, how
long it has been there, and how far past the limit. An arrow (`Dr. Khan → Dr.
Riaz`) means escalation is due and names who it would go to.

**Click any row** to open it on the right.

### Read a report

This is where cases come from in real life: a document somebody already has.

1. Press **Load sample**, or paste your own report.
2. Press **Read report**.
3. A review table appears. **Nothing has been created yet** — the board does not move.

Each row shows the test, the stage CareChaser mapped it to, and what it removed.
Removed fragments appear struck through in red: result values, reference ranges,
and words like "elevated" are stripped out before anything enters the workflow.

In the table you can:

- **Edit the test name** if it came out wrong
- **Change the stage** from the dropdown
- **Set hours already waiting**, so a case can start overdue
- **Remove a row** with ×

A row CareChaser could not map is tinted red with the reason, and its stage
dropdown is empty. Choosing a stage fixes it, and the button count goes up.

4. Press **Create N cases**. Now they are on the board.

### Create a case

Quicker than a report when you just want one case. Choose a test, a stage, how
many hours it has already been waiting, and the urgency. A case created with 90
hours waiting against a 24-hour limit is overdue the moment it exists.

---

## Right column

### Case detail

Appears when you click a case. It opens with the verdict: **STUCK** or **ON
TIME**, the hours, and how far past the limit. Below that: stage, owner,
patient, and who escalation would go to.

Then the timeline, oldest first:

- **Created from a report** — where this case came from
- **Stage completed** — each step, how long it took, who did it
- **Escalated to / Chased** — every time CareChaser pushed on it
- **Now** — where it is at this moment

Two buttons: **Escalate** and **Mark reviewed**.

### Ask CareChaser

Type or hold the button and speak. Under each answer is the tool that produced
it, so you can see the number came from the engine.

Things to try:

| Say | What happens |
|---|---|
| "what is stuck today" | The queue, most urgent first |
| "what is the status of CC 001" | That case, and it opens on the right |
| "what is waiting on Dr. Khan" | Only what he personally owns |
| "escalate CC 001" | A confirmation card appears — nothing is recorded yet |
| "yes escalate CC 001" | Recorded, and the timeline updates |
| "chase CC 001" | Drafts a follow-up message for you to approve |
| "anything I should know" | What was chased before and still has not moved |
| "skip ahead a day" | Moves the demo clock |

Spoken case IDs are understood loosely: "CC 001", "cc-001" and "CC001" all work.

### Agent activity

A running list of what CareChaser just did: which tools ran, what was created,
when the clock moved.

---

## A five-minute tour

1. **Look at the board.** The critical potassium result is top, above a case
   that has been waiting 74 hours. Urgency outranks lateness.
2. **Click CC-001.** Read the timeline.
3. **Say "escalate CC-001".** A confirmation card appears showing who it would
   go to and how long it has been waiting. Nothing is recorded yet.
4. **Say "yes".** The timeline gains an escalation entry.
5. **Press +1 day.** A day passes. Nobody acted.
6. **Press Start over**, then ask **"anything I should know"**.
   CareChaser: *"You escalated CC-001 yesterday to Dr. Riaz. It is still in
   Result Released, now 98 hours."* New conversation, same memory.
7. **Say "mark CC-001 reviewed".** Ask again — nothing outstanding.
8. **Load the sample report and read it.** Watch the values being stripped, fix
   anything wrong, and press Create.

That is the whole product: the conversation ends, the workflow does not.

---

## If something looks wrong

| Problem | Fix |
|---|---|
| Board empty, nothing stuck | Press **+1 day**, or create a case |
| Page will not load | Use the full `https://` address |
| Voice button greyed out | Browser has no speech support; type instead |
| A case says it "starts later than the current demo time" | It was made while the clock was ahead; press +1 day |
| Want a clean slate | Press **Reset** on the clock |

---

## For the technically curious

```
Voice / browser
      ↓
MCP server (Streamable HTTP, protocol 2026-07-28)
      ↓
Deterministic engine   ← decides everything: stages, limits, who to chase
      ↓
Case store · follow-up log · audit history
      ↑
Amazon Bedrock        ← reads reports, writes chase messages. Decides nothing.
```

The split is the point. Bedrock reads a report into a list and phrases a
follow-up message. Everything that counts as a judgement — is this overdue, who
owns it, who gets told next — is code, with the rules in a configuration file
you can read.

Every drafted message is checked before it can be sent: if it contains a number
the engine did not supply, it is rejected and a deterministic template is sent
instead.
