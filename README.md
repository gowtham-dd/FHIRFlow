# 🏥 Healthcare Claims Multi-Agent System

An end-to-end AI-powered pipeline for automating CPT/HCPCS medical claim validation, built with **LangGraph**, **Groq Llama 3.1**, **Deepgram**, **Pinecone**, and **FHIR**. The system detects policy changes, updates patient records, validates claims against live payer policies, routes approved claims for EDI submission, and conducts real voice calls with patients for rejected claims.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Running the System](#running-the-system)
- [Dashboard UI](#dashboard-ui)
- [API Reference](#api-reference)
- [Data Formats](#data-formats)
- [How the Voice Agent Works](#how-the-voice-agent-works)

---

## Overview

When a healthcare provider submits a claim, this system:

1. **Checks** whether any payer policies have changed that affect the claim
2. **Updates** the FHIR procedure database with the latest policy flags
3. **Validates** the claim using a Groq LLM that reasons against real Pinecone policy vectors
4. **Routes** approved claims to EDI 837 generation for payer submission
5. **Calls the patient** via a real AI voice call (Deepgram TTS + STT + Groq LLM) for rejected claims, offering a correction

All agents are orchestrated by LangGraph. A Flask dashboard provides real-time workflow visibility — each agent's status updates live as the pipeline runs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Flask Dashboard (app.py)                  │
│              Upload JSON  →  Trigger Workflow                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Workflow (langgraph_workflow.py)       │
│                                                             │
│   Agent 2 → Agent 3 → Agent 4 ──APPROVED──▶ Agent 5A       │
│                             │                               │
│                          REJECTED                           │
│                             │                               │
│                             ▼                               │
│                          Agent 5B ──YES──▶ Agent 5A         │
│                             │                               │
│                            NO                               │
│                             ▼                               │
│                      Escalation Ticket                      │
└─────────────────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
    Pinecone DB    FHIR SQLite    SQLite Claims DB
   (policy vecs)  (procedures)   (dashboard data)
```

---

## Agent Pipeline

### Agent 2 — Policy Change Detector
**File:** `agents/agent2/agent2.py`

Compares newly downloaded policy PDFs against existing Pinecone vectors to detect changes in payer coverage rules. Uses Groq Llama 3.1 to extract structured diffs (prior auth requirements, dose limits, CPT/HCPCS code changes) and writes them to a timestamped JSON for Agent 3 to consume.

- **Input:** Policy PDF files from `agent1` downloads
- **Output:** `policy_changes_<timestamp>.json`
- **Models:** Groq `llama-3.1-8b-instant`, Pinecone vector similarity
- **Key logic:** Retrieves old policy text from Pinecone, sends old+new to LLM for structured diff, re-vectorises the new policy

### Agent 3 — FHIR Batch Updater
**File:** `agents/agent3/agent3.py`

Applies policy changes from Agent 2 to the FHIR procedure database. Flags procedures that are now non-compliant, updates `statusReason` fields, and writes a full audit log. Follows the FHIR R4 Procedure resource schema exactly.

- **Input:** `policy_changes_*.json` from Agent 2
- **Output:** Updated `fhir_procedure_db.sqlite` + audit log entries
- **Schema tables:** `procedures`, `procedure_performers`, `procedure_policy_flags`, `audit_log`
- **Key logic:** Batch updates in groups of 100, flags affected procedures, records every change to audit_log

### Agent 4 — Claims Validator
**File:** `agents/agent4/agent4.py`

The core decision engine. Fetches the patient's procedure history from FHIR, retrieves relevant policy documents from Pinecone via semantic search, and sends everything to Groq Llama 3.1 for a structured APPROVED/REJECTED decision with reasoning, confidence, and suggested CPT/HCPCS alternatives.

- **Input:** Claim JSON (`patient_id`, `code`, `dose`, `units`, `date_of_service`, `diagnosis`)
- **Output:** `{ decision, confidence, reasoning, policy_citations, suggested_alternatives, next_steps }`
- **Models:** Groq `llama-3.1-8b-instant`, Pinecone cosine similarity, HuggingFace `all-MiniLM-L6-v2` embeddings
- **Key logic:** Prompt includes patient history, procedure frequency, policy text, and explicit reasoning chain

### Agent 5A — Approval Router
**File:** `agents/agent5/agent5a.py`

Processes approved claims. Converts the validated claim to **EDI 837 format**, generates a submission file, and records the payer submission in the database.

- **Input:** Approved decision dict from Agent 4
- **Output:** EDI 837 `.edi` file + `claim_submissions` DB record
- **Key logic:** Generates ISA/GS/ST/BHT/NM1/CLM/SV1 EDI segments, tracks submission status and payer acknowledgements

### Agent 5B — Voice Agent
**File:** `agents/agent5/agent5b.py`

For rejected claims, Agent 5B places a real AI voice call to the patient. It speaks the rejection reason using **Deepgram TTS** (Aura Asteria model), listens to the patient's response via **Deepgram Nova-2 STT** over a live WebSocket, and classifies the spoken YES/NO using **Groq LLM**. If the patient agrees, the claim is corrected and re-submitted through Agent 5A. If they refuse, an escalation ticket is created.

- **Input:** Rejected decision + patient contact details
- **Output:** Patient intent (YES/NO) → correction or escalation ticket
- **Models:** Deepgram `nova-2` STT, Deepgram `aura-asteria-en` TTS, Groq `llama-3.1-8b-instant` intent classification
- **Key logic:** Runs in an isolated thread with its own asyncio event loop to avoid deadlocking LangGraph's outer loop. 8-second silence timer defaults to YES if no speech detected.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent orchestration | LangGraph (StateGraph + MemorySaver) |
| LLM inference | Groq API — Llama 3.1 8B Instant |
| Vector database | Pinecone |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Speech-to-text | Deepgram Nova-2 (WebSocket streaming) |
| Text-to-speech | Deepgram Aura Asteria (WebSocket streaming) |
| Audio I/O | PyAudio |
| Patient records | FHIR R4 Procedure schema (SQLite) |
| Backend API | Flask |
| Frontend | HTML + Tailwind CSS |
| Claim format | EDI 837 |

---

## Project Structure

```
.
├── app.py                        # Flask backend + REST API
├── langgraph_workflow.py         # LangGraph orchestration of all agents
├── data/
│   └── claims.db                 # SQLite: claims, workflows, tickets, patients
├── uploads/                      # Temporary uploaded JSON files
├── agents/
│   ├── agent2/
│   │   └── agent2.py             # Policy Change Detector
│   ├── agent3/
│   │   ├── agent3.py             # FHIR Batch Updater
│   │   └── fhir_procedure_db.sqlite
│   ├── agent4/
│   │   └── agent4.py             # Claims Validator
│   └── agent5/
│       ├── agent5a.py            # Approval Router (EDI 837)
│       └── agent5b.py            # Voice Agent (Deepgram)
├── helper/
│   └── helper.py                 # load_pdf_files(), download_embeddings()
├── templates/
│   └── index.html                # Dashboard UI
└── .env                          # API keys (not committed)
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- `portaudio` system library (required for PyAudio)

```bash
# macOS
brew install portaudio

# Ubuntu / Debian
sudo apt-get install portaudio19-dev
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

A complete `requirements.txt` should include:

```
flask
python-dotenv
langgraph
langchain-groq
langchain-pinecone
langchain-huggingface
langchain-text-splitters
langchain-core
sentence-transformers
pinecone-client
deepgram-sdk
websockets
pyaudio
pandas
requests
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Groq — LLM inference for all agents
GROQ_API_KEY=gsk_...

# Pinecone — policy vector database
PINECONE_API_KEY=pcsk_...

# Deepgram — STT (Agent 5B microphone input)
DEEPGRAM_STT_KEY=...

# Deepgram — TTS (Agent 5B voice output)
DEEPGRAM_TTS_KEY=...

# Flask
SECRET_KEY=your-secret-key-here

# Optional: override Flask base URL for voice-response polling
# FLASK_BASE_URL=http://127.0.0.1:5000
```

> **Note:** `DEEPGRAM_STT_KEY` and `DEEPGRAM_TTS_KEY` can be the same key if your Deepgram project has both STT and TTS enabled.

The Pinecone index must be named `fhirdb` (or update `pinecone_index_name` in the agent constructors). It should be pre-populated with your payer policy documents using `download_embeddings()` from `helper/helper.py`.

---

## Running the System

### 1. Start the Flask dashboard

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

### 2. Submit a claim

Upload a JSON file via the dashboard, or use the **Sample** / **Batch** buttons to run with built-in test data.

**Single claim format:**
```json
{
  "id": "CLM001",
  "patient_id": "pat001",
  "code": "J3420",
  "description": "Vitamin B12 injection",
  "dose": 1500,
  "units": 1,
  "date_of_service": "2026-03-15",
  "diagnosis": "E53.8"
}
```

**Batch format:** wrap multiple claim objects in a JSON array `[{...}, {...}]`.

### 3. Run Agent 5B standalone (voice only)

To test the voice agent in isolation against a pre-existing Agent 4 results file:

```bash
cd agents/agent5
python agent5b.py
```

It reads `../agent4/agent4_results_<timestamp>.json`, filters for rejected claims, and calls each one.

---

## Dashboard UI

The dashboard shows live workflow progress as each agent runs:

| Element | Description |
|---------|-------------|
| Agent pipeline icons | Turn blue (active) → green (complete) → red (error) |
| Progress bar | Advances as each agent completes its step |
| Agent log | Live streaming terminal showing each agent's messages |
| Voice interface | Appears automatically when Agent 5B activates — shows waveform, transcript, and Yes/No buttons as a manual fallback |
| Recent Claims table | Updates with APPROVED / REJECTED badge after Agent 4 decides |
| Stats counters | Total, Approved, Rejected, Open Tickets — refresh every 30 seconds |

The overlay can be closed and reopened without interrupting the running workflow.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a claim JSON and start the workflow |
| `GET` | `/api/claims` | Get last 50 processed claims |
| `GET` | `/api/stats` | Get dashboard counts (total, approved, rejected, tickets) |
| `GET` | `/api/workflows` | Get last 20 workflow summaries |
| `GET` | `/api/workflow/<id>` | Get full state for a workflow including final result |
| `GET` | `/api/workflow/<id>/messages` | Get live agent messages for polling (called every 1s by UI) |
| `POST` | `/api/workflow/<id>/voice-response` | Submit patient's YES/NO response from UI buttons |
| `GET` | `/api/tickets` | Get open escalation tickets |
| `GET` | `/api/policy-updates` | Get recent policy changes detected by Agent 2 |

### Voice response payload

```json
{ "response": "YES" }
```

or

```json
{ "response": "NO" }
```

---

## Data Formats

### Agent 4 decision output

```json
{
  "decision": "REJECTED",
  "confidence": "high",
  "reasoning": "J3420 doses exceeding 1000mcg require prior authorization per UHC policy effective 2026-01-01. Patient has no prior auth on file.",
  "policy_citations": ["UHC-J3420-2026", "CMS-B12-LCD"],
  "suggested_alternatives": ["G0008", "J3421"],
  "next_steps": "Submit prior authorization request or rebill with G0008",
  "claim_id": "CLM001",
  "patient_id": "pat001",
  "procedure_code": "J3420"
}
```

### Workflow message (UI polling)

```json
{
  "agent": "Agent 4",
  "message": "❌ Claim REJECTED — J3420 doses exceeding 1000mcg require prior auth…",
  "status": "complete",
  "timestamp": "2026-03-14T14:07:05.123456"
}
```

`status` values: `running` | `complete` | `error` | `speaking` | `listening` | `response` | `info`

---

## How the Voice Agent Works

Agent 5B runs **entirely separately from the LangGraph event loop** to avoid an asyncio deadlock. Here is the exact flow:

```
LangGraph node (sync)
        │
        ▼
_run_voice_agent_in_thread()
        │
        ├── spawns daemon Thread
        │       │
        │       ├── asyncio.new_event_loop()     ← isolated loop
        │       │
        │       ├── VoiceAgent.speak(intro)
        │       │       └── Deepgram TTS WebSocket
        │       │               └── PyAudio output stream
        │       │
        │       └── VoiceAgent.run_flux(claim)
        │               ├── Deepgram STT WebSocket  ← mic input
        │               ├── asyncio.gather(
        │               │       send_audio(),        ← PyAudio → WS
        │               │       receive_transcript(), ← WS → Groq LLM
        │               │       silence_timer()       ← 8s fallback
        │               │   )
        │               └── sets agent._last_intent = "YES" | "NO"
        │
        ├── threading.Event.wait(timeout=60)
        └── returns agent._last_intent to LangGraph node
```

The `threading.Event` blocks the sync LangGraph node until the voice call completes. The isolated event loop means PyAudio's blocking reads never starve Flask or LangGraph's outer loop.

---

## Notes

- The Pinecone index (`fhirdb`) must be populated with payer policy vectors before Agent 2 and Agent 4 can make meaningful decisions. Run your policy ingestion pipeline first.
- Agent 5B requires a working microphone and audio output device. In server/cloud deployments without audio hardware, the voice agent will fall back gracefully — the 8-second silence timer triggers YES and the correction proceeds automatically.
- The `helper/helper.py` module must expose `load_pdf_files(dirs)` and `download_embeddings()` — these are imported by Agents 2 and 4.
- All agents write logs to individual `.log` files in their working directories (`agent2_change_detector.log`, `agent3_fhir_updater.log`, etc.).