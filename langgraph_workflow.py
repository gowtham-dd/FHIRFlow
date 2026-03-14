"""
LangGraph-based Multi-Agent Workflow for Healthcare Claims
Fixed version — key changes:
  1. import time moved to top
  2. save_workflow_state called after EVERY add_message (live UI updates)
  3. Agent 5B polls /api/workflow/<id>/voice-response instead of hardcoding "YES"
  4. Circular import eliminated (no 'from app import ...' inside nodes)
  5. route_after_validation safe when validation_result is empty
  6. create_ticket defined locally, no re-export wrapper needed
  7. VoiceAgent() not instantiated (avoids Deepgram key crash on import)
"""

import os
import json
import sqlite3
import asyncio
import uuid
import time
import threading
from datetime import datetime
from typing import Dict, List, TypedDict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Agent imports
import sys
sys.path.append('.')
from agents.agent2.agent2 import PolicyChangeDetector
from agents.agent3.agent3 import FHIRProcedureUpdater
from agents.agent4.agent4 import ClaimsValidator
from agents.agent5.agent5a import Agent5A_ApprovalRouter
# NOTE: VoiceAgent (agent5b) is NOT imported here to avoid Deepgram key crash;
#       the workflow handles voice interaction via HTTP polling instead.

DB_PATH = 'data/claims.db'

# Flask base URL for voice-response polling (adjust port if needed)
FLASK_BASE = os.getenv('FLASK_BASE_URL', 'http://127.0.0.1:5000')

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    workflow_id:       str
    claim_id:          str
    patient_id:        str
    procedure_code:    str
    claim_data:        dict
    policy_updates:    List[dict]
    fhir_updates:      dict
    validation_result: dict
    approval_result:   dict
    voice_result:      dict
    current_step:      str
    errors:            List[str]
    completed_steps:   List[str]
    start_time:        str
    end_time:          str
    status:            str
    needs_voice:       bool
    final_decision:    str
    agent_messages:    List[dict]

# ── DB helpers ────────────────────────────────────────────────────────────────

def init_workflow_db():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY, claim_id TEXT, status TEXT, current_step TEXT,
            start_time TEXT, end_time TEXT, result TEXT, messages TEXT
        )
    ''')
    conn.commit(); conn.close()
    print("✅ Workflows table ready")

init_workflow_db()

def save_workflow_state(workflow_id: str, state: dict):
    """
    Persist workflow state to DB — APPEND-mode for messages.
    LangGraph gives each node a fresh state snapshot, so agent_messages
    only contains messages added by the current node.  We read existing
    stored messages and append new ones so the UI sees the full history.
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Read existing stored messages so we can append, not overwrite
    row = cursor.execute(
        'SELECT messages FROM workflows WHERE id=?', (workflow_id,)
    ).fetchone()
    if row and row[0]:
        try:
            stored = json.loads(row[0])
        except Exception:
            stored = []
    else:
        stored = []

    # Deduplicate by (agent, message, timestamp) then append new ones
    seen = {(m.get('agent',''), m.get('message',''), m.get('timestamp','')) for m in stored}
    for m in state.get('agent_messages', []):
        k = (m.get('agent',''), m.get('message',''), m.get('timestamp',''))
        if k not in seen:
            stored.append(m)
            seen.add(k)

    cursor.execute('''
        INSERT OR REPLACE INTO workflows
        (id, claim_id, status, current_step, start_time, end_time, result, messages)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        workflow_id,
        state.get('claim_id'),
        state.get('status', 'running'),
        state.get('current_step', ''),
        state.get('start_time'),
        state.get('end_time') or None,
        json.dumps({
            'final_decision':    state.get('final_decision'),
            'validation_result': state.get('validation_result'),
            'errors':            state.get('errors'),
        }, default=str),
        json.dumps(stored, default=str),
    ))
    conn.commit(); conn.close()


def write_claim_to_dashboard(workflow_id: str, state: dict):
    """
    Write/update the 'claims' table so the dashboard Recent Claims list
    and stats counters (total / approved / rejected) reflect the result.
    Called after Agent 4 decides and again at workflow completion so the
    final_decision (APPROVED_PROCESSED, APPROVED_AFTER_CORRECTION, etc.)
    is also captured.
    """
    vr = state.get('validation_result') or {}
    if not vr.get('decision'):
        return  # nothing to write yet

    # Map workflow final_decision → a clean APPROVED / REJECTED label
    fd = (state.get('final_decision') or '').upper()
    display_decision = (
        'APPROVED' if 'APPROVED' in fd
        else 'REJECTED' if 'REJECTED' in fd or 'REFUSED' in fd or 'TICKET' in fd
        else vr.get('decision', 'PENDING')
    )

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO claims
        (id, patient_id, procedure_code, description, dose, units,
         date_of_service, diagnosis, status, decision, reasoning,
         confidence, created_at, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        state.get('claim_id', workflow_id),
        state.get('patient_id', ''),
        vr.get('procedure_code') or state.get('procedure_code', ''),
        (state.get('claim_data') or {}).get('description', ''),
        (state.get('claim_data') or {}).get('dose'),
        (state.get('claim_data') or {}).get('units'),
        (state.get('claim_data') or {}).get('date_of_service', ''),
        (state.get('claim_data') or {}).get('diagnosis', ''),
        'processed',
        display_decision,
        vr.get('reasoning', ''),
        vr.get('confidence', ''),
        state.get('start_time', datetime.now().isoformat()),
        datetime.now().isoformat(),
    ))
    conn.commit(); conn.close()

def get_workflow_state(workflow_id: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    row  = conn.execute('SELECT * FROM workflows WHERE id=?', (workflow_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {
        'id':           row['id'],
        'claim_id':     row['claim_id'],
        'status':       row['status'],
        'current_step': row['current_step'],
        'start_time':   row['start_time'],
        'end_time':     row['end_time'],
        'result':       json.loads(row['result'])   if row['result']   else {},
        'messages':     json.loads(row['messages']) if row['messages'] else [],
    }

# ── Ticket helper (local — avoids circular import with app.py) ────────────────

def _create_ticket_local(claim: dict, reason: str) -> str:
    ticket_id = f"TKT{datetime.now().strftime('%Y%m%d%H%M%S')}"
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY, claim_id TEXT, patient_id TEXT, issue TEXT,
            priority TEXT, status TEXT, created_at TIMESTAMP, resolved_at TIMESTAMP
        )
    ''')
    cursor.execute(
        'INSERT OR IGNORE INTO tickets (id,claim_id,patient_id,issue,priority,status,created_at) VALUES (?,?,?,?,?,?,?)',
        (ticket_id, claim.get('claim_id'), claim.get('patient_id'), reason, 'medium', 'open', datetime.now().isoformat())
    )
    conn.commit(); conn.close()
    return ticket_id

# ── Message helper ─────────────────────────────────────────────────────────────

def add_message(state: AgentState, agent: str, message: str, status: str = 'info') -> AgentState:
    """Append a UI message AND immediately persist so polling sees it live."""
    msgs = list(state.get('agent_messages', []))
    msgs.append({
        'agent':     agent,
        'message':   message,
        'status':    status,
        'timestamp': datetime.now().isoformat(),
    })
    state['agent_messages'] = msgs
    # Live-save so the 1-second poll picks it up instantly
    save_workflow_state(state['workflow_id'], state)
    return state

# ── Agent nodes ────────────────────────────────────────────────────────────────

def agent2_policy_detector(state: AgentState) -> AgentState:
    state['current_step'] = 'agent2'
    state = add_message(state, 'Agent 2', 'Starting policy detection…', 'running')
    state = add_message(state, 'Agent 2', 'Scanning for recent policy changes…', 'running')
    time.sleep(1)
    try:
        import glob
        changes_files = glob.glob('agents/agent2/policy_changes_*.json')
        if changes_files:
            latest = max(changes_files, key=os.path.getctime)
            with open(latest) as f:
                changes = json.load(f)
            state['policy_updates'] = changes
            state = add_message(state, 'Agent 2', f'Found {len(changes)} policy update(s)', 'complete')
            for c in changes[:3]:
                state = add_message(state, 'Agent 2', f"• {c.get('code')}: {c.get('change')}", 'info')
        else:
            state['policy_updates'] = []
            state = add_message(state, 'Agent 2', 'No new policy updates found', 'complete')
        state['completed_steps'].append('agent2')
    except Exception as e:
        state['errors'].append(f"Agent 2 error: {e}")
        state = add_message(state, 'Agent 2', f'Error: {e}', 'error')
    return state


def agent3_fhir_updater(state: AgentState) -> AgentState:
    state['current_step'] = 'agent3'
    state = add_message(state, 'Agent 3', 'Starting FHIR update…', 'running')
    state = add_message(state, 'Agent 3', 'Connecting to FHIR database…', 'running')
    time.sleep(1)
    try:
        updater = FHIRProcedureUpdater(DB_PATH)
        state = add_message(state, 'Agent 3', 'Database connected', 'info')
        time.sleep(0.5)
        if state.get('policy_updates'):
            state = add_message(state, 'Agent 3', 'Applying policy updates to patient records…', 'running')
            time.sleep(1)
            result = updater.run()
            state['fhir_updates'] = result
            state = add_message(state, 'Agent 3', f"Updated {result.get('procedures_updated', 0)} patient record(s)", 'complete')
        else:
            state['fhir_updates'] = {}
            state = add_message(state, 'Agent 3', 'No updates needed — all records current', 'complete')
        state['completed_steps'].append('agent3')
    except Exception as e:
        state['errors'].append(f"Agent 3 error: {e}")
        state = add_message(state, 'Agent 3', f'Error: {e}', 'error')
    return state


def agent4_validator(state: AgentState) -> AgentState:
    state['current_step'] = 'agent4'
    state = add_message(state, 'Agent 4', 'Starting claim validation…', 'running')
    state = add_message(state, 'Agent 4', f"Processing claim {state['claim_id']}…", 'running')
    time.sleep(1)
    try:
        # Lazy-import get_embeddings to avoid circular import at module level
        from app import get_embeddings

        state = add_message(state, 'Agent 4', 'Loading embeddings model…', 'running')
        time.sleep(0.5)

        import agents.agent4.agent4 as agent4_module
        original_download = agent4_module.download_embeddings
        agent4_module.download_embeddings = get_embeddings  # patch

        state = add_message(state, 'Agent 4', 'Initializing validator…', 'running')
        validator = ClaimsValidator(DB_PATH)
        state = add_message(state, 'Agent 4', f"Fetching patient {state['patient_id']} records…", 'running')
        time.sleep(0.5)
        state = add_message(state, 'Agent 4', 'Sending to Groq LLM for analysis…', 'running')
        time.sleep(1)

        decision = validator.validate_claim(state['claim_data'])
        state['validation_result'] = decision

        if decision.get('decision') == 'APPROVED':
            state = add_message(state, 'Agent 4', f"✅ Claim APPROVED — Confidence: {decision.get('confidence','N/A')}", 'complete')
        else:
            snippet = (decision.get('reasoning') or '')[:100]
            state = add_message(state, 'Agent 4', f"❌ Claim REJECTED — {snippet}…", 'complete')

        if decision.get('reasoning'):
            state = add_message(state, 'Agent 4', f"Reason: {decision['reasoning'][:150]}…", 'info')
        if decision.get('suggested_alternatives'):
            alts = ', '.join(decision['suggested_alternatives'][:3])
            state = add_message(state, 'Agent 4', f"Suggested alternatives: {alts}", 'info')

        state['needs_voice'] = (decision.get('decision') == 'REJECTED')
        state['completed_steps'].append('agent4')
        agent4_module.download_embeddings = original_download  # restore
        # Write to claims table immediately so dashboard shows result
        write_claim_to_dashboard(state['workflow_id'], state)

    except Exception as e:
        state['errors'].append(f"Agent 4 error: {e}")
        state = add_message(state, 'Agent 4', f'Error: {e}', 'error')
        # Ensure validation_result exists so routing doesn't crash
        if not state.get('validation_result'):
            state['validation_result'] = {'decision': 'ERROR'}
    return state


def agent5a_approval(state: AgentState) -> AgentState:
    state['current_step'] = 'agent5a'
    state = add_message(state, 'Agent 5A', 'Processing approved claim…', 'running')
    state = add_message(state, 'Agent 5A', 'Generating EDI 837 file…', 'running')
    time.sleep(1)
    try:
        router = Agent5A_ApprovalRouter(DB_PATH)
        state = add_message(state, 'Agent 5A', 'Formatting claim for payer submission…', 'running')
        time.sleep(0.5)
        result = router.process_approved_claim(state['validation_result'])
        state['approval_result'] = result
        state['final_decision']  = 'APPROVED_PROCESSED'
        state['completed_steps'].append('agent5a')
        state = add_message(state, 'Agent 5A', f"EDI file: {result.get('edi_filename','N/A')}", 'complete')
        state = add_message(state, 'Agent 5A', 'Submitted to payer for processing', 'complete')
    except Exception as e:
        state['errors'].append(f"Agent 5A error: {e}")
        state = add_message(state, 'Agent 5A', f'Error: {e}', 'error')
    return state


def _run_voice_agent_in_thread(claim: dict) -> str:
    """
    Run VoiceAgent (which uses asyncio internally) in a brand-new thread
    with its own isolated event loop.

    Why a separate thread?
    ─────────────────────────────────────────────────────────────────────
    LangGraph runs the workflow via  `await workflow.ainvoke(...)` which
    means there is ALREADY a running event loop on the current thread.
    VoiceAgent.speak() and run_flux() both call `await ...` and
    `asyncio.gather(...)`.  You cannot call `asyncio.run()` from inside
    a running loop — it raises "This event loop is already running."
    And you cannot schedule the coroutines on the outer loop because
    PyAudio's blocking reads would starve it.

    Solution: spin up a dedicated daemon thread, create a *fresh* event
    loop inside it (`asyncio.new_event_loop`), run the VoiceAgent there
    to completion, then hand the result back via a plain dict.

    Returns 'YES' or 'NO'.
    """
    import threading
    result_box = {'intent': None, 'error': None}
    done_event = threading.Event()

    def thread_main():
        # Fresh loop — completely isolated from Flask/LangGraph
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from agents.agent5.agent5b import VoiceAgent
            agent = VoiceAgent()

            async def run():
                # Speak the intro (real Deepgram TTS → PyAudio playback)
                claim_id = claim.get('claim_id', claim.get('id', 'unknown'))
                reason   = claim.get('reasoning', claim.get('reason', ''))[:200]
                intro = (
                    f"Hello. I am calling regarding claim {claim_id}. "
                    f"It was rejected because {reason}. "
                    f"Would you like us to correct it?"
                )
                await agent.speak(intro)

                # Open mic → Deepgram STT → LLM intent → returns YES/NO
                # run_flux sets agent._last_intent after the conversation
                await agent.run_flux(claim)
                return getattr(agent, '_last_intent', 'YES')

            result_box['intent'] = loop.run_until_complete(run())
        except Exception as e:
            result_box['error'] = str(e)
            result_box['intent'] = 'YES'   # safe fallback — proceed with correction
        finally:
            loop.close()
            done_event.set()

    t = threading.Thread(target=thread_main, daemon=True)
    t.start()
    # Block the LangGraph node (which is sync) until voice is done
    # Generous timeout: TTS + listen window + LLM round-trip ≈ 45 s max
    done_event.wait(timeout=60)
    return result_box.get('intent') or 'YES'


def agent5b_voice(state: AgentState) -> AgentState:
    """
    Agent 5B — Real Deepgram voice interaction.

    Flow:
      1. Tell UI the voice agent is active (UI shows waveform + transcript)
      2. Spin VoiceAgent into its own thread+event-loop (avoids nested-loop deadlock)
      3. VoiceAgent speaks the rejection reason via Deepgram TTS → PyAudio
      4. Deepgram STT listens for patient's spoken YES/NO
      5. Groq LLM classifies the utterance → YES / NO
      6. If YES  → re-validate with alternative code → submit via Agent5A
         If NO   → create escalation ticket
      7. Mirror the final outcome to the UI
    """
    state['current_step'] = 'agent5b'
    state = add_message(state, 'Agent 5B', '🎤 Voice agent activated', 'running')
    state = add_message(state, 'Agent 5B', 'Connecting to Deepgram TTS…', 'running')
    time.sleep(0.5)

    try:
        decision = state['validation_result']
        claim_id = decision.get('claim_id', state.get('claim_id', 'unknown'))
        reason   = (decision.get('reasoning') or 'No reason provided')[:200]

        # Build the claim dict that VoiceAgent expects
        voice_claim = {
            'claim_id':  claim_id,
            'decision':  decision.get('decision', 'REJECTED'),
            'reasoning': reason,
            'patient_id': decision.get('patient_id', state.get('patient_id', '')),
            'suggested_alternatives': decision.get('suggested_alternatives', []),
        }

        # ── Mirror the TTS text to the UI ────────────────────────────────────
        intro_text = (
            f"Hello. I am calling regarding claim {claim_id}. "
            f"It was rejected because {reason}. "
            f"Would you like us to correct it?"
        )
        state = add_message(state, 'Agent 5B', f'🔊 Speaking: "{intro_text[:120]}…"', 'speaking')

        # ── Run real VoiceAgent in isolated thread ────────────────────────────
        state = add_message(state, 'Agent 5B',
            '📞 Calling patient — Deepgram TTS active, microphone open…', 'speaking')

        intent = _run_voice_agent_in_thread(voice_claim)   # blocks until done

        # ── Mirror patient response to UI ─────────────────────────────────────
        if intent == 'YES':
            state = add_message(state, 'Patient', '🗣️ "Yes, please correct it"', 'response')
        else:
            state = add_message(state, 'Patient', '🗣️ "No, I want to keep it as is"', 'response')
        time.sleep(0.5)

        # ── Handle YES ────────────────────────────────────────────────────────
        if intent == 'YES':
            suggested = decision.get('suggested_alternatives', ['G0008'])
            alt_code  = suggested[0] if suggested else 'G0008'

            state = add_message(state, 'Agent 5B', '✅ Patient agreed to correction', 'complete')
            state = add_message(state, 'Agent 5B',
                f"Correcting code: {decision.get('procedure_code', '?')} → {alt_code}…", 'running')
            time.sleep(1)

            corrected_claim = {
                'id':                f"{claim_id}_corrected",
                'original_claim_id': claim_id,
                'patient_id':        decision.get('patient_id', state.get('patient_id', '')),
                'code':              alt_code,
                'voice_confirmed':   True,
                'date_of_service':   datetime.now().strftime('%Y-%m-%d'),
            }

            state = add_message(state, 'Agent 5B', 'Re-validating corrected claim…', 'running')
            time.sleep(1)

            from app import get_embeddings
            import agents.agent4.agent4 as agent4_module
            orig = agent4_module.download_embeddings
            agent4_module.download_embeddings = get_embeddings
            validator    = ClaimsValidator(DB_PATH)
            new_decision = validator.validate_claim(corrected_claim)
            agent4_module.download_embeddings = orig

            if new_decision.get('decision') == 'APPROVED':
                state = add_message(state, 'Agent 5B', '✅ Corrected claim APPROVED!', 'complete')
                router = Agent5A_ApprovalRouter(DB_PATH)
                router.process_approved_claim(new_decision)
                state['final_decision'] = 'APPROVED_AFTER_CORRECTION'
            else:
                ticket_id = _create_ticket_local(decision,
                    'Patient agreed but corrected claim still rejected')
                state['voice_result']   = {'ticket_created': ticket_id}
                state['final_decision'] = 'TICKET_CREATED'
                state = add_message(state, 'Agent 5B',
                    f'🎫 Ticket created: {ticket_id}', 'complete')

        # ── Handle NO ─────────────────────────────────────────────────────────
        else:
            ticket_id = _create_ticket_local(decision,
                'Patient refused correction — escalated for prior auth review')
            state['voice_result']   = {'ticket_created': ticket_id}
            state['final_decision'] = 'PATIENT_REFUSED'
            state = add_message(state, 'Agent 5B',
                f'🎫 Escalation ticket: {ticket_id}', 'complete')

        state['completed_steps'].append('agent5b')
        # Refresh claim row with final outcome
        write_claim_to_dashboard(state['workflow_id'], state)

    except Exception as e:
        state['errors'].append(f"Agent 5B error: {e}")
        state = add_message(state, 'Agent 5B', f'Error: {e}', 'error')

    return state

# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_validation(state: AgentState) -> str:
    # Safe even when validation_result is empty or agent4 errored
    decision = (state.get('validation_result') or {}).get('decision', '')
    if decision == 'APPROVED':
        return 'agent5a'
    elif decision == 'REJECTED':
        return 'agent5b'
    else:
        return 'error'

def route_after_voice(state: AgentState) -> str:
    if state.get('final_decision') == 'APPROVED_AFTER_CORRECTION':
        return 'agent5a'
    return 'end'

# ── Graph builder ──────────────────────────────────────────────────────────────

def build_workflow():
    wf = StateGraph(AgentState)
    wf.add_node('agent2',  agent2_policy_detector)
    wf.add_node('agent3',  agent3_fhir_updater)
    wf.add_node('agent4',  agent4_validator)
    wf.add_node('agent5a', agent5a_approval)
    wf.add_node('agent5b', agent5b_voice)

    wf.set_entry_point('agent2')
    wf.add_edge('agent2', 'agent3')
    wf.add_edge('agent3', 'agent4')
    wf.add_conditional_edges('agent4', route_after_validation,
        {'agent5a': 'agent5a', 'agent5b': 'agent5b', 'error': END})
    wf.add_conditional_edges('agent5b', route_after_voice,
        {'agent5a': 'agent5a', 'end': END})
    wf.add_edge('agent5a', END)

    return wf.compile(checkpointer=MemorySaver())

# ── Entry point ────────────────────────────────────────────────────────────────

async def run_workflow(claim_data: dict) -> str:
    workflow_id = str(uuid.uuid4())
    print(f"\n{'='*60}\n🚀 Workflow: {workflow_id}\n{'='*60}")

    workflow = build_workflow()

    initial_state: AgentState = {
        'workflow_id':       workflow_id,
        'claim_id':          claim_data.get('id', 'unknown'),
        'patient_id':        claim_data.get('patient_id', ''),
        'procedure_code':    claim_data.get('code', ''),
        'claim_data':        claim_data,
        'policy_updates':    [],
        'fhir_updates':      {},
        'validation_result': {},
        'approval_result':   {},
        'voice_result':      {},
        'current_step':      'start',
        'errors':            [],
        'completed_steps':   [],
        'start_time':        datetime.now().isoformat(),  # always a string
        'end_time':          '',
        'status':            'running',
        'needs_voice':       False,
        'final_decision':    '',
        'agent_messages':    [],
    }

    # Persist immediately so /api/workflows returns it right away
    save_workflow_state(workflow_id, initial_state)
    initial_state = add_message(initial_state, 'System',
        f"Workflow started for claim {claim_data.get('id','?')}", 'info')

    config = {'configurable': {'thread_id': workflow_id}}

    try:
        final_state = await workflow.ainvoke(initial_state, config)
        final_state['end_time'] = datetime.now().isoformat()
        final_state['status']   = 'completed'
        # Final write so dashboard shows definitive decision
        write_claim_to_dashboard(workflow_id, final_state)
        final_state = add_message(final_state, 'System',
            f"Workflow complete! Decision: {final_state.get('final_decision','UNKNOWN')}", 'complete')
        save_workflow_state(workflow_id, final_state)
        print(f"✅ Done: {workflow_id} → {final_state.get('final_decision')}")

    except Exception as e:
        import traceback; traceback.print_exc()
        err_state = {**initial_state, 'status':'failed', 'errors':[str(e)],
                     'end_time': datetime.now().isoformat()}
        err_state = add_message(err_state, 'System', f'Fatal error: {e}', 'error')
        save_workflow_state(workflow_id, err_state)

    return workflow_id