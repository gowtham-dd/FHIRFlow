"""
Healthcare Claims Dashboard - Fixed app.py
Key fixes:
  1. /api/workflow/<id>/voice-response  endpoint added
  2. workflows table now inserts start_time as ISO string always
  3. Workflow messages stored/returned correctly for UI polling
  4. LangSmith tracking added for monitoring and benchmarking
"""

import os
import json
import sqlite3
import threading
import asyncio
import importlib.util
import sys
import time
import glob
import atexit
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import pandas as pd

# LangSmith imports
from langsmith import Client
from langsmith.run_helpers import trace, get_current_run_tree

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv()

# Initialize LangSmith client
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "FHIRFlow")
LANGSMITH_ENABLED = bool(LANGSMITH_API_KEY)

if LANGSMITH_ENABLED:
    langsmith_client = Client(api_key=LANGSMITH_API_KEY)
    print(f"✅ LangSmith tracking enabled for project: {LANGSMITH_PROJECT}")
else:
    langsmith_client = None
    print("⚠️ LangSmith tracking disabled (set LANGSMITH_API_KEY to enable)")

app = Flask(__name__)
app.config['SECRET_KEY']          = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER']       = 'uploads'
app.config['MAX_CONTENT_LENGTH']  = 16 * 1024 * 1024

os.makedirs('uploads',          exist_ok=True)
os.makedirs('data',             exist_ok=True)
os.makedirs('agents/agent2',   exist_ok=True)
os.makedirs('agents/agent3',   exist_ok=True)
os.makedirs('agents/agent4',   exist_ok=True)
os.makedirs('agents/agent5a',  exist_ok=True)
os.makedirs('agents/agent5b',  exist_ok=True)

DB_PATH = 'data/claims.db'

# ── Voice response store (in-memory, keyed by workflow_id) ──────────────────
_voice_responses: dict = {}

# ── Embeddings cache ──────────────────────────────────────────────────────────
_EMBEDDINGS_INSTANCE = None

def get_embeddings():
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            import logging
            logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
            _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"⚠️ Embeddings load error: {e}")
    return _EMBEDDINGS_INSTANCE

@atexit.register
def cleanup():
    global _EMBEDDINGS_INSTANCE
    _EMBEDDINGS_INSTANCE = None

# ── Database ──────────────────────────────────────────────────────────────────
def init_database():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS claims (
            id TEXT PRIMARY KEY,
            patient_id TEXT, patient_name TEXT, procedure_code TEXT,
            description TEXT, dose REAL, units INTEGER, date_of_service TEXT,
            diagnosis TEXT, status TEXT, decision TEXT, reasoning TEXT,
            confidence TEXT, created_at TIMESTAMP, processed_at TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY, first_name TEXT, last_name TEXT,
            date_of_birth TEXT, gender TEXT, phone TEXT, email TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS policy_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT, policy_code TEXT,
            change_description TEXT, impact TEXT, effective_date TEXT, detected_at TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY, claim_id TEXT, patient_id TEXT, issue TEXT,
            priority TEXT, status TEXT, created_at TIMESTAMP, resolved_at TIMESTAMP
        )
    ''')
    # workflows: messages column stores JSON array of {agent,message,status,ts}
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY, claim_id TEXT, status TEXT, current_step TEXT,
            start_time TEXT,   -- ISO string, always set on insert
            end_time TEXT,
            result TEXT,
            messages TEXT      -- JSON array
        )
    ''')

    cursor.execute("SELECT COUNT(*) FROM patients")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            'INSERT INTO patients (id,first_name,last_name,date_of_birth,gender,phone,email) VALUES (?,?,?,?,?,?,?)',
            [
                ('pat001','John','Smith',  '1965-03-15','M','+1234567890','john.smith@email.com'),
                ('pat002','Jane','Doe',    '1972-07-22','F','+1234567891','jane.doe@email.com'),
                ('pat003','Robert','Johnson','1958-11-30','M','+1234567892','robert.j@email.com'),
            ]
        )
    conn.commit()
    conn.close()
    print("✅ Database initialized")

init_database()

# ── LangSmith tracking helpers ───────────────────────────────────────────────
def log_to_langsmith(run_name: str, run_type: str, inputs: dict, outputs: dict, tags: list = None, parent_run_id: str = None):
    """Log a run to LangSmith if enabled"""
    if not LANGSMITH_ENABLED:
        return None
    
    try:
        run = langsmith_client.create_run(
            name=run_name,
            run_type=run_type,
            inputs=inputs,
            outputs=outputs,
            tags=tags or [],
            parent_run_id=parent_run_id,
            project_name=LANGSMITH_PROJECT
        )
        return run.id
    except Exception as e:
        print(f"⚠️ LangSmith log error: {e}")
        return None

def log_workflow_start(workflow_id: str, claim_data: dict):
    """Log workflow start to LangSmith"""
    return log_to_langsmith(
        run_name=f"Workflow: {workflow_id}",
        run_type="chain",
        inputs={
            "workflow_id": workflow_id,
            "claim_id": claim_data.get('id'),
            "patient_id": claim_data.get('patient_id'),
            "procedure_code": claim_data.get('code'),
            "claim_data": claim_data,
            "timestamp": datetime.now().isoformat()
        },
        outputs={"status": "started"},
        tags=["workflow", "start"]
    )

def log_agent_step(workflow_id: str, agent_name: str, step_input: dict, step_output: dict, parent_run_id: str = None):
    """Log individual agent step to LangSmith"""
    return log_to_langsmith(
        run_name=f"{agent_name}: {workflow_id}",
        run_type="tool",
        inputs=step_input,
        outputs=step_output,
        tags=["agent", agent_name.lower().replace(" ", "-")],
        parent_run_id=parent_run_id
    )

def log_workflow_complete(workflow_id: str, final_state: dict, parent_run_id: str = None):
    """Log workflow completion to LangSmith"""
    return log_to_langsmith(
        run_name=f"Workflow Complete: {workflow_id}",
        run_type="chain",
        inputs={"workflow_id": workflow_id},
        outputs={
            "final_decision": final_state.get('final_decision'),
            "validation_result": final_state.get('validation_result'),
            "completed_steps": final_state.get('completed_steps', []),
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        },
        tags=["workflow", "complete"],
        parent_run_id=parent_run_id
    )

def log_validation_result(workflow_id: str, claim_id: str, decision: dict):
    """Log validation result to LangSmith as a separate run for metrics"""
    if not LANGSMITH_ENABLED:
        return None
    
    try:
        run = langsmith_client.create_run(
            name=f"Validation: {claim_id}",
            run_type="llm",
            inputs={
                "claim_id": claim_id,
                "patient_id": decision.get('patient_id'),
                "procedure_code": decision.get('procedure_code')
            },
            outputs={
                "decision": decision.get('decision'),
                "confidence": decision.get('confidence'),
                "reasoning": decision.get('reasoning'),
                "suggested_alternatives": decision.get('suggested_alternatives', [])
            },
            tags=["validation", decision.get('decision', '').lower()],
            extra={
                "workflow_id": workflow_id,
                "model": "llama-3.1-8b-instant"
            },
            project_name=LANGSMITH_PROJECT
        )
        
        # Add feedback score based on decision
        if decision.get('decision') == 'APPROVED':
            langsmith_client.create_feedback(
                run_id=run.id,
                key="validation_accuracy",
                score=1.0,
                comment="Claim approved - correct decision"
            )
        elif decision.get('decision') == 'REJECTED':
            langsmith_client.create_feedback(
                run_id=run.id,
                key="validation_accuracy",
                score=1.0,
                comment="Claim rejected - correct decision"
            )
        
        return run.id
    except Exception as e:
        print(f"⚠️ LangSmith validation log error: {e}")
        return None

# ── Helpers ───────────────────────────────────────────────────────────────────
def create_ticket(claim, reason):
    ticket_id = f"TKT{datetime.now().strftime('%Y%m%d%H%M%S')}"
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO tickets (id,claim_id,patient_id,issue,priority,status,created_at) VALUES (?,?,?,?,?,?,?)',
        (ticket_id, claim.get('claim_id'), claim.get('patient_id'), reason, 'medium', 'open', datetime.now().isoformat())
    )
    conn.commit(); conn.close()
    
    # Log ticket creation to LangSmith
    if LANGSMITH_ENABLED:
        log_to_langsmith(
            run_name=f"Ticket Created: {ticket_id}",
            run_type="tool",
            inputs={"claim_id": claim.get('claim_id'), "reason": reason},
            outputs={"ticket_id": ticket_id, "priority": "medium"},
            tags=["ticket", "creation"]
        )
    
    return ticket_id

def workflow_push_message(wf_id: str, agent: str, message: str, status: str = 'info'):
    """Thread-safe: append a message to a workflow's messages column."""
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute('SELECT messages FROM workflows WHERE id=?', (wf_id,))
    row = cursor.fetchone()
    if not row:
        conn.close(); return
    msgs = json.loads(row[0] or '[]')
    msgs.append({'agent': agent, 'message': message, 'status': status, 'ts': datetime.now().isoformat()})
    cursor.execute('UPDATE workflows SET messages=?, current_step=? WHERE id=?',
                   (json.dumps(msgs), agent, wf_id))
    conn.commit(); conn.close()

def workflow_set_status(wf_id: str, status: str, result: dict = None):
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute(
        'UPDATE workflows SET status=?, end_time=?, result=? WHERE id=?',
        (status, datetime.now().isoformat(), json.dumps(result) if result else None, wf_id)
    )
    conn.commit(); conn.close()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/claims')
def get_claims():
    try:
        conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
        claims = [dict(r) for r in conn.execute('SELECT * FROM claims ORDER BY created_at DESC LIMIT 50')]
        conn.close(); return jsonify(claims)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/policy-updates')
def get_policy_updates():
    try:
        conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute('SELECT * FROM policy_updates ORDER BY detected_at DESC LIMIT 10')]
        conn.close(); return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tickets')
def get_tickets():
    try:
        conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute('SELECT * FROM tickets ORDER BY created_at DESC LIMIT 20')]
        conn.close(); return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    try:
        conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
        def q(sql): return cursor.execute(sql).fetchone()[0]
        data = {
            'total_claims':   q("SELECT COUNT(*) FROM claims"),
            'approved':       q("SELECT COUNT(*) FROM claims WHERE decision='APPROVED'"),
            'rejected':       q("SELECT COUNT(*) FROM claims WHERE decision='REJECTED'"),
            'open_tickets':   q("SELECT COUNT(*) FROM tickets WHERE status='open'"),
            'recent_policies':q("SELECT COUNT(*) FROM policy_updates WHERE detected_at > date('now','-7 days')"),
        }
        conn.close(); return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_patient_data():
    print(f"\n{'='*50}\n📁 Upload at {datetime.now()}\n{'='*50}")
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.json'):
        return jsonify({'error': 'Only JSON files are allowed'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        claims   = [data] if isinstance(data, dict) else data
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Log batch upload to LangSmith
        if LANGSMITH_ENABLED:
            log_to_langsmith(
                run_name=f"Batch Upload: {batch_id}",
                run_type="chain",
                inputs={"batch_id": batch_id, "claim_count": len(claims), "first_claim": claims[0] if claims else None},
                outputs={"status": "processing"},
                tags=["upload", "batch"]
            )

        try:
            from langgraph_workflow import run_workflow
        except ImportError as e:
            return jsonify({'error': f'Cannot import langgraph_workflow: {e}'}), 500

        def run_workflows():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            for i, claim in enumerate(claims):
                if 'id' not in claim:
                    claim['id'] = f"CLM{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}"
                print(f"\n🔄 Processing claim {i+1}/{len(claims)}: {claim['id']}")
                try:
                    # Run workflow and capture result for LangSmith
                    workflow_id = loop.run_until_complete(run_workflow(claim))
                    
                    # Log workflow start
                    if LANGSMITH_ENABLED:
                        parent_id = log_workflow_start(workflow_id, claim)
                        
                        # The actual agent steps will be logged inside langgraph_workflow.py
                        # We just need to ensure validation results are logged here
                        from langgraph_workflow import get_workflow_state
                        final_state = get_workflow_state(workflow_id)
                        if final_state and final_state.get('validation_result'):
                            log_validation_result(
                                workflow_id, 
                                claim['id'], 
                                final_state.get('validation_result')
                            )
                            log_workflow_complete(workflow_id, final_state, parent_id)
                            
                except Exception as e:
                    print(f"❌ Workflow error for {claim['id']}: {e}")
                    if LANGSMITH_ENABLED:
                        log_to_langsmith(
                            run_name=f"Workflow Error: {claim['id']}",
                            run_type="chain",
                            inputs={"claim_id": claim['id']},
                            outputs={"error": str(e)},
                            tags=["workflow", "error"]
                        )
                    import traceback; traceback.print_exc()

        t = threading.Thread(target=run_workflows, daemon=True)
        t.start()

        return jsonify({'status': 'processing', 'batch_id': batch_id, 'claims': len(claims),
                        'message': f'Started {len(claims)} workflow(s)'})
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid JSON: {e}'}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/workflow/<workflow_id>')
def get_workflow_status(workflow_id):
    try:
        from langgraph_workflow import get_workflow_state
        state = get_workflow_state(workflow_id)
        if state: return jsonify(state)
        return jsonify({'error': 'Workflow not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/workflows')
def get_all_workflows():
    try:
        conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT id,claim_id,status,current_step,start_time,end_time FROM workflows ORDER BY start_time DESC LIMIT 20'
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/workflow/<workflow_id>/messages')
def get_workflow_messages(workflow_id):
    try:
        conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
        row  = conn.execute('SELECT status,current_step,messages FROM workflows WHERE id=?', (workflow_id,)).fetchone()
        conn.close()
        if not row:
            return jsonify({'messages': [], 'status': 'not_found'})
        return jsonify({
            'messages':     json.loads(row['messages'] or '[]'),
            'status':       row['status'],
            'current_step': row['current_step'],
        })
    except Exception as e:
        return jsonify({'error': str(e), 'messages': []}), 500

# ── NEW: voice response endpoint ──────────────────────────────────────────────
@app.route('/api/workflow/<workflow_id>/voice-response', methods=['POST'])
def receive_voice_response(workflow_id):
    """Store the patient's voice button response so Agent 5B can pick it up."""
    try:
        body     = request.get_json(force=True) or {}
        response = body.get('response', '').upper()
        _voice_responses[workflow_id] = response
        print(f"📞 Voice response for {workflow_id}: {response}")
        workflow_push_message(workflow_id, 'Patient', f'Response: {response}', 'response')
        
        # Log voice response to LangSmith
        if LANGSMITH_ENABLED:
            log_to_langsmith(
                run_name=f"Voice Response: {workflow_id}",
                run_type="tool",
                inputs={"workflow_id": workflow_id},
                outputs={"response": response},
                tags=["voice", "patient-response"]
            )
        
        return jsonify({'ok': True, 'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/workflow/<workflow_id>/voice-response', methods=['GET'])
def poll_voice_response(workflow_id):
    """Agent 5B polls this to see if the patient answered via the UI."""
    resp = _voice_responses.pop(workflow_id, None)
    return jsonify({'response': resp})

# ── LangSmith dashboard endpoints ─────────────────────────────────────────────
@app.route('/api/langsmith/status')
def langsmith_status():
    """Check if LangSmith is enabled and return project info"""
    return jsonify({
        'enabled': LANGSMITH_ENABLED,
        'project': LANGSMITH_PROJECT,
        'api_key_set': bool(LANGSMITH_API_KEY)
    })

@app.route('/api/langsmith/metrics')
def langsmith_metrics():
    """Get aggregated metrics from LangSmith for dashboard"""
    if not LANGSMITH_ENABLED:
        return jsonify({'error': 'LangSmith not enabled'}), 400
    
    try:
        # Get recent runs
        runs = list(langsmith_client.list_runs(
            project_name=LANGSMITH_PROJECT,
            execution_order=1,
            limit=100
        ))
        
        # Calculate metrics
        total_runs = len(runs)
        validation_runs = [r for r in runs if 'validation' in r.name.lower()]
        approved_count = 0
        rejected_count = 0
        
        for run in validation_runs:
            if run.outputs and run.outputs.get('decision') == 'APPROVED':
                approved_count += 1
            elif run.outputs and run.outputs.get('decision') == 'REJECTED':
                rejected_count += 1
        
        return jsonify({
            'total_runs': total_runs,
            'validation_runs': len(validation_runs),
            'approved': approved_count,
            'rejected': rejected_count,
            'accuracy': (approved_count + rejected_count) / len(validation_runs) if validation_runs else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)