"""
ticket_notifier.py
──────────────────────────────────────────────────────────────────────────────
Professional email ticket system for Healthcare Claims.

Lifecycle
─────────
  1. Agent 4 REJECTS a claim  →  agent5b_voice() is triggered
  2. On entry to Agent 5B     →  raise_ticket()   sends an OPEN ticket email
  3. After voice conversation:
       • Patient says YES      →  close_ticket()  sends a RESOLVED email
       • Patient says NO       →  close_ticket()  sends an ESCALATED email
       • Correction still fails→  close_ticket()  sends a FAILED email

Integration (drop into langgraph_workflow.py)
─────────────────────────────────────────────
  from ticket_notifier import raise_ticket, close_ticket

  # Inside agent5b_voice() — right after building voice_claim:
  ticket_id = raise_ticket(state, decision)

  # After intent is resolved, pass ticket_id to close_ticket():
  close_ticket(ticket_id, state, decision, outcome="RESOLVED"|"ESCALATED"|"FAILED")

Environment variables required (.env)
──────────────────────────────────────
  TICKET_EMAIL_FROM      = claims-system@yourorg.com
  TICKET_EMAIL_TO        = you@yourorg.com          # comma-separated for multiple
  SMTP_HOST              = smtp.gmail.com
  SMTP_PORT              = 587
  SMTP_USER              = your-smtp-user@gmail.com
  SMTP_PASSWORD          = your-app-password
  TICKET_EMAIL_CC        = manager@yourorg.com      # optional
"""

import os
import json
import sqlite3
import smtplib
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

SMTP_HOST      = os.getenv("SMTP_HOST",           "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT",       "587"))
SMTP_USER      = os.getenv("SMTP_USER",           "")
SMTP_PASSWORD  = os.getenv("SMTP_PASSWORD",       "")
FROM_ADDR      = os.getenv("TICKET_EMAIL_FROM",   SMTP_USER)
TO_ADDRS_RAW   = os.getenv("TICKET_EMAIL_TO",     "")
CC_ADDRS_RAW   = os.getenv("TICKET_EMAIL_CC",     "")
DB_PATH        = os.getenv("DB_PATH",             "data/claims.db")

# ── Priority mapping ──────────────────────────────────────────────────────────

def _derive_priority(decision: dict) -> str:
    confidence = str(decision.get("confidence", "")).lower()
    alts        = decision.get("suggested_alternatives", [])
    reasoning   = str(decision.get("reasoning", "")).lower()

    if "high" in confidence or "critical" in reasoning or "urgent" in reasoning:
        return "CRITICAL"
    if alts:
        return "HIGH"
    if "medium" in confidence or "moderate" in reasoning:
        return "MEDIUM"
    return "LOW"

PRIORITY_COLORS = {
    "CRITICAL": "#C0392B",
    "HIGH":     "#E67E22",
    "MEDIUM":   "#2980B9",
    "LOW":      "#27AE60",
}

PRIORITY_BADGES = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🔵",
    "LOW":      "🟢",
}

# ── DB helpers ────────────────────────────────────────────────────────────────

def _ensure_tickets_table():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id           TEXT PRIMARY KEY,
            claim_id     TEXT,
            patient_id   TEXT,
            issue        TEXT,
            priority     TEXT,
            status       TEXT,
            outcome      TEXT,
            created_at   TIMESTAMP,
            resolved_at  TIMESTAMP,
            open_email_sent    INTEGER DEFAULT 0,
            closed_email_sent  INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()


def _upsert_ticket(ticket_id: str, claim_id: str, patient_id: str,
                   issue: str, priority: str, status: str,
                   outcome: str = None, resolved_at: str = None):
    _ensure_tickets_table()
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    existing = cursor.execute(
        "SELECT id FROM tickets WHERE id=?", (ticket_id,)
    ).fetchone()

    if existing:
        cursor.execute('''
            UPDATE tickets
            SET status=?, outcome=?, resolved_at=?,
                closed_email_sent = closed_email_sent
            WHERE id=?
        ''', (status, outcome, resolved_at, ticket_id))
    else:
        cursor.execute('''
            INSERT INTO tickets
            (id, claim_id, patient_id, issue, priority, status, outcome, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (ticket_id, claim_id, patient_id, issue, priority, status,
              outcome, datetime.now().isoformat()))

    conn.commit()
    conn.close()


# ── Email transport ───────────────────────────────────────────────────────────

def _send_email(subject: str, html_body: str, plain_body: str) -> bool:
    """Send a multi-part HTML + plain-text email. Returns True on success."""
    to_list = [a.strip() for a in TO_ADDRS_RAW.split(",") if a.strip()]
    cc_list = [a.strip() for a in CC_ADDRS_RAW.split(",") if a.strip()]

    if not to_list:
        logger.warning("TICKET_EMAIL_TO is not set — skipping email delivery")
        return False

    msg              = MIMEMultipart("alternative")
    msg["Subject"]   = subject
    msg["From"]      = FROM_ADDR
    msg["To"]        = ", ".join(to_list)
    if cc_list:
        msg["Cc"]    = ", ".join(cc_list)

    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body,  "html"))

    recipients = to_list + cc_list

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(FROM_ADDR, recipients, msg.as_string())
        logger.info("Ticket email sent → %s", recipients)
        return True
    except Exception as exc:
        logger.error("Failed to send ticket email: %s", exc)
        return False


# ── HTML helpers ──────────────────────────────────────────────────────────────

_CSS = """
  body{font-family:'Segoe UI',Arial,sans-serif;background:#f4f6f9;margin:0;padding:0}
  .wrapper{max-width:680px;margin:32px auto;background:#fff;border-radius:10px;
           box-shadow:0 2px 12px rgba(0,0,0,.10);overflow:hidden}
  .header{padding:28px 36px;color:#fff}
  .header h1{margin:0;font-size:22px;font-weight:700;letter-spacing:.5px}
  .header p{margin:6px 0 0;font-size:13px;opacity:.85}
  .body{padding:28px 36px}
  .meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px}
  .meta-card{background:#f8f9fb;border-radius:8px;padding:14px 18px;
             border-left:4px solid #3498db}
  .meta-card .label{font-size:11px;color:#7f8c8d;text-transform:uppercase;
                    letter-spacing:.6px;margin-bottom:4px}
  .meta-card .value{font-size:15px;font-weight:600;color:#2c3e50}
  .section{margin-bottom:22px}
  .section-title{font-size:13px;font-weight:700;color:#7f8c8d;
                 text-transform:uppercase;letter-spacing:.7px;
                 margin-bottom:10px;padding-bottom:6px;
                 border-bottom:1px solid #ecf0f1}
  .detail-row{display:flex;gap:10px;margin-bottom:8px;font-size:14px;color:#34495e}
  .detail-row .key{min-width:190px;font-weight:600;color:#2c3e50}
  .badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;
         font-weight:700;color:#fff;letter-spacing:.4px}
  .alert-box{border-radius:8px;padding:16px 20px;margin-bottom:20px;
             font-size:14px;line-height:1.6}
  .alert-error  {background:#fdedec;border-left:4px solid #e74c3c;color:#922b21}
  .alert-warn   {background:#fef9e7;border-left:4px solid #f39c12;color:#7d6608}
  .alert-success{background:#eafaf1;border-left:4px solid #27ae60;color:#1e8449}
  .alt-pill{display:inline-block;background:#ebf5fb;color:#1a5276;
            border:1px solid #aed6f1;border-radius:14px;padding:4px 12px;
            font-size:13px;margin:3px 4px 3px 0}
  .timeline{border-left:3px solid #3498db;padding-left:20px;margin-top:8px}
  .tl-step{position:relative;margin-bottom:14px;font-size:14px;color:#34495e}
  .tl-step::before{content:'';position:absolute;left:-26px;top:4px;
                   width:10px;height:10px;border-radius:50%;background:#3498db}
  .footer{background:#f8f9fb;padding:18px 36px;font-size:12px;color:#95a5a6;
          border-top:1px solid #ecf0f1;text-align:center}
  .divider{border:none;border-top:1px solid #ecf0f1;margin:20px 0}
"""

def _html_wrap(header_color: str, header_icon: str, header_title: str,
               header_sub: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><style>{_CSS}</style></head>
<body>
<div class="wrapper">
  <div class="header" style="background:{header_color}">
    <h1>{header_icon} {header_title}</h1>
    <p>{header_sub}</p>
  </div>
  <div class="body">
    {body_html}
  </div>
  <div class="footer">
    Healthcare Claims AI System &nbsp;|&nbsp; {datetime.now().strftime('%d %b %Y, %H:%M')} UTC
    &nbsp;|&nbsp; This is an automated notification — do not reply directly.
  </div>
</div>
</body>
</html>"""


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def raise_ticket(state: dict, decision: dict) -> str:
    """
    Called when Agent 5B is triggered (claim was REJECTED by Agent 4).

    Parameters
    ──────────
    state    : LangGraph AgentState dict
    decision : validation_result dict from Agent 4

    Returns
    ───────
    ticket_id : str  (e.g. "TKT20260315143022")
    """
    ticket_id  = f"TKT{datetime.now().strftime('%Y%m%d%H%M%S')}"
    claim_id   = decision.get("claim_id",   state.get("claim_id",   "UNKNOWN"))
    patient_id = decision.get("patient_id", state.get("patient_id", "UNKNOWN"))
    proc_code  = decision.get("procedure_code", state.get("procedure_code", "N/A"))
    reasoning  = decision.get("reasoning",  "No reason provided")
    confidence = decision.get("confidence", "N/A")
    alts       = decision.get("suggested_alternatives", [])
    citations  = decision.get("policy_citations",       [])
    next_steps = decision.get("next_steps",             [])
    priority   = _derive_priority(decision)
    p_color    = PRIORITY_COLORS[priority]
    p_badge    = PRIORITY_BADGES[priority]

    # ── persist to DB ──────────────────────────────────────────────────────
    _upsert_ticket(ticket_id, claim_id, patient_id,
                   f"Claim REJECTED — {reasoning[:120]}", priority, "OPEN")

    # ── build alternative pills ────────────────────────────────────────────
    def _format_alt(a):
        if isinstance(a, dict):
            return a.get("code") or a.get("description") or str(a)
        return str(a)

    alt_pills = "".join(
        f'<span class="alt-pill">✦ {_format_alt(a)}</span>' for a in alts
    ) or "<span style='color:#7f8c8d;font-size:13px'>No alternatives suggested</span>"

    cit_html = "".join(
        f'<div class="detail-row"><span>📌</span><span>{c}</span></div>'
        for c in citations
    ) or "<span style='color:#7f8c8d;font-size:13px'>None referenced</span>"

    steps_html = ""
    if isinstance(next_steps, list):
        steps_html = "".join(f'<div class="tl-step">{s}</div>' for s in next_steps)
    elif next_steps:
        steps_html = f'<div class="tl-step">{next_steps}</div>'

    body = f"""
    <!-- Meta grid -->
    <div class="meta-grid">
      <div class="meta-card" style="border-color:{p_color}">
        <div class="label">Ticket ID</div>
        <div class="value">{ticket_id}</div>
      </div>
      <div class="meta-card">
        <div class="label">Claim ID</div>
        <div class="value">{claim_id}</div>
      </div>
      <div class="meta-card">
        <div class="label">Patient ID</div>
        <div class="value">{patient_id}</div>
      </div>
      <div class="meta-card">
        <div class="label">Procedure Code</div>
        <div class="value">{proc_code}</div>
      </div>
    </div>

    <!-- Status row -->
    <div class="detail-row">
      <span class="key">Status</span>
      <span class="badge" style="background:#e74c3c">OPEN</span>
    </div>
    <div class="detail-row">
      <span class="key">Priority</span>
      <span class="badge" style="background:{p_color}">{p_badge} {priority}</span>
    </div>
    <div class="detail-row">
      <span class="key">AI Confidence</span>
      <span class="value" style="font-size:14px">{confidence}</span>
    </div>
    <div class="detail-row">
      <span class="key">Raised At</span>
      <span>{datetime.now().strftime('%d %b %Y, %H:%M:%S')} UTC</span>
    </div>

    <hr class="divider">

    <!-- Rejection reason -->
    <div class="section">
      <div class="section-title">❌ What Went Wrong</div>
      <div class="alert-box alert-error">
        {reasoning}
      </div>
    </div>

    <!-- Policy citations -->
    <div class="section">
      <div class="section-title">📋 Policy Citations</div>
      {cit_html}
    </div>

    <!-- Suggestions -->
    <div class="section">
      <div class="section-title">💡 Suggested Corrections / Alternatives</div>
      <div style="margin-top:4px">{alt_pills}</div>
    </div>

    <!-- Next Steps -->
    <div class="section">
      <div class="section-title">🗺 Recommended Next Steps</div>
      <div class="timeline">
        {steps_html if steps_html else '<div class="tl-step">Await patient voice response via Agent 5B</div>'}
        <div class="tl-step">Patient will be contacted via voice agent for correction approval</div>
        <div class="tl-step">Ticket will auto-close once outcome is confirmed</div>
      </div>
    </div>

    <!-- Action note -->
    <div class="alert-box alert-warn">
      <strong>⚠ Action Required:</strong> Voice Agent (Agent 5B) is now contacting
      the patient. This ticket will be automatically updated once the patient responds.
    </div>
    """

    html = _html_wrap(
        header_color="#c0392b",
        header_icon="🎫",
        header_title=f"New Claim Ticket — {ticket_id}",
        header_sub=f"Claim {claim_id} was REJECTED and requires patient confirmation",
        body_html=body,
    )

    plain = f"""
NEW CLAIM TICKET — {ticket_id}
{'='*60}
Ticket ID     : {ticket_id}
Claim ID      : {claim_id}
Patient ID    : {patient_id}
Procedure     : {proc_code}
Status        : OPEN
Priority      : {priority}
Confidence    : {confidence}
Raised At     : {datetime.now().strftime('%d %b %Y, %H:%M:%S')} UTC

WHAT WENT WRONG
{'-'*40}
{reasoning}

POLICY CITATIONS
{'-'*40}
{chr(10).join(f'  • {c}' for c in citations) or '  None referenced'}

SUGGESTED CORRECTIONS / ALTERNATIVES
{'-'*40}
{chr(10).join(f'  • {_format_alt(a)}' for a in alts) or '  No alternatives suggested'}

NEXT STEPS
{'-'*40}
  • Voice agent (Agent 5B) is contacting the patient
  • Ticket will auto-close once outcome is confirmed

This is an automated notification from the Healthcare Claims AI System.
"""

    subject = f"[TICKET OPENED] {ticket_id} | Claim {claim_id} Rejected — {priority} Priority"
    _send_email(subject, html, plain)

    logger.info("Ticket %s raised for claim %s", ticket_id, claim_id)
    return ticket_id


def close_ticket(ticket_id: str, state: dict, decision: dict,
                 outcome: str, corrected_code: str = None,
                 extra_notes: str = None) -> bool:
    """
    Called after the voice agent interaction is complete.

    Parameters
    ──────────
    ticket_id      : from raise_ticket()
    state          : LangGraph AgentState dict
    decision       : original validation_result from Agent 4
    outcome        : one of "RESOLVED" | "ESCALATED" | "FAILED"
                     RESOLVED  — patient said YES, corrected claim approved
                     ESCALATED — patient said NO, escalated for manual review
                     FAILED    — patient said YES but corrected claim still rejected
    corrected_code : the alternative code used (for RESOLVED / FAILED)
    extra_notes    : any additional context to append

    Returns
    ───────
    True if email was dispatched successfully.
    """
    claim_id   = decision.get("claim_id",   state.get("claim_id",   "UNKNOWN"))
    patient_id = decision.get("patient_id", state.get("patient_id", "UNKNOWN"))
    proc_code  = decision.get("procedure_code", state.get("procedure_code", "N/A"))
    reasoning  = decision.get("reasoning",  "No reason provided")
    alts       = decision.get("suggested_alternatives", [])
    confidence = decision.get("confidence", "N/A")
    priority   = _derive_priority(decision)
    p_color    = PRIORITY_COLORS[priority]
    p_badge    = PRIORITY_BADGES[priority]
    resolved_ts = datetime.now().isoformat()

    def _format_alt(a):
        if isinstance(a, dict):
            return a.get("code") or a.get("description") or str(a)
        return str(a)

    # ── Update DB ──────────────────────────────────────────────────────────
    _upsert_ticket(ticket_id, claim_id, patient_id,
                   f"Claim REJECTED — {reasoning[:120]}",
                   priority, "CLOSED", outcome, resolved_ts)

    # ── Outcome-specific copy ──────────────────────────────────────────────
    if outcome == "RESOLVED":
        header_color = "#1e8449"
        header_icon  = "✅"
        header_title = f"Ticket Closed — {ticket_id}"
        header_sub   = f"Claim {claim_id} successfully corrected and approved"
        status_badge = ("CLOSED — RESOLVED", "#27ae60")
        outcome_html = f"""
        <div class="alert-box alert-success">
          <strong>✅ Outcome: RESOLVED</strong><br>
          The patient confirmed correction via voice agent. The claim was
          re-submitted with corrected procedure code
          <strong>{corrected_code or 'N/A'}</strong> and has been
          <strong>APPROVED</strong>.
        </div>"""
        plain_outcome = f"OUTCOME: RESOLVED\nCorrected code {corrected_code or 'N/A'} was approved."

    elif outcome == "ESCALATED":
        header_color = "#d35400"
        header_icon  = "📌"
        header_title = f"Ticket Escalated — {ticket_id}"
        header_sub   = f"Claim {claim_id} — patient declined correction, manual review needed"
        status_badge = ("CLOSED — ESCALATED", "#e67e22")
        outcome_html = f"""
        <div class="alert-box alert-warn">
          <strong>📌 Outcome: ESCALATED</strong><br>
          The patient refused the correction via voice agent.
          This claim has been escalated for <strong>prior-authorization
          review</strong> by the clinical team. No further automated
          action will be taken.
        </div>"""
        plain_outcome = "OUTCOME: ESCALATED\nPatient refused correction. Manual prior-auth review required."

    else:  # FAILED
        header_color = "#8e44ad"
        header_icon  = "⚠️"
        header_title = f"Ticket Closed — {ticket_id}"
        header_sub   = f"Claim {claim_id} — correction approved by patient but still rejected"
        status_badge = ("CLOSED — FAILED", "#8e44ad")
        outcome_html = f"""
        <div class="alert-box alert-error">
          <strong>⚠️ Outcome: CORRECTION FAILED</strong><br>
          The patient agreed to correction and the claim was re-submitted
          with code <strong>{corrected_code or 'N/A'}</strong>, but the
          corrected claim was <strong>still rejected</strong> by the
          validation engine. Manual intervention is required.
        </div>"""
        plain_outcome = (
            f"OUTCOME: FAILED\nPatient agreed but corrected claim "
            f"({corrected_code or 'N/A'}) was still rejected. Manual review needed."
        )

    alt_pills = "".join(
        f'<span class="alt-pill">✦ {_format_alt(a)}</span>' for a in alts
    ) or "<span style='color:#7f8c8d;font-size:13px'>None</span>"

    body = f"""
    <!-- Meta grid -->
    <div class="meta-grid">
      <div class="meta-card" style="border-color:{p_color}">
        <div class="label">Ticket ID</div>
        <div class="value">{ticket_id}</div>
      </div>
      <div class="meta-card">
        <div class="label">Claim ID</div>
        <div class="value">{claim_id}</div>
      </div>
      <div class="meta-card">
        <div class="label">Patient ID</div>
        <div class="value">{patient_id}</div>
      </div>
      <div class="meta-card">
        <div class="label">Procedure Code</div>
        <div class="value">{proc_code}</div>
      </div>
    </div>

    <!-- Status -->
    <div class="detail-row">
      <span class="key">Final Status</span>
      <span class="badge" style="background:{status_badge[1]}">{status_badge[0]}</span>
    </div>
    <div class="detail-row">
      <span class="key">Priority</span>
      <span class="badge" style="background:{p_color}">{p_badge} {priority}</span>
    </div>
    <div class="detail-row">
      <span class="key">Original AI Confidence</span>
      <span>{confidence}</span>
    </div>
    <div class="detail-row">
      <span class="key">Resolved At</span>
      <span>{datetime.now().strftime('%d %b %Y, %H:%M:%S')} UTC</span>
    </div>

    <hr class="divider">

    {outcome_html}

    <!-- Original rejection reason -->
    <div class="section">
      <div class="section-title">❌ Original Rejection Reason</div>
      <div class="alert-box alert-error" style="font-size:13px">
        {reasoning}
      </div>
    </div>

    <!-- Suggested alternatives -->
    <div class="section">
      <div class="section-title">💡 Suggested Alternatives (from Agent 4)</div>
      <div style="margin-top:4px">{alt_pills}</div>
    </div>

    {f'''
    <!-- Extra notes -->
    <div class="section">
      <div class="section-title">📝 Additional Notes</div>
      <div style="font-size:14px;color:#34495e;line-height:1.6">{extra_notes}</div>
    </div>
    ''' if extra_notes else ''}

    <!-- Resolution timeline -->
    <div class="section">
      <div class="section-title">🗓 Resolution Timeline</div>
      <div class="timeline">
        <div class="tl-step">Claim {claim_id} submitted for validation</div>
        <div class="tl-step">Agent 4 (Groq LLM) rejected claim — reason logged</div>
        <div class="tl-step">Ticket {ticket_id} raised, Agent 5B voice agent activated</div>
        <div class="tl-step">Patient contacted via Deepgram TTS / STT pipeline</div>
        <div class="tl-step"><strong>Ticket closed — {outcome}</strong></div>
      </div>
    </div>
    """

    html = _html_wrap(
        header_color=header_color,
        header_icon=header_icon,
        header_title=header_title,
        header_sub=header_sub,
        body_html=body,
    )

    plain = f"""
TICKET CLOSED — {ticket_id}
{'='*60}
Ticket ID     : {ticket_id}
Claim ID      : {claim_id}
Patient ID    : {patient_id}
Procedure     : {proc_code}
Priority      : {priority}
Resolved At   : {datetime.now().strftime('%d %b %Y, %H:%M:%S')} UTC

{plain_outcome}

ORIGINAL REJECTION REASON
{'-'*40}
{reasoning}

SUGGESTED ALTERNATIVES
{'-'*40}
{chr(10).join(f'  • {_format_alt(a)}' for a in alts) or '  None'}

{f'ADDITIONAL NOTES{chr(10)}{"-"*40}{chr(10)}{extra_notes}' if extra_notes else ''}

This is an automated notification from the Healthcare Claims AI System.
"""

    subject = (
        f"[TICKET CLOSED — {outcome}] {ticket_id} | "
        f"Claim {claim_id} | {priority} Priority"
    )
    ok = _send_email(subject, html, plain)
    logger.info("Ticket %s closed with outcome=%s, email=%s", ticket_id, outcome, ok)
    return ok