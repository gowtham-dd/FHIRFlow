"""
AGENT 4: Claims Validator + Reasoning Layer
Purpose: Validate claims against patient records and policy rules
Input: Claim data + FHIR patient records + Pinecone policies
Output: APPROVED/REJECTED decision with reasoning

This uses REAL Groq Llama 3.1 model via API - NOT simulation!
Each claim is sent to Groq for actual AI-powered validation.
"""

import os
import json
import sqlite3
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# LangChain & Groq imports - REAL API calls to Groq
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import your helper
import sys
sys.path.append('../..')
try:
    from helper.helper import download_embeddings
except ImportError:
    # Define fallback if helper not available
    def download_embeddings():
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent4_claims_validator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ClaimsValidator:
    """
    Validates claims against patient records and policy rules
    Uses REAL Groq Llama 3.1 API for decisions - NOT simulation
    """
    
    def __init__(self, 
                 db_path: str = "../agent3/fhir_procedure_db.sqlite",
                 pinecone_index: str = "fhirdb"):
        
        # Connect to FHIR database (from Agent 3)
        self.db_path = db_path
        self._verify_database()
        
        # Initialize embeddings and Pinecone
        try:
            self.embeddings = download_embeddings()
            self.vectorstore = PineconeVectorStore(
                index_name=pinecone_index,
                embedding=self.embeddings
            )
            self.pinecone_available = True
            logger.info("  Pinecone connected - using real policy vectors")
        except Exception as e:
            logger.warning(f"⚠️ Pinecone not available: {e}")
            self.pinecone_available = False
        
        # Initialize Groq for REAL inference (not simulation)
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("❌ GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.1-8b-instant",  # REAL Llama 3.1 model on Groq
            temperature=0.1,  # Low for consistent decisions
            max_tokens=1000
        )
        logger.info("  Groq Llama 3.1 initialized - making REAL API calls")
        
        # Statistics tracking
        self.stats = {
            "claims_processed": 0,
            "approved": 0,
            "rejected": 0,
            "pending_review": 0,
            "error": 0
        }
        
        # Setup prompts
        self.setup_prompts()
    
    def _verify_database(self):
        """Verify database exists and has required tables"""
        if not os.path.exists(self.db_path):
            logger.warning(f"Database not found: {self.db_path}")
            logger.info("Creating new database with sample data...")
            self._create_sample_database()
    
    def _create_sample_database(self):
        """Create a sample database if none exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create minimal tables for validation
        cursor.executescript('''
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                date_of_birth TEXT,
                gender TEXT
            );
            
            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                code TEXT,
                subject_patient_id TEXT,
                performedDateTime TEXT,
                status TEXT,
                note TEXT,
                FOREIGN KEY (subject_patient_id) REFERENCES patients(id)
            );
            
            CREATE TABLE IF NOT EXISTS procedure_policy_flags (
                procedure_id TEXT,
                policy_code TEXT,
                requirement TEXT,
                effective_date TEXT,
                applied_at TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT,
                action TEXT,
                procedure_id TEXT,
                policy_code TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Add sample patients
        patients = [
            ('pat001', 'John', 'Smith', '1965-03-15', 'M'),
            ('pat002', 'Jane', 'Doe', '1972-07-22', 'F'),
            ('pat003', 'Robert', 'Johnson', '1958-11-30', 'M'),
            ('pat004', 'Mary', 'Williams', '1980-05-12', 'F'),
            ('pat005', 'David', 'Brown', '1995-09-08', 'M'),
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO patients (id, first_name, last_name, date_of_birth, gender)
            VALUES (?, ?, ?, ?, ?)
        ''', patients)
        
        # Add sample procedures
        procedures = [
            ('proc001', 'J3420', 'pat001', '2026-02-15T09:30:00Z', 'completed', 'Routine B12 injection'),
            ('proc002', 'J3420', 'pat001', '2026-03-01T10:00:00Z', 'completed', 'Follow-up B12 injection'),
            ('proc003', '97110', 'pat002', '2026-02-20T14:00:00Z', 'completed', 'Physical therapy session'),
            ('proc004', '82652', 'pat003', '2026-02-10T08:00:00Z', 'completed', 'Vitamin D test'),
            ('proc005', 'J3420', 'pat004', '2026-03-05T11:00:00Z', 'completed', 'B12 injection'),
            ('proc006', '97110', 'pat002', '2026-03-10T15:00:00Z', 'completed', 'Physical therapy follow-up'),
            ('proc007', '97110', 'pat002', '2026-03-15T15:00:00Z', 'completed', 'Physical therapy session'),
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO procedures (id, code, subject_patient_id, performedDateTime, status, note)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', procedures)
        
        # Add policy flags (from Agent 3 updates)
        policy_flags = [
            ('proc002', 'J3420', '{"requirement": "prior_auth_required >1000mcg", "impact": "high"}', '2026-03-01', datetime.now().isoformat()),
            ('proc006', '97110', '{"requirement": "visit_limit_12_per_year", "impact": "medium"}', '2026-03-01', datetime.now().isoformat()),
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO procedure_policy_flags 
            (procedure_id, policy_code, requirement, effective_date, applied_at)
            VALUES (?, ?, ?, ?, ?)
        ''', policy_flags)
        
        conn.commit()
        conn.close()
        
        logger.info("Sample database created successfully")
    
    def setup_prompts(self):
        """Setup prompts for claim validation"""
        
        # Main validation prompt
        self.validation_prompt = PromptTemplate(
            template="""You are a healthcare claims validator. Determine if this claim should be APPROVED or REJECTED based on:

PATIENT INFORMATION:
{patient_info}

CLAIM DETAILS:
{claim_details}

RELEVANT POLICY RULES:
{policy_rules}

PREVIOUS PROCEDURES (last 12 months):
{previous_procedures}

POLICY FLAGS ON PATIENT:
{policy_flags}

Analyze the claim against policy rules and return a JSON decision with exactly these fields:
- decision: either "APPROVED" or "REJECTED"
- confidence: either "high", "medium", or "low"
- reasoning: clear explanation of why approved/rejected
- policy_citations: list of specific policy rules applied
- suggested_alternatives: list of alternative codes if rejected (empty list if approved)
- next_steps: what patient/provider should do next

Focus on:
1. Prior authorization requirements
2. Dose/frequency limits
3. Medical necessity criteria
4. Coding accuracy
5. Patient history

Be specific and cite exact policy rules.

Your response must be valid JSON only, no other text.
""",
            input_variables=["patient_info", "claim_details", "policy_rules", 
                           "previous_procedures", "policy_flags"]
        )
        
        # Parser for structured output
        self.parser = JsonOutputParser()
    
    def get_patient_info(self, patient_id: str) -> Dict:
        """Fetch patient information from FHIR database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT * FROM patients WHERE id = ?
        ''', (patient_id,))
        
        patient = cursor.fetchone()
        conn.close()
        
        if patient:
            return dict(patient)
        return {"id": patient_id, "error": "Patient not found"}
    
    def get_previous_procedures(self, patient_id: str, code: str = None, months: int = 12) -> List[Dict]:
        """Get patient's previous procedures for frequency checking"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        query = '''
            SELECT * FROM procedures 
            WHERE subject_patient_id = ?
        '''
        params = [patient_id]
        
        if code:
            query += " AND code = ?"
            params.append(code)
        
        query += " ORDER BY performedDateTime DESC LIMIT 20"
        
        cursor = conn.execute(query, params)
        procedures = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return procedures
    
    def get_policy_flags(self, patient_id: str) -> List[Dict]:
        """Get policy flags applied to this patient's procedures"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT ppf.*, p.code as procedure_code, p.performedDateTime
            FROM procedure_policy_flags ppf
            JOIN procedures p ON ppf.procedure_id = p.id
            WHERE p.subject_patient_id = ?
            ORDER BY ppf.applied_at DESC
        ''', (patient_id,))
        
        flags = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return flags
    
    def query_policy_rules(self, claim_details: Dict) -> str:
        """Query Pinecone for relevant policy rules"""
        if not self.pinecone_available:
            return "Pinecone not available. Using default policy rules."
        
        try:
            # Create search query from claim
            search_query = f"""
            Policy for {claim_details.get('code', 'unknown')} 
            {claim_details.get('description', '')}
            Prior authorization requirements
            Coverage limits
            """
            
            # Search Pinecone
            docs = self.vectorstore.similarity_search(
                search_query,
                k=3
            )
            
            if docs:
                return "\n\n---\n\n".join([doc.page_content for doc in docs])
            else:
                return "No specific policy rules found in vector store."
                
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return f"Error querying policy rules: {str(e)}"
    
    def validate_claim(self, claim: Dict) -> Dict:
        """
        Validate a single claim using REAL Groq API
        This makes an actual API call to Groq's Llama 3.1 model
        """
        self.stats["claims_processed"] += 1
        
        try:
            # Extract claim details
            patient_id = claim.get('patient_id')
            procedure_code = claim.get('code')
            
            if not patient_id:
                raise ValueError("Missing patient_id in claim")
            
            # Get all relevant data
            patient_info = self.get_patient_info(patient_id)
            previous_procedures = self.get_previous_procedures(patient_id, procedure_code)
            policy_flags = self.get_policy_flags(patient_id)
            policy_rules = self.query_policy_rules(claim)
            
            # Format for prompt
            patient_info_str = json.dumps(patient_info, indent=2)
            claim_details_str = json.dumps(claim, indent=2)
            prev_proc_str = json.dumps(previous_procedures, indent=2) if previous_procedures else "None"
            policy_flags_str = json.dumps(policy_flags, indent=2) if policy_flags else "None"
            
            # Make REAL API call to Groq
            logger.info(f"  Calling Groq API for claim {claim.get('id')}...")
            response = self.llm.invoke(
                self.validation_prompt.format(
                    patient_info=patient_info_str,
                    claim_details=claim_details_str,
                    policy_rules=policy_rules,
                    previous_procedures=prev_proc_str,
                    policy_flags=policy_flags_str
                )
            )
            
            # Parse response
            try:
                # Try to parse entire response as JSON
                decision = json.loads(response.content)
                logger.info(f"  Groq API responded with valid JSON")
            except:
                # Fallback: extract JSON from response
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group(0))
                    logger.info(f"  Extracted JSON from Groq response")
                else:
                    # Create a structured decision from text
                    content = response.content.lower()
                    if "approve" in content:
                        decision_text = "APPROVED"
                    elif "reject" in content:
                        decision_text = "REJECTED"
                    else:
                        decision_text = "PENDING_REVIEW"
                    
                    decision = {
                        "decision": decision_text,
                        "confidence": "low",
                        "reasoning": response.content[:500],
                        "policy_citations": [],
                        "suggested_alternatives": [],
                        "next_steps": "Manual review required"
                    }
                    logger.warning(f"⚠️ Using fallback parsing for Groq response")
            
            # Ensure required fields exist
            required_fields = ["decision", "confidence", "reasoning", "policy_citations", "suggested_alternatives", "next_steps"]
            for field in required_fields:
                if field not in decision:
                    if field == "policy_citations" or field == "suggested_alternatives":
                        decision[field] = []
                    else:
                        decision[field] = "Not provided"
            
            # FIX: Clean up suggested_alternatives to ensure they're strings
            if 'suggested_alternatives' in decision and decision['suggested_alternatives']:
                cleaned_alternatives = []
                for alt in decision['suggested_alternatives']:
                    if isinstance(alt, dict):
                        # If it's a dict with code, use that
                        if 'code' in alt:
                            cleaned_alternatives.append(alt['code'])
                        else:
                            # Otherwise stringify the whole dict
                            cleaned_alternatives.append(str(alt))
                    else:
                        cleaned_alternatives.append(str(alt))
                decision['suggested_alternatives'] = cleaned_alternatives
            
            # Add metadata
            decision["claim_id"] = claim.get('id', 'unknown')
            decision["patient_id"] = patient_id
            decision["procedure_code"] = procedure_code
            decision["timestamp"] = datetime.now().isoformat()
            
            # Update stats
            if decision.get('decision') == 'APPROVED':
                self.stats["approved"] += 1
            elif decision.get('decision') == 'REJECTED':
                self.stats["rejected"] += 1
            else:
                self.stats["pending_review"] += 1
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Claim validation failed: {e}")
            self.stats["error"] += 1
            return {
                "decision": "ERROR",
                "confidence": "low",
                "reasoning": f"Validation error: {str(e)}",
                "policy_citations": [],
                "suggested_alternatives": [],
                "next_steps": "Technical support required",
                "claim_id": claim.get('id', 'unknown'),
                "patient_id": claim.get('patient_id', 'unknown'),
                "procedure_code": claim.get('code', 'unknown'),
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_validate_claims(self, claims: List[Dict]) -> List[Dict]:
        """Validate multiple claims"""
        results = []
        total = len(claims)
        
        for i, claim in enumerate(claims):
            logger.info(f"Validating claim {i+1}/{total}: {claim.get('id', 'unknown')}")
            result = self.validate_claim(claim)
            results.append(result)
            
            # Print progress
            decision = result.get('decision', 'UNKNOWN')
            reason_preview = result.get('reasoning', '')[:80] + "..." if len(result.get('reasoning', '')) > 80 else result.get('reasoning', '')
            print(f"   Claim {i+1}: {decision} - {reason_preview}")
        
        return results
    
    def generate_sample_claims(self) -> List[Dict]:
        """Generate sample claims for testing including wrong claims"""
        return [
            # ===== CORRECT CLAIMS (Should be APPROVED) =====
            {
                "id": "claim001_correct",
                "patient_id": "pat001",
                "code": "J3420",
                "description": "Vitamin B12 injection, 500mcg",
                "dose": 500,
                "units": 1,
                "date_of_service": "2026-03-15",
                "provider": "prat002",
                "diagnosis": "E53.8",
                "charge_amount": 85.00
            },
            {
                "id": "claim002_correct",
                "patient_id": "pat003",
                "code": "82652",
                "description": "Vitamin D test with deficiency",
                "date_of_service": "2026-03-12",
                "provider": "prat002",
                "diagnosis": "E55.9",  # Correct diagnosis for deficiency
                "charge_amount": 45.00
            },
            
            # ===== WRONG CLAIMS (Should be REJECTED) =====
            
            # Wrong Claim 1: Dose too high (1500mcg > 1000mcg limit)
            {
                "id": "claim003_wrong_dose",
                "patient_id": "pat001",
                "code": "J3420",
                "description": "Vitamin B12 injection, 1500mcg",
                "dose": 1500,
                "units": 1,
                "date_of_service": "2026-03-16",
                "provider": "prat002",
                "diagnosis": "E53.8",
                "charge_amount": 85.00
            },
            
            # Wrong Claim 2: Exceeds visit limit (already had 2 PT visits)
            {
                "id": "claim004_wrong_limit",
                "patient_id": "pat002",
                "code": "97110",
                "description": "Physical therapy session",
                "units": 1.5,
                "date_of_service": "2026-03-20",
                "provider": "prat001",
                "diagnosis": "M54.5",
                "charge_amount": 120.00
            },
            
            # Wrong Claim 3: Wrong diagnosis for Vitamin D test (no deficiency code)
            {
                "id": "claim005_wrong_diagnosis",
                "patient_id": "pat003",
                "code": "82652",
                "description": "Vitamin D test",
                "date_of_service": "2026-03-18",
                "provider": "prat002",
                "diagnosis": "Z13.6",  # Screening, not deficiency
                "charge_amount": 45.00
            },
            
            # Wrong Claim 4: Patient doesn't exist
            {
                "id": "claim006_wrong_patient",
                "patient_id": "pat999",
                "code": "J3420",
                "description": "Vitamin B12 injection, 500mcg",
                "dose": 500,
                "units": 1,
                "date_of_service": "2026-03-17",
                "provider": "prat002",
                "diagnosis": "E53.8",
                "charge_amount": 85.00
            },
            
            # Wrong Claim 5: Invalid procedure code
            {
                "id": "claim007_wrong_code",
                "patient_id": "pat001",
                "code": "INVALID",
                "description": "Unknown procedure",
                "dose": 1,
                "units": 1,
                "date_of_service": "2026-03-19",
                "provider": "prat002",
                "diagnosis": "E53.8",
                "charge_amount": 999.99
            },
            
            # Wrong Claim 6: Missing prior authorization for high dose
            {
                "id": "claim008_wrong_pa",
                "patient_id": "pat001",
                "code": "J3420",
                "description": "Vitamin B12 injection, 1200mcg",
                "dose": 1200,
                "units": 1,
                "date_of_service": "2026-03-21",
                "provider": "prat002",
                "diagnosis": "E53.8",
                "prior_auth": False,  # Missing PA
                "charge_amount": 85.00
            },
            
            # Wrong Claim 7: Duplicate claim (same as claim001)
            {
                "id": "claim009_duplicate",
                "patient_id": "pat001",
                "code": "J3420",
                "description": "Vitamin B12 injection, 500mcg",
                "dose": 500,
                "units": 1,
                "date_of_service": "2026-03-15",  # Same date as claim001
                "provider": "prat002",
                "diagnosis": "E53.8",
                "charge_amount": 85.00
            },
            
            # Wrong Claim 8: Excessive units
            {
                "id": "claim010_wrong_units",
                "patient_id": "pat001",
                "code": "J3420",
                "description": "Vitamin B12 injection",
                "dose": 500,
                "units": 10,  # 10 units in one day
                "date_of_service": "2026-03-22",
                "provider": "prat002",
                "diagnosis": "E53.8",
                "charge_amount": 850.00
            },
        ]
    
    def route_decision(self, decision: Dict) -> str:
        """
        Route the decision to the next agent
        Returns: "agent_5a" (approval), "agent_5b" (correction), or "agent_6" (escalation)
        """
        if decision.get('decision') == 'APPROVED':
            logger.info(f"Routing to Agent 5A (Approval Router)")
            return "agent_5a"
        elif decision.get('decision') == 'REJECTED':
            logger.info(f"Routing to Agent 5B (Correction + Voice)")
            return "agent_5b"
        else:
            logger.info(f"Routing to Agent 6 (Escalator)")
            return "agent_6"
    
    def run_demo(self):
        """Run a demo with sample claims including wrong ones"""
        print("\n" + "=" * 80)
        print("AGENT 4: CLAIMS VALIDATOR + REASONING LAYER")
        print("=" * 80)
        print("🔴 Using REAL Groq Llama 3.1 API - NOT simulation")
        print("=" * 80)
        
        # Generate sample claims
        claims = self.generate_sample_claims()
        
        # Separate correct and wrong claims for display
        correct_claims = [c for c in claims if 'correct' in c['id']]
        wrong_claims = [c for c in claims if 'wrong' in c['id']]
        
        print(f"\n📊 Test Dataset:")
        print(f"     Correct Claims: {len(correct_claims)}")
        print(f"    Wrong Claims: {len(wrong_claims)}")
        print(f"    Total Claims: {len(claims)}")
        
        print(f"\nValidating {len(claims)} sample claims with Groq API...\n")
        
        # Validate each claim
        results = []
        for i, claim in enumerate(claims):
            print(f"\n{'-' * 50}")
            print(f"Processing Claim {i+1}: {claim['id']}")
            print(f"   Patient: {claim['patient_id']}")
            print(f"   Procedure: {claim['code']} - {claim['description']}")
            if 'dose' in claim:
                print(f"   Dose: {claim['dose']}mcg")
            if claim.get('diagnosis'):
                print(f"   Diagnosis: {claim['diagnosis']}")
            print(f"{'-' * 50}")
            
            # Validate
            decision = self.validate_claim(claim)
            results.append(decision)
            
            # Show decision with formatting
            self.print_decision(decision)
            
            # Show routing
            next_agent = self.route_decision(decision)
            print(f"\nRouting to: {next_agent}")
        
        # Summary
        self.print_summary(results)
        
        # Save results
        output_file = f"agent4_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats,
                "results": results,
                "note": "These decisions were made by REAL Groq Llama 3.1 API"
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return results
    
    def print_decision(self, decision: Dict):
        """Print a decision"""
        decision_type = decision.get('decision', 'UNKNOWN')
        
        if decision_type == 'APPROVED':
            print(f"\n  DECISION: APPROVED (Groq AI)")
        elif decision_type == 'REJECTED':
            print(f"\n❌ DECISION: REJECTED (Groq AI)")
        else:
            print(f"\n⚠️ DECISION: {decision_type} (Groq AI)")
        
        print(f"   Confidence: {decision.get('confidence', 'N/A')}")
        print(f"\n   Reasoning:")
        print(f"   {decision.get('reasoning', 'N/A')}")
        
        if decision.get('policy_citations'):
            print(f"\n   Policy Citations:")
            for citation in decision['policy_citations']:
                print(f"   - {citation}")
        
        if decision.get('suggested_alternatives'):
            print(f"\n   Suggested Alternatives:")
            for alt in decision['suggested_alternatives']:
                print(f"   - {alt}")
        
        if decision.get('next_steps'):
            print(f"\n   Next Steps:")
            if isinstance(decision['next_steps'], list):
                for step in decision['next_steps']:
                    print(f"   - {step}")
            else:
                print(f"   {decision['next_steps']}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary of all decisions"""
        print("\n" + "=" * 80)
        print("AGENT 4 EXECUTION SUMMARY")
        print("=" * 80)
        
        approved = [r for r in results if r.get('decision') == 'APPROVED']
        rejected = [r for r in results if r.get('decision') == 'REJECTED']
        pending = [r for r in results if r.get('decision') not in ['APPROVED', 'REJECTED']]
        errors = [r for r in results if r.get('decision') == 'ERROR']
        
        print(f"\n📊 Statistics:")
        print(f"   Total Claims: {len(results)}")
        print(f"     Approved: {len(approved)}")
        print(f"   ❌ Rejected: {len(rejected)}")
        print(f"   ⏳ Pending Review: {len(pending)}")
        print(f"   ⚠️ Errors: {len(errors)}")
        
        if approved:
            print(f"\n  Approved Claims:")
            for r in approved:
                print(f"   - {r.get('claim_id')}: {r.get('procedure_code')}")
        
        if rejected:
            print(f"\n❌ Rejected Claims:")
            for r in rejected:
                print(f"   - {r.get('claim_id')}: {r.get('procedure_code')}")
                reason = r.get('reasoning', '')
                print(f"     Reason: {reason[:100]}..." if len(reason) > 100 else f"     Reason: {reason}")
                
                # FIX: Handle both string and dict in suggested_alternatives
                if r.get('suggested_alternatives'):
                    alternatives = []
                    for alt in r['suggested_alternatives']:
                        if isinstance(alt, dict):
                            # If it's a dict, try to get code or description
                            if 'code' in alt:
                                alternatives.append(alt['code'])
                            elif 'description' in alt:
                                alternatives.append(alt['description'])
                            else:
                                alternatives.append(str(alt))
                        else:
                            alternatives.append(str(alt))
                    print(f"     Suggested: {', '.join(alternatives)}")
        
        # Calculate accuracy
        expected_correct = len([c for c in results if 'correct' in c.get('claim_id', '')])
        actual_correct = len([r for r in approved if 'correct' in r.get('claim_id', '')])
        expected_wrong = len([c for c in results if 'wrong' in c.get('claim_id', '')])
        actual_wrong = len([r for r in rejected if 'wrong' in r.get('claim_id', '')])
        
        if expected_correct + expected_wrong > 0:
            accuracy = (actual_correct + actual_wrong) / (expected_correct + expected_wrong) * 100
            print(f"\n🎯 Accuracy:")
            print(f"   Correct claims correctly approved: {actual_correct}/{expected_correct}")
            print(f"   Wrong claims correctly rejected: {actual_wrong}/{expected_wrong}")
            print(f"   Overall accuracy: {accuracy:.1f}%")
        
        # Show routing summary
        print(f"\n  Routing Summary:")
        for r in results:
            next_agent = self.route_decision(r)
            icon = " " if next_agent == "agent_5a" else "❌" if next_agent == "agent_5b" else "⚠️"
            print(f"   {icon} {r.get('claim_id')}: {r.get('decision')} -> {next_agent}")


# Main execution
def main():
    """Entry point for Agent 4"""
    
    print("\n" + "=" * 80)
    print("INITIALIZING AGENT 4: CLAIMS VALIDATOR")
    print("=" * 80)
    
    # Check for API keys
    if not os.getenv("GROQ_API_KEY"):
        print("\n❌ ERROR: GROQ_API_KEY not found. Please add to .env file")
        print("Create a .env file with:")
        print("GROQ_API_KEY=your_api_key_here")
        print("PINECONE_API_KEY=your_key_here (optional)")
        return
    
    print(f"  GROQ_API_KEY found: {os.getenv('GROQ_API_KEY')[:5]}...")
    
    # Initialize validator
    try:
        validator = ClaimsValidator()
        print("\n  Claims Validator initialized successfully")
        print("   Using REAL Groq Llama 3.1 API for all decisions")
    except Exception as e:
        print(f"\n❌ ERROR initializing validator: {e}")
        return
    
    # Run demo
    results = validator.run_demo()
    
    print("\n" + "=" * 80)
    print("AGENT 4 EXECUTION COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()