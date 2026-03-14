"""
AGENT 5A: Approval Router
Purpose: Process approved claims from Agent 4 and prepare for payer submission
Input: agent4_results_*.json file with approved claims
Output: EDI 837 files + submission tracking
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent5a_approval_router.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Agent5A_ApprovalRouter:
    """
    Processes approved claims from Agent 4:
    1. Reads Agent 4 output JSON
    2. Filters for APPROVED decisions
    3. Converts to EDI 837 format
    4. Prepares for payer submission
    5. Updates database with submission status
    """
    
    def __init__(self, db_path: str = "../agent3/fhir_procedure_db.sqlite"):
        self.db_path = db_path
        self.submissions = []
        self._verify_database()
    
    def _verify_database(self):
        """Ensure database has required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create submissions table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS claim_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT,
                patient_id TEXT,
                procedure_code TEXT,
                edi_filename TEXT,
                submission_status TEXT,
                payer_acknowledgement TEXT,
                submitted_at TIMESTAMP,
                UNIQUE(claim_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database verified: {self.db_path}")
    
    def find_latest_agent4_output(self) -> Optional[str]:
        """Find the most recent Agent 4 output file"""
        import glob
        
        # Search in current directory and parent directories
        search_paths = [
            "./agent4_results_*.json",
            "../agent4/agent4_results_*.json",
            "../../agent4/agent4_results_*.json"
        ]
        
        all_files = []
        for pattern in search_paths:
            all_files.extend(glob.glob(pattern))
        
        if not all_files:
            logger.error("No Agent 4 output files found")
            return None
        
        # Get the most recent file by modification time
        latest_file = max(all_files, key=os.path.getmtime)
        logger.info(f"Found latest Agent 4 output: {latest_file}")
        return latest_file
    
    def load_approved_decisions(self, agent4_file: str = None) -> List[Dict]:
        """
        Load Agent 4 output and filter for APPROVED decisions
        """
        if not agent4_file:
            agent4_file = self.find_latest_agent4_output()
            if not agent4_file:
                return []
        
        try:
            with open(agent4_file, 'r') as f:
                data = json.load(f)
            
            # Extract results from Agent 4 output
            if 'results' in data:
                all_decisions = data['results']
            elif isinstance(data, list):
                all_decisions = data
            else:
                all_decisions = [data]
            
            # Filter for APPROVED decisions
            approved = [d for d in all_decisions if d.get('decision') == 'APPROVED']
            
            logger.info(f"Loaded {len(approved)} approved decisions from {len(all_decisions)} total")
            return approved
            
        except Exception as e:
            logger.error(f"Failed to load Agent 4 output: {e}")
            return []
    
    def get_claim_details(self, claim_id: str, patient_id: str, procedure_code: str) -> Dict:
        """
        Fetch additional claim details from database
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Try to get from claims table if it exists
        try:
            cursor = conn.execute('''
                SELECT c.*, p.first_name, p.last_name, p.date_of_birth, p.gender
                FROM claims c
                JOIN patients p ON c.patient_id = p.id
                WHERE c.id = ?
            ''', (claim_id,))
            
            claim = cursor.fetchone()
            if claim:
                return dict(claim)
        except:
            pass
        
        # Fallback to procedures table
        try:
            cursor = conn.execute('''
                SELECT p.*, pt.first_name, pt.last_name, pt.date_of_birth, pt.gender
                FROM procedures p
                JOIN patients pt ON p.subject_patient_id = pt.id
                WHERE p.id = ? OR (p.subject_patient_id = ? AND p.code = ?)
                ORDER BY p.performedDateTime DESC
                LIMIT 1
            ''', (claim_id, patient_id, procedure_code))
            
            procedure = cursor.fetchone()
            if procedure:
                result = dict(procedure)
                # Map to claim structure
                return {
                    "id": result.get('id'),
                    "patient_id": result.get('subject_patient_id'),
                    "first_name": result.get('first_name'),
                    "last_name": result.get('last_name'),
                    "date_of_birth": result.get('date_of_birth'),
                    "gender": result.get('gender'),
                    "procedure_code": result.get('code'),
                    "date_of_service": result.get('performedDateTime', '').split('T')[0] if result.get('performedDateTime') else datetime.now().strftime('%Y-%m-%d'),
                    "charge_amount": 85.00,  # Default amount
                    "units": 1
                }
        except:
            pass
        
        conn.close()
        
        # Return minimal info from decision
        return {
            "id": claim_id,
            "patient_id": patient_id,
            "procedure_code": procedure_code,
            "date_of_service": datetime.now().strftime('%Y-%m-%d'),
            "charge_amount": 85.00,
            "units": 1,
            "first_name": "Unknown",
            "last_name": "Patient"
        }
    
    def generate_edi_837(self, claim: Dict, decision: Dict) -> str:
        """
        Generate EDI 837 file from claim and decision data
        Following X12 837P format for professional claims
        """
        edi_lines = []
        
        # Generate control numbers
        interchange_control = hash(f"{claim['id']}_{datetime.now().isoformat()}") % 1000000000
        group_control = hash(claim['id']) % 1000000
        
        # Get provider ID (default if not available)
        provider_id = claim.get('provider', 'PROVIDER123')
        payer_id = claim.get('payer_id', 'UHC87726')
        
        # Format dates
        submission_date = datetime.now().strftime('%y%m%d')
        submission_time = datetime.now().strftime('%H%M')
        service_date = claim.get('date_of_service', datetime.now().strftime('%Y-%m-%d')).replace('-', '')
        
        # Get patient name
        last_name = claim.get('last_name', 'UNKNOWN').upper()[:35]
        first_name = claim.get('first_name', 'PATIENT').upper()[:25]
        patient_dob = claim.get('date_of_birth', '19000101').replace('-', '')
        patient_gender = claim.get('gender', 'U')[:1]
        
        # Claim amount
        charge_amount = claim.get('charge_amount', 85.00)
        units = claim.get('units', 1)
        
        # ISA Segment - Interchange Control Header
        edi_lines.append(f"ISA*00*          *00*          *ZZ*{provider_id:<15}*ZZ*{payer_id:<15}*{submission_date}*{submission_time}*U*00401*{interchange_control:09d}*0*P*:~")
        
        # GS Segment - Functional Group Header
        edi_lines.append(f"GS*HC*{provider_id}*{payer_id}*{datetime.now().strftime('%Y%m%d')}*{submission_time}*{group_control:06d}*X*004010X098A1~")
        
        # ST Segment - Transaction Set Header
        edi_lines.append("ST*837*0001~")
        
        # BHT Segment - Beginning of Hierarchical Transaction
        edi_lines.append(f"BHT*0019*00*{claim['id']}*{datetime.now().strftime('%Y%m%d')}*{submission_time}*CH~")
        
        # Loop 1000A - Submitter Name
        edi_lines.append("NM1*41*2*SUBMITTER NAME*****46*123456789~")
        edi_lines.append("PER*IC*SUBMITTER CONTACT*TE*8005551234~")
        
        # Loop 1000B - Receiver Name
        edi_lines.append(f"NM1*40*2*{payer_id}*****46*{payer_id}~")
        
        # Loop 2000A - Billing Provider Hierarchical Level
        edi_lines.append("HL*1**20*1~")
        edi_lines.append(f"NM1*85*1*{provider_id}*****XX*1234567890~")
        edi_lines.append("N3*123 MAIN STREET~")
        edi_lines.append("N4*ANYTOWN*CA*90210~")
        edi_lines.append("REF*EI*123456789~")
        
        # Loop 2010AA - Billing Provider
        edi_lines.append(f"NM1*85*1*{provider_id}*****XX*1234567890~")
        edi_lines.append("N3*123 MAIN STREET~")
        edi_lines.append("N4*ANYTOWN*CA*90210~")
        
        # Loop 2000B - Subscriber Hierarchical Level
        edi_lines.append("HL*2*1*22*0~")
        edi_lines.append("SBR*P*18*******MC~")
        
        # Loop 2010BA - Subscriber Name
        edi_lines.append(f"NM1*IL*1*{last_name}*{first_name}****MI*{claim['patient_id']}~")
        edi_lines.append(f"DMG*D8*{patient_dob}*{patient_gender}~")
        
        # Loop 2010BB - Payer Name
        edi_lines.append(f"NM1*PR*2*{payer_id}*****PI*{payer_id}~")
        
        # Loop 2300 - Claim
        edi_lines.append(f"CLM*{claim['id']}*{charge_amount:.2f}***11:B:1*Y*A*Y*Y~")
        edi_lines.append(f"DTP*472*D8*{service_date}~")
        
        # Loop 2400 - Service Line
        edi_lines.append("LX*1~")
        edi_lines.append(f"SV1*HC:{decision.get('procedure_code', claim.get('procedure_code', 'UNKNOWN'))}*{units}*{charge_amount/units:.2f}***1~")
        edi_lines.append(f"DTP*472*D8*{service_date}~")
        
        # Add policy citations as note if available
        if decision.get('policy_citations'):
            citations = "; ".join(decision['policy_citations'][:3])
            edi_lines.append(f"PWK*OZ*FX***{citations[:80]}~")
        
        # Loop 2300 - Claim Note
        reasoning = decision.get('reasoning', '')
        if reasoning:
            edi_lines.append(f"NTE*ADD*{reasoning[:80]}~")
        
        # SE Segment - Transaction Set Trailer
        edi_lines.append("SE*25*0001~")
        
        # GE Segment - Functional Group Trailer
        edi_lines.append("GE*1*1~")
        
        # IEA Segment - Interchange Control Trailer
        edi_lines.append(f"IEA*1*{interchange_control:09d}~")
        
        return '\n'.join(edi_lines)
    
    def save_edi_file(self, claim_id: str, edi_content: str) -> str:
        """
        Save EDI content to file
        """
        # Create edi_output directory if it doesn't exist
        os.makedirs("edi_output", exist_ok=True)
        
        filename = f"edi_output/837_{claim_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.edi"
        
        with open(filename, 'w') as f:
            f.write(edi_content)
        
        logger.info(f"EDI file saved: {filename}")
        return filename
    
    def submit_to_payer(self, edi_filename: str, claim: Dict) -> Dict:
        """
        Simulate submission to payer
        In production, this would SFTP/API to payer
        """
        # Simulate submission
        submission_id = f"SUB{datetime.now().strftime('%Y%m%d%H%M%S')}_{claim['id']}"
        
        # In production, actual submission would happen here
        logger.info(f"Submitting {edi_filename} to payer")
        
        return {
            "status": "accepted",
            "acknowledgement": "997_ack_received",
            "submission_id": submission_id,
            "submitted_at": datetime.now().isoformat()
        }
    
    def update_database(self, claim_id: str, submission_result: Dict, edi_filename: str):
        """
        Update database with submission status
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            conn.execute('''
                INSERT OR REPLACE INTO claim_submissions 
                (claim_id, patient_id, procedure_code, edi_filename, submission_status, payer_acknowledgement, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                claim_id,
                submission_result.get('patient_id', 'unknown'),
                submission_result.get('procedure_code', 'unknown'),
                edi_filename,
                submission_result.get('status', 'unknown'),
                submission_result.get('acknowledgement', ''),
                submission_result.get('submitted_at', datetime.now().isoformat())
            ))
            
            # Also try to update claims table if it exists
            try:
                conn.execute('''
                    UPDATE claims 
                    SET status = 'submitted',
                        submission_id = ?,
                        submitted_at = ?
                    WHERE id = ?
                ''', (submission_result.get('submission_id'), datetime.now().isoformat(), claim_id))
            except:
                pass
            
            conn.commit()
            logger.info(f"Database updated for claim {claim_id}")
            
        except Exception as e:
            logger.error(f"Failed to update database: {e}")
        finally:
            conn.close()
    
    def process_approved_claim(self, decision: Dict) -> Dict:
        """
        Process a single approved claim
        """
        logger.info(f"Processing approved claim: {decision.get('claim_id')}")
        
        # Get claim details
        claim_id = decision.get('claim_id', 'unknown')
        patient_id = decision.get('patient_id', 'unknown')
        procedure_code = decision.get('procedure_code', 'unknown')
        
        claim_details = self.get_claim_details(claim_id, patient_id, procedure_code)
        
        # Add decision info to claim details
        claim_details['reasoning'] = decision.get('reasoning', '')
        claim_details['policy_citations'] = decision.get('policy_citations', [])
        
        # Generate EDI 837
        edi_content = self.generate_edi_837(claim_details, decision)
        
        # Save EDI file
        edi_filename = self.save_edi_file(claim_id, edi_content)
        
        # Submit to payer (simulated)
        submission_result = self.submit_to_payer(edi_filename, claim_details)
        submission_result['patient_id'] = patient_id
        submission_result['procedure_code'] = procedure_code
        
        # Update database
        self.update_database(claim_id, submission_result, edi_filename)
        
        # Prepare result
        result = {
            "claim_id": claim_id,
            "patient_id": patient_id,
            "procedure_code": procedure_code,
            "decision_summary": {
                "reasoning": decision.get('reasoning', '')[:100] + "..." if len(decision.get('reasoning', '')) > 100 else decision.get('reasoning', ''),
                "policy_citations_count": len(decision.get('policy_citations', []))
            },
            "edi_filename": edi_filename,
            "submission_status": submission_result['status'],
            "acknowledgement": submission_result.get('acknowledgement'),
            "submission_id": submission_result.get('submission_id'),
            "submitted_at": submission_result.get('submitted_at'),
            "edi_content_preview": edi_content[:200] + "..."  # Preview for logging
        }
        
        self.submissions.append(result)
        return result
    
    def run(self, agent4_file: str = None) -> Dict:
        """
        Main execution: Process all approved claims from Agent 4
        """
        print("\n" + "=" * 80)
        print("AGENT 5A: APPROVAL ROUTER")
        print("=" * 80)
        
        # Load approved decisions
        approved_decisions = self.load_approved_decisions(agent4_file)
        
        if not approved_decisions:
            print("\nNo approved claims found in Agent 4 output.")
            return {
                "status": "no_claims",
                "message": "No approved claims to process",
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"\nFound {len(approved_decisions)} approved claims to process")
        
        # Process each approved claim
        results = []
        for i, decision in enumerate(approved_decisions, 1):
            print(f"\n{'-' * 50}")
            print(f"Processing claim {i}/{len(approved_decisions)}: {decision.get('claim_id')}")
            print(f"  Procedure: {decision.get('procedure_code')}")
            print(f"  Reasoning: {decision.get('reasoning', '')[:100]}...")
            
            result = self.process_approved_claim(decision)
            results.append(result)
            
            print(f"  -> EDI: {result['edi_filename']}")
            print(f"  -> Status: {result['submission_status']}")
            if result.get('submission_id'):
                print(f"  -> Submission ID: {result['submission_id']}")
        
        # Generate summary
        summary = self.generate_summary(results)
        self.print_summary(summary)
        
        # Save results
        output_file = f"agent5a_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "claims_processed": len(results),
                "submission_status": {
                    "successful": len([r for r in results if r['submission_status'] == 'accepted']),
                    "failed": len([r for r in results if r['submission_status'] != 'accepted'])
                },
                "results": results,
                "next_agent": "none",  # Terminal agent in approval path
                "message": "Claims submitted to payer, awaiting response"
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return {
            "status": "completed",
            "claims_processed": len(results),
            "results": results,
            "output_file": output_file
        }
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """
        Generate summary statistics
        """
        return {
            "total_claims": len(results),
            "successful_submissions": len([r for r in results if r['submission_status'] == 'accepted']),
            "failed_submissions": len([r for r in results if r['submission_status'] != 'accepted']),
            "edi_files": [r['edi_filename'] for r in results],
            "procedure_codes": list(set([r['procedure_code'] for r in results])),
            "patients": list(set([r['patient_id'] for r in results]))
        }
    
    def print_summary(self, summary: Dict):
        """
        Print execution summary
        """
        print("\n" + "=" * 80)
        print("AGENT 5A EXECUTION SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal Claims Processed: {summary['total_claims']}")
        print(f"Successful Submissions: {summary['successful_submissions']}")
        print(f"Failed Submissions: {summary['failed_submissions']}")
        
        if summary['procedure_codes']:
            print(f"\nProcedure Codes Processed:")
            for code in summary['procedure_codes']:
                count = len([r for r in self.submissions if r['procedure_code'] == code])
                print(f"  - {code}: {count} claims")
        
        if summary['patients']:
            print(f"\nPatients:")
            for patient in summary['patients'][:5]:  # Show first 5
                print(f"  - {patient}")
            if len(summary['patients']) > 5:
                print(f"    ... and {len(summary['patients']) - 5} more")
        
        if summary['edi_files']:
            print(f"\nEDI Files Generated:")
            for edi in summary['edi_files'][:3]:  # Show first 3
                print(f"  - {edi}")
            if len(summary['edi_files']) > 3:
                print(f"    ... and {len(summary['edi_files']) - 3} more")
        
        print(f"\nNext Steps: Awaiting 997/999 acknowledgement from payer")


# Main execution
def main():
    """Entry point for Agent 5A"""
    
    print("\n" + "=" * 80)
    print("INITIALIZING AGENT 5A: APPROVAL ROUTER")
    print("=" * 80)
    
    # Check for Agent 4 output
    agent5a = Agent5A_ApprovalRouter()
    
    # Find latest Agent 4 output
    latest_file = agent5a.find_latest_agent4_output()
    
    if not latest_file:
        print("\nERROR: No Agent 4 output files found.")
        print("Please run Agent 4 first to generate claim decisions.")
        return
    
    print(f"\nUsing Agent 4 output: {latest_file}")
    
    # Preview approved claims
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    if 'results' in data:
        all_decisions = data['results']
    else:
        all_decisions = data if isinstance(data, list) else [data]
    
    approved = [d for d in all_decisions if d.get('decision') == 'APPROVED']
    
    print(f"\nAgent 4 Results Summary:")
    print(f"  Total decisions: {len(all_decisions)}")
    print(f"  Approved claims: {len(approved)}")
    
    if approved:
        print(f"\nApproved Claims:")
        for d in approved:
            print(f"  - {d.get('claim_id')}: {d.get('procedure_code')}")
            print(f"    Reason: {d.get('reasoning', '')[:100]}...")
    
    # Run the approval router
    result = agent5a.run(latest_file)
    
    print("\n" + "=" * 80)
    print("AGENT 5A EXECUTION COMPLETE")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    main()