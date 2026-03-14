"""
AGENT 3: FHIR Batch Updater (Procedure Schema)
Purpose: Update patient procedure records based on policy changes
Input: Policy changes JSON from Agent 2
Output: Updated procedure records + audit log

Procedure Schema:
- id
- status (preparation/in-progress/completed)
- statusReason
- category
- code (SNOMED/CPT/HCPCS)
- subject (ref Patient)
- encounter (ref Encounter)
- basedOn (ref ServiceRequest)
- partOf (ref Procedure/Encounter)
- performedDateTime / performedPeriod
- performer
- location
- reasonCode / reasonReference
- bodySite
- outcome
- complication
- followUp
- note
- focalDevice
"""

import os
import json
import sqlite3
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent3_fhir_updater.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FHIRProcedureUpdater:
    """
    Updates procedure records in FHIR database based on policy changes
    Follows the exact Procedure schema provided
    """
    
    def __init__(self, db_path: str = "fhir_procedure_db.sqlite"):
        self.db_path = db_path
        self.batch_size = 100
        
        # Statistics tracking
        self.stats = {
            "total_affected": 0,
            "procedures_updated": 0,
            "procedures_flagged": 0,
            "failed_updates": 0,
            "procedure_ids": []
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with the exact Procedure schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create Procedure table matching the provided schema
        cursor.executescript('''
            -- Procedure table (FHIR Procedure resource)
            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                status TEXT CHECK(status IN ('preparation', 'in-progress', 'completed', 'suspended', 'aborted', 'entered-in-error')),
                statusReason TEXT,
                category TEXT,
                code TEXT,  -- SNOMED/CPT/HCPCS
                subject_patient_id TEXT,  -- Reference to Patient
                encounter_id TEXT,  -- Reference to Encounter
                basedOn_service_request_id TEXT,  -- Reference to ServiceRequest
                partOf_procedure_id TEXT,  -- Reference to parent Procedure
                partOf_encounter_id TEXT,  -- Reference to Encounter
                performedDateTime TEXT,
                performedPeriod_start TEXT,
                performedPeriod_end TEXT,
                location_id TEXT,  -- Reference to Location
                reasonCode TEXT,
                reasonReference TEXT,
                bodySite TEXT,
                outcome TEXT,
                complication TEXT,
                followUp TEXT,  -- Reference to next Appointment/ServiceRequest
                note TEXT,
                focalDevice TEXT,  -- Equipment used
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Performers table (since performer is an array in FHIR)
            CREATE TABLE IF NOT EXISTS procedure_performers (
                id TEXT PRIMARY KEY,
                procedure_id TEXT,
                performer_function TEXT,
                actor_practitioner_id TEXT,  -- Reference to Practitioner
                FOREIGN KEY (procedure_id) REFERENCES procedures(id)
            );
            
            -- Patients table
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                mrn TEXT UNIQUE,
                first_name TEXT,
                last_name TEXT,
                date_of_birth TEXT,
                gender TEXT
            );
            
            -- Practitioners table
            CREATE TABLE IF NOT EXISTS practitioners (
                id TEXT PRIMARY KEY,
                npi TEXT UNIQUE,
                first_name TEXT,
                last_name TEXT,
                specialty TEXT
            );
            
            -- Encounters table
            CREATE TABLE IF NOT EXISTS encounters (
                id TEXT PRIMARY KEY,
                patient_id TEXT,
                status TEXT,
                class TEXT,
                period_start TEXT,
                period_end TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            );
            
            -- Locations table
            CREATE TABLE IF NOT EXISTS locations (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                address TEXT
            );
            
            -- Policy flags table (for tracking policy requirements)
            CREATE TABLE IF NOT EXISTS procedure_policy_flags (
                flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                procedure_id TEXT,
                policy_code TEXT,
                requirement TEXT,
                effective_date TEXT,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(procedure_id, policy_code)
            );
            
            -- Audit log
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
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized: {self.db_path}")
    
    def load_policy_changes(self, changes_file: str = None) -> List[Dict]:
        """Load policy changes from Agent 2 output"""
        if not changes_file:
            # Try different possible locations
            possible_paths = [
                "../agent2/policy_changes_*.json",
                "./policy_changes_*.json",
                "../../agents/agent2/policy_changes_*.json",
                "../agent2/agent2_output.json"
            ]
            
            import glob
            for pattern in possible_paths:
                files = glob.glob(pattern)
                if files:
                    changes_file = max(files, key=os.path.getctime)
                    break
        
        if not changes_file or not os.path.exists(changes_file):
            # Use sample changes for demo
            logger.warning("No policy changes file found. Using sample changes.")
            return self._get_sample_changes()
        
        logger.info(f"Loading policy changes from: {changes_file}")
        with open(changes_file, 'r') as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, dict):
            if 'changes' in data:
                changes = data['changes']
            elif 'change' in data:
                changes = [data]
            else:
                changes = [data]
        elif isinstance(data, list):
            changes = data
        else:
            changes = []
        
        logger.info(f"Loaded {len(changes)} policy changes")
        return changes
    
    def _get_sample_changes(self) -> List[Dict]:
        """Return sample policy changes for demonstration"""
        return [
            {
                "code": "J3420",
                "change": "prior_auth_required >1000mcg",
                "impact": "high",
                "affected_codes": ["J3420", "G0008"],
                "details": "B12 injections >1000mcg now require prior authorization",
                "effective_date": "2026-03-01"
            },
            {
                "code": "97110",
                "change": "visit_limit_12_per_year",
                "impact": "medium",
                "affected_codes": ["97110", "97112", "97116"],
                "details": "Physical therapy visits limited to 12 per year",
                "effective_date": "2026-03-01"
            },
            {
                "code": "82652",
                "change": "coverage_limited_to_deficiency",
                "impact": "medium",
                "affected_codes": ["82652"],
                "details": "Vitamin D testing only covered for documented deficiency",
                "effective_date": "2026-03-01"
            }
        ]
    
    def find_affected_procedures(self, policy_change: Dict) -> List[Dict]:
        """
        Find all procedures with the affected procedure codes
        Returns procedures matching the FHIR schema
        """
        affected_codes = policy_change.get('affected_codes', [policy_change.get('code')])
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        
        # Build query with parameters
        placeholders = ','.join(['?' for _ in affected_codes])
        query = f'''
            SELECT 
                p.*,
                pt.first_name as patient_first_name,
                pt.last_name as patient_last_name,
                pt.date_of_birth as patient_dob,
                pt.gender as patient_gender,
                e.status as encounter_status,
                e.period_start as encounter_start,
                l.name as location_name
            FROM procedures p
            LEFT JOIN patients pt ON p.subject_patient_id = pt.id
            LEFT JOIN encounters e ON p.encounter_id = e.id
            LEFT JOIN locations l ON p.location_id = l.id
            WHERE p.code IN ({placeholders})
            AND p.status IN ('in-progress', 'completed')
            ORDER BY p.performedDateTime DESC
        '''
        
        cursor = conn.execute(query, affected_codes)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        procedures = [dict(row) for row in rows]
        
        # Also get performers for each procedure
        for proc in procedures:
            performer_query = '''
                SELECT * FROM procedure_performers 
                WHERE procedure_id = ?
            '''
            cursor = conn.execute(performer_query, (proc['id'],))
            performers = [dict(row) for row in cursor.fetchall()]
            proc['performers'] = performers
        
        conn.close()
        
        logger.info(f"Found {len(procedures)} procedures with codes {affected_codes}")
        return procedures
    
    def update_procedure_policy_flag(self, procedure: Dict, policy_change: Dict) -> bool:
        """
        Update policy flag for a single procedure
        Adds a note and potentially updates status if prior auth required
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            procedure_id = procedure['id']
            policy_code = policy_change['code']
            
            # Check if flag already exists
            cursor.execute('''
                SELECT flag_id FROM procedure_policy_flags
                WHERE procedure_id = ? AND policy_code = ?
            ''', (procedure_id, policy_code))
            
            existing = cursor.fetchone()
            
            # Create policy flag details
            flag_details = json.dumps({
                "requirement": policy_change['change'],
                "impact": policy_change.get('impact', 'medium'),
                "effective_date": policy_change.get('effective_date'),
                "details": policy_change.get('details', ''),
                "affected_codes": policy_change.get('affected_codes', [])
            })
            
            if existing:
                # Update existing flag
                cursor.execute('''
                    UPDATE procedure_policy_flags
                    SET requirement = ?,
                        effective_date = ?,
                        applied_at = CURRENT_TIMESTAMP
                    WHERE procedure_id = ? AND policy_code = ?
                ''', (
                    flag_details,
                    policy_change.get('effective_date', datetime.now().strftime('%Y-%m-%d')),
                    procedure_id,
                    policy_code
                ))
                action = "updated"
            else:
                # Insert new flag
                cursor.execute('''
                    INSERT INTO procedure_policy_flags 
                    (procedure_id, policy_code, requirement, effective_date)
                    VALUES (?, ?, ?, ?)
                ''', (
                    procedure_id,
                    policy_code,
                    flag_details,
                    policy_change.get('effective_date', datetime.now().strftime('%Y-%m-%d'))
                ))
                action = "added"
            
            # If this is a high-impact change, add a note to the procedure
            if policy_change.get('impact') == 'high':
                # Get existing note
                cursor.execute('SELECT note FROM procedures WHERE id = ?', (procedure_id,))
                current_note = cursor.fetchone()
                
                new_note = f"[POLICY UPDATE {policy_change.get('effective_date')}] {policy_change['change']}. {policy_change.get('details', '')}"
                
                if current_note and current_note[0]:
                    new_note = current_note[0] + "\n" + new_note
                
                # Update procedure note
                cursor.execute('''
                    UPDATE procedures
                    SET note = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (new_note, procedure_id))
            
            # Log to audit
            cursor.execute('''
                INSERT INTO audit_log 
                (agent_name, action, procedure_id, policy_code, details)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                "fhir_batch_updater",
                f"policy_flag_{action}",
                procedure_id,
                policy_code,
                json.dumps({"change": policy_change['change'], "impact": policy_change.get('impact')})
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"{action.title()} policy flag for procedure {procedure_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update procedure {procedure.get('id')}: {e}")
            return False
    
    def batch_update_procedures(self, procedures: List[Dict], policy_change: Dict) -> Dict:
        """
        Update multiple procedures in batch
        """
        stats = {
            "total": len(procedures),
            "success": 0,
            "failed": 0,
            "procedure_ids": []
        }
        
        # Process in batches
        for i in range(0, len(procedures), self.batch_size):
            batch = procedures[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}: {len(batch)} procedures")
            
            for procedure in batch:
                success = self.update_procedure_policy_flag(procedure, policy_change)
                if success:
                    stats["success"] += 1
                    stats["procedure_ids"].append(procedure['id'])
                else:
                    stats["failed"] += 1
        
        return stats
    
    def generate_compliance_report(self, policy_change: Dict, update_stats: Dict):
        """
        Generate compliance report for the update
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "policy_change": {
                "code": policy_change['code'],
                "change": policy_change['change'],
                "impact": policy_change.get('impact'),
                "effective_date": policy_change.get('effective_date')
            },
            "update_stats": update_stats,
            "affected_procedures": update_stats['procedure_ids'][:10],  # First 10 for preview
            "total_affected": update_stats['total'],
            "compliance_status": "updated"
        }
        
        # Save report
        report_file = f"compliance_report_{policy_change['code']}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_file
    
    def create_sample_data(self):
        """Create sample procedure data for testing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sample patients
        patients = [
            ("pat001", "MRN001", "John", "Smith", "1965-03-15", "M"),
            ("pat002", "MRN002", "Jane", "Doe", "1972-07-22", "F"),
            ("pat003", "MRN003", "Robert", "Johnson", "1958-11-30", "M"),
            ("pat004", "MRN004", "Maria", "Garcia", "1980-05-10", "F"),
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO patients (id, mrn, first_name, last_name, date_of_birth, gender)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', patients)
        
        # Sample practitioners
        practitioners = [
            ("prat001", "1234567890", "Sarah", "Chen", "Physical Therapy"),
            ("prat002", "2345678901", "Michael", "Patel", "Internal Medicine"),
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO practitioners (id, npi, first_name, last_name, specialty)
            VALUES (?, ?, ?, ?, ?)
        ''', practitioners)
        
        # Sample locations
        locations = [
            ("loc001", "Main Clinic", "outpatient", "123 Main St"),
            ("loc002", "Rehab Center", "therapy", "456 Oak Ave"),
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO locations (id, name, type, address)
            VALUES (?, ?, ?, ?)
        ''', locations)
        
        # Sample encounters
        encounters = [
            ("enc001", "pat001", "completed", "outpatient", "2026-02-15T09:00:00Z", "2026-02-15T10:00:00Z"),
            ("enc002", "pat002", "completed", "outpatient", "2026-02-16T14:00:00Z", "2026-02-16T15:00:00Z"),
            ("enc003", "pat003", "in-progress", "outpatient", "2026-02-20T11:00:00Z", None),
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO encounters (id, patient_id, status, class, period_start, period_end)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', encounters)
        
        # Sample procedures (using the exact schema)
        procedures = [
            # J3420 B12 injections
            ("proc001", "completed", None, "injection", "J3420", "pat001", "enc001", None, None, None,
             "2026-02-15T09:30:00Z", None, None, "loc001", "E11.9", None, "left arm", "successful", None,
             None, "Routine B12 injection", None),
            
            ("proc002", "completed", None, "injection", "J3420", "pat002", "enc002", None, None, None,
             "2026-02-16T14:30:00Z", None, None, "loc001", "D51.0", None, "right arm", "successful", None,
             None, "B12 deficiency treatment", None),
            
            ("proc003", "in-progress", None, "injection", "J3420", "pat003", "enc003", None, None, None,
             "2026-02-20T11:30:00Z", None, None, "loc001", "E53.8", None, "left arm", None, None,
             None, "Follow-up B12 injection", None),
            
            # Physical therapy procedures
            ("proc004", "completed", None, "therapy", "97110", "pat001", "enc001", None, None, None,
             "2026-02-15T10:00:00Z", None, None, "loc002", "M54.5", None, "lower back", "improved", None,
             None, "Therapeutic exercises", "exercise_equipment_1"),
            
            ("proc005", "completed", None, "therapy", "97112", "pat002", "enc002", None, None, None,
             "2026-02-16T15:00:00Z", None, None, "loc002", "M25.56", None, "right knee", "stable", None,
             None, "Neuromuscular reeducation", "therapy_band"),
            
            # Vitamin D test
            ("proc006", "completed", None, "laboratory", "82652", "pat004", None, None, None, None,
             "2026-02-17T08:00:00Z", None, None, "loc001", "Z13.6", None, None, "normal", None,
             None, "Annual vitamin D screening", None),
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO procedures 
            (id, status, statusReason, category, code, subject_patient_id, encounter_id,
             basedOn_service_request_id, partOf_procedure_id, partOf_encounter_id,
             performedDateTime, performedPeriod_start, performedPeriod_end, location_id,
             reasonCode, reasonReference, bodySite, outcome, complication, followUp,
             note, focalDevice)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', procedures)
        
        # Sample performers
        performers = [
            ("perf001", "proc001", "attending", "prat002"),
            ("perf002", "proc002", "attending", "prat002"),
            ("perf003", "proc003", "attending", "prat002"),
            ("perf004", "proc004", "therapist", "prat001"),
            ("perf005", "proc005", "therapist", "prat001"),
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO procedure_performers (id, procedure_id, performer_function, actor_practitioner_id)
            VALUES (?, ?, ?, ?)
        ''', performers)
        
        conn.commit()
        conn.close()
        
        logger.info("Sample procedure data created")
    
    def run(self, changes_file: str = None):
        """
        Main execution: Apply all policy changes to procedure records
        """
        logger.info(" Starting Agent 3: FHIR Procedure Batch Updater")
        
        # Create sample data if database is empty
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM procedures")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            logger.info("Creating sample procedure data...")
            self.create_sample_data()
        
        # Load policy changes
        changes = self.load_policy_changes(changes_file)
        if not changes:
            logger.info("No policy changes to process")
            return
        
        print("\n" + "="*70)
        print(" AGENT 3: FHIR PROCEDURE BATCH UPDATER")
        print("="*70)
        
        all_results = []
        
        # Process each policy change
        for change in changes:
            print(f"\n Processing policy: {change.get('code')} - {change.get('change')}")
            print(f"   Impact: {change.get('impact', 'medium')} | Effective: {change.get('effective_date', 'N/A')}")
            
            # Find affected procedures
            procedures = self.find_affected_procedures(change)
            
            if not procedures:
                print(f"     No procedures found with codes: {change.get('affected_codes', [change.get('code')])}")
                continue
            
            print(f"    Found {len(procedures)} affected procedures")
            
            # Show sample of affected procedures
            print(f"   Sample affected procedures:")
            for proc in procedures[:3]:  # Show first 3
                status_icon = "🟢" if proc['status'] == 'completed' else "🟡"
                print(f"     {status_icon} {proc['id']}: {proc['code']} - Patient: {proc.get('patient_first_name', 'Unknown')} {proc.get('patient_last_name', '')}")
                print(f"        Date: {proc.get('performedDateTime', 'N/A')}")
            
            # Batch update procedures
            stats = self.batch_update_procedures(procedures, change)
            
            # Generate compliance report
            report_file = self.generate_compliance_report(change, stats)
            
            # Update global stats
            self.stats["total_affected"] += len(procedures)
            self.stats["procedures_updated"] += stats["success"]
            self.stats["failed_updates"] += stats["failed"]
            self.stats["procedure_ids"].extend(stats["procedure_ids"])
            
            all_results.append({
                "policy": change['code'],
                "change": change['change'],
                "impact": change.get('impact'),
                "procedures_affected": len(procedures),
                "successful": stats["success"],
                "failed": stats["failed"],
                "report_file": report_file
            })
            
            print(f"\n    Updated: {stats['success']} procedures")
            if stats['failed'] > 0:
                print(f"    Failed: {stats['failed']} procedures")
            print(f"    Report: {report_file}")
        
        # Final summary
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results: List[Dict]):
        """
        Print final summary
        """
        print("\n" + "="*70)
        print(" AGENT 3 EXECUTION SUMMARY")
        print("="*70)
        
        print(f"\n Global Statistics:")
        print(f"   • Total procedures affected: {self.stats['total_affected']}")
        print(f"   • Successfully updated: {self.stats['procedures_updated']}")
        print(f"   • Failed updates: {self.stats['failed_updates']}")
        
        print(f"\n Policy Changes Applied:")
        for r in results:
            impact_icon = "🔴" if r['impact'] == 'high' else "🟡" if r['impact'] == 'medium' else "🟢"
            print(f"   {impact_icon} {r['policy']}: {r['change']}")
            print(f"      → {r['successful']} procedures updated")
        
        # Save results for Agent 4
        output = {
            "agent": "fhir_procedure_updater",
            "timestamp": datetime.now().isoformat(),
            "policies_processed": len(results),
            "stats": self.stats,
            "results": results,
            "next_agent": "claims_validator",
            "message": f"FHIR database updated: {self.stats['procedures_updated']} procedures modified"
        }
        
        output_file = f"agent3_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n Full report: {output_file}")
        print(f"\n Handing off to Agent 4: Claims Validator")
        print("="*70)
    
    def query_procedures(self, code: str = None, patient_id: str = None):
        """
        Utility method to query procedures
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        query = "SELECT * FROM procedures WHERE 1=1"
        params = []
        
        if code:
            query += " AND code = ?"
            params.append(code)
        if patient_id:
            query += " AND subject_patient_id = ?"
            params.append(patient_id)
        
        query += " ORDER BY performedDateTime DESC LIMIT 10"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df

# Main execution
def main():
    """
    Entry point for Agent 3
    """
    print("\n" + "="*70)
    print(" INITIALIZING AGENT 3: FHIR PROCEDURE UPDATER")
    print("="*70)
    
    # Initialize updater
    updater = FHIRProcedureUpdater("fhir_procedure_db.sqlite")
    
    try:
        # Find policy changes file
        changes_file = None
        import glob
        possible_files = glob.glob("../agent2/policy_changes_*.json") + \
                        glob.glob("./policy_changes_*.json") + \
                        glob.glob("../agent2/agent2_output.json")
        
        if possible_files:
            changes_file = max(possible_files, key=os.path.getctime)
            print(f"\n Using policy changes from: {changes_file}")
        else:
            print("\n No policy changes file found. Using sample changes.")
        
        # Run the batch update
        results = updater.run(changes_file)
        
        # Show sample of updated procedures
        print("\n" + "="*70)
        print("🔍 SAMPLE UPDATED PROCEDURES")
        print("="*70)
        
        # Query J3420 procedures to show updates
        j3420_procs = updater.query_procedures(code="J3420")
        if not j3420_procs.empty:
            print("\n📋 J3420 Procedures (B12 Injections):")
            for _, proc in j3420_procs.iterrows():
                # Check if this procedure has policy flags
                conn = sqlite3.connect(updater.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT requirement FROM procedure_policy_flags 
                    WHERE procedure_id = ? AND policy_code = 'J3420'
                ''', (proc['id'],))
                flag = cursor.fetchone()
                conn.close()
                
                status = " FLAGGED" if flag else " NO FLAG"
                print(f"   {status} {proc['id']}: {proc['code']} - {proc['performedDateTime']}")
                if flag:
                    req = json.loads(flag[0])
                    print(f"      → {req.get('requirement', 'N/A')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Agent 3 execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Install pandas if not available
    try:
        import pandas as pd
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'pandas'])
        import pandas as pd
    
    main()