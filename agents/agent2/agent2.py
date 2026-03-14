"""
AGENT 2: RAG Policy Change Detector (Groq Optimized)
Purpose: Compare new policy PDFs with existing Pinecone vectors to extract structured changes
"""

import os
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Changed import
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Import your existing helper functions
import sys
sys.path.append('D:/Data Science/SIH/Datathon-CIT')  # Add project root to path
try:
    from helper.helper import load_pdf_files, download_embeddings
except ImportError:
    # Alternative import if helper is in different location
    import sys
    sys.path.append('../../')  # Go up two levels to project root
    from helper.helper import load_pdf_files, download_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent2_change_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PolicyChangeDetector:
    """
    Detects policy changes by comparing new PDFs with existing Pinecone vectors
    Uses Groq for ultra-fast Llama 3 inference
    """
    
    def __init__(self, pinecone_index_name: str = "fhirdb"):
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = pinecone_index_name
        self.index = self.pc.Index(pinecone_index_name)
        
        # Initialize embeddings
        self.embeddings = download_embeddings()
        
        # Connect to existing vector store
        self.vectorstore = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=self.embeddings
        )
        
        # Initialize Groq with Llama 3.1
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=2000,
            timeout=30,
            max_retries=2
        )
        
        # Setup prompts
        self.setup_prompts()
        
    def setup_prompts(self):
        """Setup prompts for change detection"""
        
        # Prompt to compare old vs new policy
        self.change_detection_prompt = PromptTemplate(
            template="""You are a healthcare policy analyst. Compare the OLD policy text with the NEW policy text and identify specific changes.

OLD POLICY:
{old_policy}

NEW POLICY:
{new_policy}

Identify changes in the following categories:
1. Prior authorization requirements
2. Coverage criteria (dose limits, frequency limits)
3. Medical necessity criteria
4. Coding/billing requirements (CPT/HCPCS codes)
5. Documentation requirements

For each change found, extract:
- Policy code (if available, e.g., J3420)
- Change description (specific and actionable)
- Impact level (high/medium/low)
- Affected codes (list of CPT/HCPCS codes)

Format your response as a JSON array of changes:
[
  {{
    "code": "J3420",
    "change": "prior_auth_required >1000mcg",
    "impact": "high",
    "affected_codes": ["J3420", "G0008"],
    "details": "B12 injections >1000mcg now require prior authorization"
  }}
]

If no changes found, return empty array [].
""",
            input_variables=["old_policy", "new_policy"]
        )
        
        # Prompt to extract policy code from text
        self.code_extraction_prompt = PromptTemplate(
            template="""Extract the primary policy code (e.g., J3420, 82652, G0008, etc.) from this policy text.
Look for HCPCS/CPT codes that identify this specific policy.

Policy text: {policy_text}

Return just the code or "UNKNOWN" if not found:""",
            input_variables=["policy_text"]
        )
        
        # Prompt for quick policy summarization
        self.summary_prompt = PromptTemplate(
            template="""Summarize this healthcare policy in 2-3 sentences, focusing on coverage criteria and requirements:

Policy: {policy_text}

Summary:""",
            input_variables=["policy_text"]
        )
    
    def load_new_policies(self, agent1_output_file: str = "../agent1/downloads/policies/policies_for_vectorization.csv") -> List[Dict]:
        """Load new policies detected by Agent 1"""
        try:
            # Check if file exists
            if not Path(agent1_output_file).exists():
                logger.warning(f"Agent 1 output file not found: {agent1_output_file}")
                # Try alternative path
                alt_path = "./downloads/policies/policies_for_vectorization.csv"
                if Path(alt_path).exists():
                    agent1_output_file = alt_path
                    logger.info(f"Found policies at: {alt_path}")
                else:
                    return []
            
            df = pd.read_csv(agent1_output_file)
            policies = df.to_dict('records')
            logger.info(f"Loaded {len(policies)} new/updated policies from Agent 1")
            return policies
        except Exception as e:
            logger.error(f"Failed to load Agent 1 output: {e}")
            return []
    
    def extract_policy_code(self, policy_text: str) -> str:
        """Extract policy code using Groq LLM"""
        try:
            # Use Groq for fast extraction
            response = self.llm.invoke(
                self.code_extraction_prompt.format(policy_text=policy_text[:1500])
            )
            code = response.content.strip()
            
            if code and code != "UNKNOWN":
                logger.info(f"Extracted policy code: {code}")
                return code
                
        except Exception as e:
            logger.warning(f"LLM code extraction failed: {e}")
        
        # Fallback: regex for common code patterns
        patterns = [
            r'[A-Z][0-9]{4}',  # HCPCS: J3420
            r'[0-9]{5}',        # CPT: 82652
            r'[0-9]{4}[A-Z]',   # Some modifiers
            r'G[0-9]{4}',       # G-codes
            r'C[0-9]{4}'        # C-codes
        ]
        
        for pattern in patterns:
            match = re.search(pattern, policy_text)
            if match:
                code = match.group(0)
                logger.info(f"Extracted policy code via regex: {code}")
                return code
        
        return "UNKNOWN"
    
    def retrieve_old_policy(self, policy_code: str, policy_text: str) -> Optional[str]:
        """
        Retrieve old version of policy from Pinecone using similarity search
        """
        try:
            # First, check if Pinecone has any vectors
            stats = self.index.describe_index_stats()
            if stats.total_vector_count == 0:
                logger.info("Pinecone is empty - no old policies to compare")
                return None
            
            # Create query from policy summary for better retrieval
            summary = self.llm.invoke(
                self.summary_prompt.format(policy_text=policy_text[:2000])
            ).content
            
            # Search with metadata filter if code exists
            search_kwargs = {"k": 5}
            if policy_code != "UNKNOWN":
                # Try with code filter first
                try:
                    search_kwargs["filter"] = {"code": policy_code}
                    results = self.vectorstore.similarity_search(
                        summary,
                        **search_kwargs
                    )
                    
                    if results:
                        old_policy = "\n\n---\n\n".join([doc.page_content for doc in results])
                        logger.info(f"Retrieved old policy using code filter ({len(results)} chunks)")
                        return old_policy
                except Exception as e:
                    logger.warning(f"Filtered search failed: {e}")
            
            # Fallback to unfiltered similarity search
            search_kwargs.pop("filter", None)
            results = self.vectorstore.similarity_search(
                summary,
                **search_kwargs
            )
            
            if results:
                old_policy = "\n\n---\n\n".join([doc.page_content for doc in results])
                logger.info(f"Retrieved old policy using similarity search ({len(results)} chunks)")
                return old_policy
            
        except Exception as e:
            logger.error(f"Failed to retrieve old policy: {e}")
        
        return None
    
    def detect_changes(self, new_policy_text: str, old_policy_text: str) -> List[Dict]:
        """
        Use Groq LLM to detect structured changes between old and new policy
        """
        try:
            # Truncate if too long
            max_chars = 5000
            old_trunc = old_policy_text[:max_chars] if len(old_policy_text) > max_chars else old_policy_text
            new_trunc = new_policy_text[:max_chars] if len(new_policy_text) > max_chars else new_policy_text
            
            # Run change detection
            response = self.llm.invoke(
                self.change_detection_prompt.format(
                    old_policy=old_trunc,
                    new_policy=new_trunc
                )
            )
            
            # Parse JSON response
            response_text = response.content
            
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                changes = json.loads(json_match.group(0))
                logger.info(f"Detected {len(changes)} changes")
                return changes
            else:
                # Try to parse entire response as JSON
                try:
                    changes = json.loads(response_text)
                    if isinstance(changes, list):
                        return changes
                except:
                    pass
                
                logger.warning("No JSON array found in LLM response")
                logger.debug(f"Raw response: {response_text[:500]}")
                return []
                
        except Exception as e:
            logger.error(f"Change detection failed: {e}")
            return []
    
    def vectorize_new_policy(self, policy: Dict, policy_text: str):
        """
        Add new policy to Pinecone vector store
        """
        try:
            # Create document with enhanced metadata
            doc = Document(
                page_content=policy_text,
                metadata={
                    "code": policy.get('code', 'UNKNOWN'),
                    "title": policy.get('title', ''),
                    "effective_date": policy.get('effective_date', ''),
                    "source": "uhc_medicare_advantage",
                    "payer": "unitedhealthcare",
                    "version": policy.get('effective_date', '').replace('.', '') if policy.get('effective_date') else '',
                    "detected_at": policy.get('detected_at', datetime.now().isoformat()),
                    "document_type": "medical_policy"
                }
            )
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = text_splitter.split_documents([doc])
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk"] = i
                chunk.metadata["total_chunks"] = len(chunks)
            
            # Add to vector store
            self.vectorstore.add_documents(chunks)
            logger.info(f"Vectorized new policy: {policy.get('code')} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"Failed to vectorize policy: {e}")
    
    def process_policy(self, policy: Dict) -> List[Dict]:
        """
        Process a single policy: detect changes and update vector store
        """
        changes = []
        
        try:
            # Load PDF file
            file_path = policy.get('file_path')
            if not file_path or not Path(file_path).exists():
                logger.warning(f"PDF not found: {file_path}")
                return changes
            
            # Load and extract text from PDF
            pdf_dir = str(Path(file_path).parent)
            docs = load_pdf_files([pdf_dir])
            
            if not docs:
                logger.warning(f"Failed to extract text from {file_path}")
                return changes
            
            # Find the specific document we want
            target_doc = None
            for doc in docs:
                if doc.metadata.get('source', '').endswith(Path(file_path).name):
                    target_doc = doc
                    break
            
            if not target_doc:
                target_doc = docs[0]  # Fallback to first doc
            
            new_policy_text = target_doc.page_content
            
            # Try to extract policy code from text if not provided
            policy_code = policy.get('code')
            if policy_code in [None, 'UNKNOWN', '']:
                policy_code = self.extract_policy_code(new_policy_text)
                policy['code'] = policy_code
            
            # Retrieve old policy from Pinecone
            old_policy_text = self.retrieve_old_policy(policy_code, new_policy_text)
            
            if old_policy_text:
                # Detect changes
                changes = self.detect_changes(new_policy_text, old_policy_text)
                
                # Add metadata to changes
                for change in changes:
                    change['effective_date'] = policy.get('effective_date')
                    change['detected_at'] = datetime.now().isoformat()
                    change['policy_title'] = policy.get('title')
                    change['payer'] = 'unitedhealthcare'
            else:
                # This is a brand new policy
                logger.info(f"No old policy found for {policy_code} - treating as new")
                changes.append({
                    "code": policy_code,
                    "change": "new_policy_added",
                    "impact": "medium",
                    "affected_codes": [policy_code] if policy_code != "UNKNOWN" else [],
                    "details": f"New policy added: {policy.get('title', 'Unknown')}",
                    "effective_date": policy.get('effective_date'),
                    "detected_at": datetime.now().isoformat(),
                    "policy_title": policy.get('title'),
                    "payer": "unitedhealthcare"
                })
            
            # Always vectorize the new policy
            self.vectorize_new_policy(policy, new_policy_text)
            
        except Exception as e:
            logger.error(f"Error processing policy {policy.get('code')}: {e}")
        
        return changes
    
    def run(self, agent1_input_file: str = "../agent1/downloads/policies/policies_for_vectorization.csv") -> List[Dict]:
        """
        Main execution: process all new policies and detect changes
        """
        logger.info("Starting Agent 2: RAG Policy Change Detector (Groq Optimized)")
        
        # Load policies from Agent 1
        new_policies = self.load_new_policies(agent1_input_file)
        
        if not new_policies:
            logger.info("No new policies to process")
            return []
        
        # Process each policy
        all_changes = []
        for policy in new_policies:
            logger.info(f"Processing policy: {policy.get('code')} - {policy.get('title')}")
            changes = self.process_policy(policy)
            all_changes.extend(changes)
        
        # Output results for Agent 3
        self.output_changes(all_changes)
        
        return all_changes
    
    def output_changes(self, changes: List[Dict]):
        """
        Format changes for Agent 3 (FHIR Batch Updater)
        """
        print("\n" + "="*60)
        print("🚀 AGENT 2 OUTPUT: Policy Changes Detected (Groq + Llama 3.1)")
        print("="*60)
        
        if not changes:
            print("\n  No policy changes detected")
            return
        
        # Save to file for Agent 3
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"./policy_changes_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(changes, f, indent=2)
        
        # Also save a summary
        summary_file = f"./policy_changes_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Policy Changes Detected: {len(changes)}\n")
            f.write(f"Detection Time: {datetime.now().isoformat()}\n")
            f.write("-" * 40 + "\n")
            for change in changes:
                f.write(f"\n{change.get('code')}: {change.get('change')}\n")
                f.write(f"  Impact: {change.get('impact')}\n")
                f.write(f"  Effective: {change.get('effective_date')}\n")
        
        # Print summary
        print(f"\n📊 Detected {len(changes)} policy changes:")
        
        # Group by impact
        high_impact = [c for c in changes if c.get('impact') == 'high']
        medium_impact = [c for c in changes if c.get('impact') == 'medium']
        low_impact = [c for c in changes if c.get('impact') == 'low']
        
        if high_impact:
            print(f"\n🔴 HIGH IMPACT ({len(high_impact)}):")
            for change in high_impact:
                print(f"   • {change.get('code')}: {change.get('change')}")
        
        if medium_impact:
            print(f"\n🟡 MEDIUM IMPACT ({len(medium_impact)}):")
            for change in medium_impact:
                print(f"   • {change.get('code')}: {change.get('change')}")
        
        if low_impact:
            print(f"\n🟢 LOW IMPACT ({len(low_impact)}):")
            for change in low_impact:
                print(f"   • {change.get('code')}: {change.get('change')}")
        
        print(f"\n📁 Full details: {output_file}")
        print(f"📁 Summary: {summary_file}")
        print(f"\n📤 Sending to Agent 3 (FHIR Batch Updater)")

# Entry point
def main():
    """
    Main execution for Agent 2
    """
    print("\n🔍 Checking environment...")
    
    # Check for required API keys
    groq_key = os.getenv("GROQ_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    if not groq_key:
        logger.error("GROQ_API_KEY not found")
        print("\n❌ GROQ_API_KEY not set. Please add to .env file:")
        print("GROQ_API_KEY=your_api_key_here")
        return
    
    if not pinecone_key:
        logger.error("PINECONE_API_KEY not found")
        print("\n❌ PINECONE_API_KEY not set. Please add to .env file:")
        print("PINECONE_API_KEY=your_api_key_here")
        return
    
    print("  API keys found")
    
    try:
        detector = PolicyChangeDetector()
        changes = detector.run()
        
        # Structured output for downstream agents
        result = {
            "agent": "policy_change_detector_groq",
            "timestamp": datetime.now().isoformat(),
            "changes_detected": len(changes),
            "high_impact_count": len([c for c in changes if c.get('impact') == 'high']),
            "medium_impact_count": len([c for c in changes if c.get('impact') == 'medium']),
            "low_impact_count": len([c for c in changes if c.get('impact') == 'low']),
            "changes": changes,
            "next_agent": "fhir_batch_updater"
        }
        
        # Save final output
        with open("agent2_output.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\n  Agent 2 execution complete. Output saved to agent2_output.json")
        return result
        
    except Exception as e:
        logger.error(f"Agent 2 execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    main()