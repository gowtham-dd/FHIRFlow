"""
AGENT 1: UHC Medicare Advantage Policy Crawler
Purpose: Scrapes UHC provider portal for new/updated medical policies
Output: Downloads PDFs and prepares metadata for Pinecone ingestion
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UHCPolicyCrawler:
    """
    Crawls UHC Medicare Advantage policy page, detects changes,
    and downloads new/updated policy PDFs.
    """
    
    def __init__(self, download_dir: str = "./downloads/policies", 
                 state_file: str = "./crawler_state.json"):
        self.base_url = "https://www.uhcprovider.com"
        self.target_url = "https://www.uhcprovider.com/en/policies-protocols/medicare-advantage-policies/medicare-advantage-medical-policies.html"
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = Path(state_file)
        self.last_run_state = self._load_state()
        
    def _load_state(self) -> Dict:
        """Load last crawler state to detect changes"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {"last_run": None, "known_policies": {}}
    
    def _save_state(self, policies: Dict):
        """Save current state for next run"""
        state = {
            "last_run": datetime.now().isoformat(),
            "known_policies": policies
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _generate_policy_id(self, title: str, date: str) -> str:
        """Generate unique policy ID for deduplication"""
        unique_str = f"{title}|{date}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]
    
    async def handle_consent_dialog(self, page: Page):
        """Handle the initial terms and conditions popup"""
        try:
            # Wait for iframe or direct consent button
            await page.wait_for_selector('iframe[src*="consent"], button:has-text("I Agree"), button:has-text("Accept")', 
                                       timeout=10000)
            
            # Check if consent is in iframe
            iframe = await page.query_selector('iframe[src*="consent"]')
            if iframe:
                frame = await iframe.content_frame()
                agree_btn = await frame.wait_for_selector('button:has-text("I Agree")', timeout=5000)
                await agree_btn.click()
                logger.info("Clicked consent in iframe")
            else:
                # Direct button
                agree_btn = await page.wait_for_selector('button:has-text("I Agree"), button:has-text("Accept")', 
                                                        timeout=5000)
                await agree_btn.click()
                logger.info("Clicked direct consent button")
            
            # Wait for navigation to complete
            await page.wait_for_load_state("networkidle")
            
        except Exception as e:
            logger.warning(f"No consent dialog found or already accepted: {e}")
    
    async def extract_policy_links(self, page: Page) -> List[Dict]:
        """
        Extract all policy links with their metadata
        """
        policies = []
        
        try:
            # Wait for policy list to load
            await page.wait_for_selector('div[class*="policy"], a[href*=".pdf"]', timeout=30000)
            
            # Find all policy entries - they appear as sections with headings
            policy_sections = await page.query_selector_all('div[class*="policy-item"], div[class*="accordion"], section')
            
            for section in policy_sections:
                # Extract title
                title_elem = await section.query_selector('h2, h3, h4, strong, a')
                title = await title_elem.text_content() if title_elem else "Unknown Policy"
                title = title.strip()
                
                # Extract last published date
                date_text = await section.text_content()
                date = self._extract_date(date_text)
                
                # Find PDF link
                pdf_link = await section.query_selector('a[href$=".pdf"]')
                if pdf_link:
                    href = await pdf_link.get_attribute('href')
                    if href:
                        pdf_url = urljoin(self.base_url, href)
                        
                        # Extract policy code from title or content
                        code = self._extract_policy_code(title, date_text)
                        
                        policy = {
                            'id': self._generate_policy_id(title, date),
                            'title': title,
                            'code': code,
                            'date': date,
                            'url': pdf_url,
                            'detected_at': datetime.now().isoformat()
                        }
                        policies.append(policy)
                        logger.info(f"Found policy: {code} - {title} ({date})")
            
        except Exception as e:
            logger.error(f"Error extracting policies: {e}")
            
        return policies
    
    def _extract_date(self, text: str) -> str:
        """Extract date in MM.DD.YYYY format from text"""
        import re
        date_pattern = r'(\d{2}\.\d{2}\.\d{4})'
        match = re.search(date_pattern, text)
        if match:
            return match.group(1)
        return "Unknown"
    
    def _extract_policy_code(self, title: str, text: str) -> str:
        """
        Extract policy code from title or content.
        UHC policies often have codes like "J3420" in the text
        """
        import re
        # Look for CPT/HCPCS codes in the text
        code_pattern = r'[A-Z][0-9]{4}|[0-9]{5}'  # Simple pattern for HCPCS/CPT
        codes = re.findall(code_pattern, text)
        if codes:
            # Return first code found as the policy identifier
            return codes[0]
        
        # Fallback: create code from title
        words = title.split()[:3]
        return ''.join(word[:3].upper() for word in words if word[0].isalpha())
    
    def detect_changes(self, current_policies: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Compare with last run to detect new and updated policies
        Returns: (new_policies, updated_policies)
        """
        known = self.last_run_state["known_policies"]
        current_dict = {p['id']: p for p in current_policies}
        
        new_policies = []
        updated_policies = []
        
        for policy_id, policy in current_dict.items():
            if policy_id not in known:
                new_policies.append(policy)
                logger.info(f"New policy detected: {policy['code']}")
            elif known[policy_id]['date'] != policy['date']:
                updated_policies.append(policy)
                logger.info(f"Updated policy detected: {policy['code']} (was {known[policy_id]['date']}, now {policy['date']})")
        
        return new_policies, updated_policies
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def download_policy(self, policy: Dict, session: aiohttp.ClientSession) -> Optional[Path]:
        """
        Download a policy PDF with retry logic
        """
        try:
            filename = f"{policy['code']}_{policy['date'].replace('.', '')}.pdf"
            filepath = self.download_dir / filename
            
            # Skip if already downloaded and unchanged
            if filepath.exists() and policy['id'] in self.last_run_state["known_policies"]:
                if self.last_run_state["known_policies"][policy['id']]['date'] == policy['date']:
                    logger.info(f"Policy already current: {filename}")
                    return filepath
            
            # Download PDF
            async with session.get(policy['url']) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Verify it's actually a PDF
                    if content.startswith(b'%PDF'):
                        filepath.write_bytes(content)
                        logger.info(f"Downloaded: {filename} ({len(content)} bytes)")
                        return filepath
                    else:
                        logger.warning(f"URL did not return PDF: {policy['url']}")
                else:
                    logger.error(f"Failed to download {policy['url']}: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Error downloading {policy['code']}: {e}")
            
        return None
    
    async def prepare_for_vectorization(self, new_policies: List[Dict], 
                                        updated_policies: List[Dict]) -> pd.DataFrame:
        """
        Prepare policy metadata for Pinecone ingestion
        """
        all_changed = new_policies + updated_policies
        vector_data = []
        
        for policy in all_changed:
            filepath = self.download_dir / f"{policy['code']}_{policy['date'].replace('.', '')}.pdf"
            
            vector_data.append({
                'id': policy['id'],
                'code': policy['code'],
                'title': policy['title'],
                'effective_date': policy['date'],
                'file_path': str(filepath) if filepath.exists() else None,
                'change_type': 'new' if policy in new_policies else 'update',
                'detected_at': policy['detected_at'],
                'metadata': json.dumps({
                    'source_url': policy['url'],
                    'policy_type': 'medicare_advantage',
                    'payer': 'unitedhealthcare'
                })
            })
        
        df = pd.DataFrame(vector_data)
        
        # Save for downstream agents
        if not df.empty:
            df.to_csv(self.download_dir / "policies_for_vectorization.csv", index=False)
            logger.info(f"Prepared {len(df)} policies for vectorization")
        
        return df
    
    async def run(self):
        """
        Main crawler execution
        """
        logger.info("Starting UHC Policy Crawler")
        
        async with async_playwright() as p:
            # Launch browser with specific options for healthcare site
            browser = await p.chromium.launch(
                headless=False,  # Set to True in production
                args=['--disable-blink-features=AutomationControlled']
            )
            
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            page = await context.new_page()
            
            try:
                # Navigate to the page
                logger.info(f"Navigating to {self.target_url}")
                await page.goto(self.target_url, wait_until='networkidle')
                
                # Handle consent dialog
                await self.handle_consent_dialog(page)
                
                # Extract all policies
                current_policies = await self.extract_policy_links(page)
                logger.info(f"Extracted {len(current_policies)} policies")
                
                # Detect changes
                new_policies, updated_policies = self.detect_changes(current_policies)
                
                if new_policies or updated_policies:
                    logger.info(f"Changes detected: {len(new_policies)} new, {len(updated_policies)} updated")
                    
                    # Download changed policies
                    async with aiohttp.ClientSession() as session:
                        download_tasks = []
                        for policy in new_policies + updated_policies:
                            task = self.download_policy(policy, session)
                            download_tasks.append(task)
                        
                        await asyncio.gather(*download_tasks)
                    
                    # Prepare for vectorization
                    vector_df = await self.prepare_for_vectorization(new_policies, updated_policies)
                    
                    # Update state
                    known_policies = {p['id']: p for p in current_policies}
                    self._save_state(known_policies)
                    
                    # Output for Agent 2
                    print("\n" + "="*50)
                    print("AGENT 1 OUTPUT: Policies Ready for Vectorization")
                    print("="*50)
                    if not vector_df.empty:
                        for _, row in vector_df.iterrows():
                            print(f"\n📄 {row['change_type'].upper()}: {row['code']} - {row['title']}")
                            print(f"   Effective: {row['effective_date']}")
                            print(f"   File: {row['file_path']}")
                    else:
                        print("No policy changes detected")
                    
                    return vector_df.to_dict('records') if not vector_df.empty else []
                else:
                    logger.info("No policy changes detected")
                    return []
                    
            except Exception as e:
                logger.error(f"Crawler failed: {e}", exc_info=True)
                raise
            finally:
                await browser.close()

# Entry point for the agent
async def main():
    """
    Main entry point for Agent 1 execution
    """
    crawler = UHCPolicyCrawler()
    
    try:
        changed_policies = await crawler.run()
        
        # Output in format expected by Agent 2
        if changed_policies:
            print("\n📤 Sending to Agent 2 (Policy Change Detector):")
            for policy in changed_policies:
                print(f"  → {policy['code']}: {policy['change_type']}")
        
        return changed_policies
        
    except Exception as e:
        logger.error(f"Agent 1 execution failed: {e}")
        return []

if __name__ == "__main__":
    # Run the agent
    asyncio.run(main())