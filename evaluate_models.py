"""
DeepEval Evaluation Script for Medical Claims Models
Evaluates multiple models across 100 subfolders with two prompt versions
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

# DeepEval imports
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# LangChain/Groq imports
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_ROOT = "evaluation/data"
PROMPTS_ROOT = "evaluation/prompts"
RESULTS_FILE = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Models to evaluate
MODELS = [
    {
        "name": "llama-3.1-8b-instant",
        "provider": "groq",
        "api_key": os.getenv("GROQ_API_KEY"),
        "temperature": 0.1
    },
    {
        "name": "medgamma-ollama",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "temperature": 0.1
    },
    {
        "name": "llama-3-8b",
        "provider": "groq",
        "api_key": os.getenv("GROQ_API_KEY"),
        "temperature": 0.1
    }
]

class ModelEvaluator:
    """Evaluates models using DeepEval metrics"""
    
    def __init__(self):
        self.models = {}
        self.prompts = {}
        self.results = []
        self.test_cases = []
        
        # Initialize metrics
        self.metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
            ContextualPrecisionMetric(threshold=0.7),
            ContextualRecallMetric(threshold=0.7),
            HallucinationMetric(threshold=0.3)
        ]
        
        # Load prompts
        self.load_prompts()
        
    def load_prompts(self):
        """Load prompt templates from files"""
        prompt_files = {
            "prompt1": "prompt1_baseline.txt",
            "prompt2": "prompt2_monte_carlo.txt"
        }
        
        for name, filename in prompt_files.items():
            path = os.path.join(PROMPTS_ROOT, filename)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.prompts[name] = f.read()
                print(f"✅ Loaded {name}")
            else:
                print(f"⚠️ Prompt file not found: {path}")
    
    def initialize_model(self, model_config: Dict):
        """Initialize LLM based on provider"""
        provider = model_config["provider"]
        model_name = model_config["name"]
        
        try:
            if provider == "groq":
                llm = ChatGroq(
                    api_key=model_config["api_key"],
                    model=model_name,
                    temperature=model_config.get("temperature", 0.1),
                    max_tokens=2000
                )
            elif provider == "ollama":
                llm = ChatOllama(
                    model=model_name,
                    base_url=model_config.get("base_url", "http://localhost:11434"),
                    temperature=model_config.get("temperature", 0.1)
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            self.models[model_name] = llm
            print(f"✅ Initialized {model_name} ({provider})")
            return llm
        except Exception as e:
            print(f"❌ Failed to initialize {model_name}: {e}")
            return None
    
    def load_test_data(self) -> List[Dict]:
        """Load test data from 100 subfolders"""
        test_data = []
        
        # Find all subfolders
        subfolders = [d for d in Path(DATA_ROOT).iterdir() if d.is_dir()]
        subfolders = sorted(subfolders)[:100]  # Limit to 100
        
        print(f"\n📂 Loading test data from {len(subfolders)} subfolders...")
        
        for folder in tqdm(subfolders):
            input_file = folder / "input.txt"
            expected_file = folder / "expected_output.txt"
            
            if input_file.exists() and expected_file.exists():
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_text = f.read()
                
                with open(expected_file, 'r', encoding='utf-8') as f:
                    expected = f.read()
                
                # Parse input (assuming JSON format)
                try:
                    claim_data = json.loads(input_text)
                except:
                    # Fallback: treat as plain text
                    claim_data = {"text": input_text, "patient_id": "unknown", "code": "unknown"}
                
                test_data.append({
                    "folder": folder.name,
                    "input": input_text,
                    "claim_data": claim_data,
                    "expected": expected,
                    "expected_decision": json.loads(expected).get("decision", "UNKNOWN") if expected.startswith("{") else "UNKNOWN"
                })
        
        print(f"✅ Loaded {len(test_data)} test cases")
        return test_data
    
    def format_prompt(self, prompt_template: str, claim_data: Dict, context: Dict = None) -> str:
        """Format prompt with actual data"""
        # Mock data for missing fields (in production, these would come from your database)
        mock_patient = {
            "id": claim_data.get("patient_id", "pat001"),
            "age": 65,
            "gender": "M",
            "conditions": ["hypertension", "diabetes"]
        }
        
        mock_policy = """
        Policy J3420: Vitamin B12 injections
        - Dose > 1000mcg requires prior authorization
        - Maximum 12 injections per year
        
        Policy 97110: Physical Therapy
        - Limited to 12 visits per year
        - Requires documentation of functional improvement
        """
        
        return prompt_template.format(
            patient_info=json.dumps(mock_patient, indent=2),
            claim_details=json.dumps(claim_data, indent=2),
            previous_procedures=json.dumps([], indent=2),
            policy_rules=mock_policy,
            procedure_code=claim_data.get("code", "unknown")
        )
    
    async def get_model_response(self, model_name: str, prompt: str) -> str:
        """Get response from model"""
        llm = self.models.get(model_name)
        if not llm:
            return json.dumps({"error": "Model not initialized"})
        
        try:
            response = await llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            print(f"❌ Error from {model_name}: {e}")
            return json.dumps({"error": str(e)})
    
    def parse_response(self, response: str) -> Dict:
        """Parse model response to extract decision"""
        try:
            # Try to parse as JSON
            return json.loads(response)
        except:
            # Fallback: extract using simple rules
            response_lower = response.lower()
            if "approve" in response_lower:
                return {"decision": "APPROVED", "raw": response[:200]}
            elif "reject" in response_lower:
                return {"decision": "REJECTED", "raw": response[:200]}
            else:
                return {"decision": "UNKNOWN", "raw": response[:200]}
    
    def create_test_case(self, input_text: str, actual_output: str, expected_output: str, context: str = "") -> LLMTestCase:
        """Create a DeepEval test case"""
        return LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            context=[context]
        )
    
    async def evaluate_combination(self, model_name: str, prompt_name: str, test_data: List[Dict]) -> Dict:
        """Evaluate one model-prompt combination"""
        print(f"\n🔍 Evaluating {model_name} with {prompt_name}...")
        
        prompt_template = self.prompts[prompt_name]
        results = []
        metrics_scores = {metric.__class__.__name__: [] for metric in self.metrics}
        
        for i, test in enumerate(tqdm(test_data, desc=f"{model_name} - {prompt_name}")):
            # Format prompt
            formatted_prompt = self.format_prompt(prompt_template, test["claim_data"])
            
            # Get model response
            response = await self.get_model_response(model_name, formatted_prompt)
            parsed = self.parse_response(response)
            
            # Create test case
            test_case = self.create_test_case(
                input_text=formatted_prompt[:500],
                actual_output=json.dumps(parsed),
                expected_output=test["expected"],
                context=test["claim_data"].get("code", "unknown")
            )
            
            # Calculate metrics
            test_metrics = {}
            for metric in self.metrics:
                try:
                    score = await metric.a_measure(test_case)
                    test_metrics[metric.__class__.__name__] = score
                    metrics_scores[metric.__class__.__name__].append(score)
                except Exception as e:
                    print(f"  ⚠️ Metric error: {e}")
            
            # Check if decision matches expected
            expected_decision = test["expected_decision"]
            actual_decision = parsed.get("decision", "UNKNOWN")
            accuracy = 1.0 if actual_decision == expected_decision else 0.0
            
            results.append({
                "test_case": i,
                "folder": test["folder"],
                "expected_decision": expected_decision,
                "actual_decision": actual_decision,
                "accuracy": accuracy,
                "metrics": test_metrics,
                "parsed_response": parsed
            })
        
        # Calculate averages
        avg_accuracy = sum(r["accuracy"] for r in results) / len(results) * 100
        avg_metrics = {
            name: sum(scores) / len(scores) if scores else 0
            for name, scores in metrics_scores.items()
        }
        
        return {
            "model": model_name,
            "prompt": prompt_name,
            "accuracy": avg_accuracy,
            "metrics": avg_metrics,
            "total_tests": len(results),
            "correct": sum(r["accuracy"] for r in results),
            "results": results
        }
    
    async def run_full_evaluation(self):
        """Run complete evaluation across all models and prompts"""
        print("\n" + "="*80)
        print("🚀 Starting DeepEval Model Evaluation")
        print("="*80)
        
        # Initialize all models
        print("\n🔧 Initializing models...")
        for model_config in MODELS:
            self.initialize_model(model_config)
        
        # Load test data
        test_data = self.load_test_data()
        
        if not test_data:
            print("❌ No test data found!")
            return
        
        # Run evaluations
        print("\n📊 Running evaluations...")
        tasks = []
        for model_config in MODELS:
            model_name = model_config["name"]
            if model_name not in self.models:
                continue
            
            for prompt_name in self.prompts.keys():
                tasks.append(self.evaluate_combination(model_name, prompt_name, test_data))
        
        results = await asyncio.gather(*tasks)
        
        # Compile final results
        self.results = results
        self.save_results()
        self.print_summary()
        
        return results
    
    def save_results(self):
        """Save evaluation results to file"""
        summary = []
        for result in self.results:
            summary.append({
                "model": result["model"],
                "prompt": result["prompt"],
                "accuracy": result["accuracy"],
                "metrics": result["metrics"],
                "total_tests": result["total_tests"],
                "correct": result["correct"]
            })
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": len(self.results[0]["results"]) if self.results else 0,
            "summary": summary,
            "detailed_results": self.results
        }
        
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {RESULTS_FILE}")
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("📈 EVALUATION SUMMARY")
        print("="*80)
        
        # Create DataFrame for easy viewing
        rows = []
        for result in self.results:
            rows.append({
                "Model": result["model"],
                "Prompt": "Baseline" if "prompt1" in result["prompt"] else "Monte Carlo",
                "Accuracy (%)": f"{result['accuracy']:.2f}%",
                "Answer Relevancy": f"{result['metrics'].get('AnswerRelevancyMetric', 0):.3f}",
                "Faithfulness": f"{result['metrics'].get('FaithfulnessMetric', 0):.3f}",
                "Hallucination": f"{result['metrics'].get('HallucinationMetric', 0):.3f}",
                "Correct/Total": f"{result['correct']:.0f}/{result['total_tests']}"
            })
        
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        
        # Find best performer
        best = max(self.results, key=lambda x: x["accuracy"])
        print(f"\n🏆 Best Model: {best['model']} with {best['prompt']}")
        print(f"   Accuracy: {best['accuracy']:.2f}%")
        
        # Compare prompt versions
        print("\n📊 Prompt Comparison:")
        baseline_acc = next(r["accuracy"] for r in self.results if "prompt1" in r["prompt"] and "llama-3.1" in r["model"])
        monte_carlo_acc = next(r["accuracy"] for r in self.results if "prompt2" in r["prompt"] and "llama-3-8b" in r["model"])
        
        improvement = monte_carlo_acc - baseline_acc
        print(f"   Baseline (Llama 3.1): {baseline_acc:.2f}%")
        print(f"   Monte Carlo (Llama 3): {monte_carlo_acc:.2f}%")
        print(f"   Improvement: +{improvement:.2f}%")

async def main():
    """Main execution"""
    evaluator = ModelEvaluator()
    await evaluator.run_full_evaluation()

if __name__ == "__main__":
    asyncio.run(main())