#!/usr/bin/env python3
"""
GPT API Data Generator for Regex Golf RL/LoRA Training

This script processes each row from innovatorved/regex_dataset individually,
makes separate GPT queries, and outputs training data in the format:
{expr: str, yes: [list], no: [list]}
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegexGolfDataGenerator:
    """Generates regex golf training data by processing each dataset row individually"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4-turbo-preview",
                 output_file: str = "./generated_data/regex_golf_dataset.json"):
        """
        Initialize the data generator
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            model: GPT model to use
            output_file: Path to save the final JSON dataset
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests
        
    def load_regex_dataset(self) -> List[Dict[str, Any]]:
        """Load all rows from innovatorved/regex_dataset"""
        logger.info("Loading innovatorved/regex_dataset from Hugging Face...")
        dataset = load_dataset("innovatorved/regex_dataset", split="train")
        
        regex_data = []
        for item in dataset:
            if "regex" in item and item["regex"]:
                # Extract examples from the text column
                text_examples = []
                if "text" in item and item["text"]:
                    # Split text by newlines and filter out empty strings
                    text_examples = [line.strip() for line in item["text"].split('\n') if line.strip()]
                
                regex_data.append({
                    "regex": item["regex"],
                    "description": item.get("description", "No description provided"),
                    "examples": text_examples
                })
                
        logger.info(f"Loaded {len(regex_data)} regex patterns")
        return regex_data
        
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
        
    def generate_examples_for_regex(self, regex_pattern: str, description: str, examples: List[str] = None) -> Optional[Dict[str, Any]]:
        """Generate training examples for a single regex pattern"""
        self._rate_limit()
        
        prompt = self._create_prompt(regex_pattern, description, examples)
        
        try:
            # GPT-5 (o1) models use max_completion_tokens, others use max_tokens
            token_param = {}
            if "gpt-5" in self.model or "o1" in self.model:
                token_param["max_completion_tokens"] = 2000
            else:
                token_param["max_tokens"] = 2000
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                **token_param
            )
            
            content = response.choices[0].message.content
            result = self._extract_json_from_response(content, regex_pattern)
            return result
            
        except Exception as e:
            logger.error(f"Error generating examples for pattern {regex_pattern}: {e}")
            return None
            
    def _get_system_prompt(self) -> str:
        """Get the system prompt for GPT"""
        return """You are an expert in regular expressions and pattern matching. 
Your task is to generate training examples for regex golf - finding strings that match 
and don't match a given regex pattern.

For each regex pattern, provide:
1. A list of strings that SHOULD match the pattern (yes examples)
2. A list of strings that should NOT match the pattern (no examples)

Focus on:
- Creating diverse, realistic examples
- Including edge cases and boundary conditions
- Making examples educational and challenging
- Ensuring examples are correct and well-tested

Always respond with valid JSON in the exact format requested."""

    def _create_prompt(self, regex_pattern: str, description: str, examples: List[str] = None) -> str:
        """Create a prompt for generating examples for a specific regex"""
        
        # Load prompt template from file
        prompt_file = Path(__file__).parent / "prompt.txt"
        
        try:
            with open(prompt_file, 'r') as f:
                prompt_template = f.read()
            
            # Replace the placeholder with the actual regex pattern
            prompt = prompt_template.format(regex_pattern=regex_pattern)
            
            # Add examples from the dataset if available
            if examples and len(examples) > 0:
                examples_text = "\n\nThe following are known examples that should match this regex:\n"
                for i, example in enumerate(examples[:10], 1):  # Limit to first 10 examples
                    examples_text += f"{i}. `{example}`\n"
                examples_text += "\nPlease use these as reference and include some of them in your 'yes' examples, but also generate additional diverse examples to ensure comprehensive coverage."
                prompt += examples_text
            
            return prompt
            
        except FileNotFoundError:
            return "fnfe"
        
    def _extract_json_from_response(self, content: str, regex_pattern: str) -> Optional[Dict[str, Any]]:
        """Extract and validate JSON from GPT response"""
        try:
            # Find JSON object in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.error(f"No JSON found in response for pattern: {regex_pattern}")
                return None
                
            json_str = content[json_start:json_end]
            data = json.loads(json_str)
            
            # Validate the structure
            if not all(key in data for key in ["expr", "yes", "no"]):
                logger.error(f"Missing required keys in JSON for pattern: {regex_pattern}")
                return None
                
            if not isinstance(data["yes"], list) or not isinstance(data["no"], list):
                logger.error(f"'yes' and 'no' must be lists for pattern: {regex_pattern}")
                return None
                
            # Validate that the regex actually works
            if self._validate_examples(data["expr"], data["yes"], data["no"]):
                return data
            else:
                logger.warning(f"Validation failed for pattern: {regex_pattern}")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for pattern {regex_pattern}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing pattern {regex_pattern}: {e}")
            return None
            
    def _validate_examples(self, regex_pattern: str, yes_examples: List[str], no_examples: List[str]) -> bool:
        """Validate that the examples are correct for the regex pattern"""
        try:
            pattern = re.compile(regex_pattern)
            
            # Check that yes examples match
            for example in yes_examples[:5]:  # Check first 5
                if not pattern.search(example):
                    logger.warning(f"Yes example '{example}' doesn't match pattern '{regex_pattern}'")
                    return False
                    
            # Check that at least some no examples don't match
            non_matches = 0
            for example in no_examples[:5]:  # Check first 5
                if not pattern.search(example):
                    non_matches += 1
                    
            if non_matches == 0:
                logger.warning(f"All no examples match pattern '{regex_pattern}'")
                return False
                
            return True
            
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{regex_pattern}': {e}")
            return False
            
    def process_dataset(self, max_patterns: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process the entire dataset, making individual GPT queries for each regex"""
        logger.info("Starting dataset processing...")
        
        # Load all regex patterns
        regex_data = self.load_regex_dataset()
        
        if max_patterns:
            regex_data = regex_data[:max_patterns]
            logger.info(f"Processing first {max_patterns} patterns")
            
        results = []
        
        for i, item in enumerate(tqdm(regex_data, desc="Processing patterns")):
            regex = item["regex"]
            description = item["description"]
            examples = item.get("examples", [])
            
            logger.info(f"Processing {i+1}/{len(regex_data)}: {regex}")
            
            # Generate examples for this specific regex
            result = self.generate_examples_for_regex(regex, description, examples)
            
            if result:
                results.append(result)
                logger.info(f"✅ Generated examples for: {regex}")
            else:
                logger.warning(f"❌ Failed to generate examples for: {regex}")
                
            # Save intermediate results every 50 patterns
            if i % 50 == 0 and results:
                self._save_intermediate_results(results, i)
                
        logger.info(f"Successfully processed {len(results)}/{len(regex_data)} patterns")
        return results
        
    def _save_intermediate_results(self, results: List[Dict[str, Any]], index: int):
        """Save intermediate results"""
        intermediate_file = self.output_file.parent / f"intermediate_results_{index}.json"
        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved intermediate results to {intermediate_file}")
        
    def save_final_dataset(self, results: List[Dict[str, Any]]):
        """Save the final dataset"""
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved final dataset with {len(results)} examples to {self.output_file}")
        
        # Print some statistics
        total_yes = sum(len(item["yes"]) for item in results)
        total_no = sum(len(item["no"]) for item in results)
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total patterns: {len(results)}")
        logger.info(f"  Total yes examples: {total_yes}")
        logger.info(f"  Total no examples: {total_no}")
        logger.info(f"  Average yes per pattern: {total_yes/len(results):.1f}")
        logger.info(f"  Average no per pattern: {total_no/len(results):.1f}")

def main():
    """Main function to run the data generation"""
    # Configuration
    config = {
        "model": "gpt-4o",
        "max_patterns": 5,
        "output_file": "./generated_data/regex_golf_dataset.json"
    }
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.info("Please set your OpenAI API key:")
        logger.info("export OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize generator
    generator = RegexGolfDataGenerator(
        model=config["model"],
        output_file=config["output_file"]
    )
    
    # Process the dataset
    results = generator.process_dataset(max_patterns=config["max_patterns"])
    
    # Save final results
    if results:
        generator.save_final_dataset(results)
        logger.info("🎉 Dataset generation completed successfully!")
    else:
        logger.error("❌ No examples generated! Check your OpenAI API key and quota.")

if __name__ == "__main__":
    main()
