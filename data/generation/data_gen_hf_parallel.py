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
import asyncio
import aiofiles
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import random

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
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
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
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
        
    async def generate_examples_for_regex(self, regex_pattern: str, description: str, examples: List[str] = None) -> Optional[Dict[str, Any]]:
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
            
            response = await self.client.chat.completions.create(
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

    async def generate_examples_for_regex_with_retry(self, regex_pattern: str, description: str, examples: List[str] = None, max_retries: int = 10) -> Optional[Dict[str, Any]]:
        """Generate training examples with retry logic and exponential backoff"""
        for attempt in range(max_retries):
            try:
                # Add some jitter to avoid thundering herd
                await asyncio.sleep(random.uniform(0, 0.5))
                
                prompt = self._create_prompt(regex_pattern, description, examples)
                
                # GPT-5 (o1) models use max_completion_tokens, others use max_tokens
                token_param = {}
                if "gpt-5" in self.model or "o1" in self.model:
                    token_param["max_completion_tokens"] = 2000
                else:
                    token_param["max_tokens"] = 2000
                
                response = await self.client.chat.completions.create(
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
                
                if result:
                    return result
                else:
                    logger.warning(f"Failed to extract valid JSON for pattern {regex_pattern}, attempt {attempt + 1}")
                    
            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                logger.warning(f"Error generating examples for pattern {regex_pattern}, attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate examples for pattern {regex_pattern} after {max_retries} attempts")
                    
        return None

    async def process_single_pattern(self, item: Dict[str, Any], index: int, semaphore: asyncio.Semaphore) -> Optional[Dict[str, Any]]:
        """Process a single regex pattern with concurrency control"""
        async with semaphore:
            regex = item["regex"]
            description = item["description"]
            examples = item.get("examples", [])
            
            logger.info(f"Processing {index}: {regex}")
            
            result = await self.generate_examples_for_regex_with_retry(regex, description, examples)
            
            if result:
                logger.info(f"✅ Generated examples for: {regex}")
                # Save individual result to avoid corruption
                await self._save_individual_result(result, index)
                return result
            else:
                logger.warning(f"❌ Failed to generate examples for: {regex}")
                return None

    async def _save_individual_result(self, result: Dict[str, Any], index: int):
        """Save individual result to a temporary file"""
        temp_dir = self.output_file.parent / "temp_results"
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"result_{index:06d}.json"
        
        async with aiofiles.open(temp_file, 'w') as f:
            await f.write(json.dumps(result, indent=2))
            
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
            
    async def process_dataset(self, max_patterns: Optional[int] = None, max_concurrent: int = 50) -> List[Dict[str, Any]]:
        """Process the entire dataset with async parallel processing"""
        logger.info("Starting async dataset processing...")
        
        # Load all regex patterns
        regex_data = self.load_regex_dataset()
        
        if max_patterns:
            regex_data = regex_data[:max_patterns]
            logger.info(f"Processing first {max_patterns} patterns")
            
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for all patterns
        tasks = []
        for i, item in enumerate(regex_data):
            task = self.process_single_pattern(item, i, semaphore)
            tasks.append(task)
        
        # Run all tasks concurrently with progress bar
        logger.info(f"Starting {len(tasks)} concurrent tasks...")
        results = []
        
        # Use asyncio.as_completed with tqdm for progress tracking
        completed_tasks = 0
        total_tasks = len(tasks)
        
        with tqdm(total=total_tasks, desc="Processing patterns") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:
                    results.append(result)
                completed_tasks += 1
                pbar.update(1)
        
        logger.info(f"Successfully processed {len(results)}/{len(regex_data)} patterns")
        
        # Merge all individual result files
        await self._merge_temp_results()
        
        return results

    async def _merge_temp_results(self):
        """Merge all temporary result files into final output"""
        temp_dir = self.output_file.parent / "temp_results"
        if not temp_dir.exists():
            return
            
        all_results = []
        temp_files = sorted(temp_dir.glob("result_*.json"))
        
        for temp_file in temp_files:
            try:
                async with aiofiles.open(temp_file, 'r') as f:
                    content = await f.read()
                    result = json.loads(content)
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to read temp file {temp_file}: {e}")
        
        # Write merged results
        if all_results:
            async with aiofiles.open(self.output_file, 'w') as f:
                await f.write(json.dumps(all_results, indent=2))
            
            logger.info(f"Merged {len(all_results)} results into {self.output_file}")
            
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")
            
            # Remove temp directory if empty
            try:
                temp_dir.rmdir()
            except:
                pass
        
    def save_final_dataset(self, results: List[Dict[str, Any]]):
        """Save the final dataset and print statistics"""
        # Check if file already exists from async merge
        if self.output_file.exists():
            logger.info(f"Dataset already saved to {self.output_file} by async merge process")
        else:
            # Fallback: save synchronously if async merge failed
            with open(self.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved final dataset with {len(results)} examples to {self.output_file}")
        
        # Print some statistics
        if results:
            total_yes = sum(len(item["yes"]) for item in results)
            total_no = sum(len(item["no"]) for item in results)
            logger.info(f"Dataset statistics:")
            logger.info(f"  Total patterns: {len(results)}")
            logger.info(f"  Total yes examples: {total_yes}")
            logger.info(f"  Total no examples: {total_no}")
            logger.info(f"  Average yes per pattern: {total_yes/len(results):.1f}")
            logger.info(f"  Average no per pattern: {total_no/len(results):.1f}")
        else:
            logger.warning("No results to save statistics for")

async def main():
    """Main function to run the data generation"""

    start_time = time.time()

    # Configuration
    config = {
        "model": "gpt-4o",
        "max_patterns": 8550,
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
    results = await generator.process_dataset(max_patterns=config["max_patterns"])

    end_time = time.time()
    
    # Save final results
    if results:
        generator.save_final_dataset(results)
        logger.info("🎉 Dataset generation completed successfully!")
        print("time", end_time-start_time)
    else:
        logger.error("❌ No examples generated! Check your OpenAI API key and quota.")

if __name__ == "__main__":
    asyncio.run(main())
