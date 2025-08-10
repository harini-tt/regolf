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
        
        # Adaptive rate limiting
        self.rate_limit_detected = False
        self.current_concurrency = None
        
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
        
    async def generate_examples_for_regex_with_retry(self, regex_pattern: str, description: str, examples: List[str] = None, max_retries: int = 6) -> Optional[Dict[str, Any]]:
        """Generate training examples with fast retry logic - 1 minute max total time"""
        start_time = time.time()
        max_total_time = 60  # 1 minute max total time per pattern
        
        for attempt in range(max_retries):
            # Check if we've exceeded total time limit
            elapsed = time.time() - start_time
            if elapsed > max_total_time:
                logger.warning(f"⏱️ Time limit exceeded for pattern {regex_pattern} after {elapsed:.1f}s")
                break
                
            try:
                # Minimal jitter to avoid thundering herd
                initial_delay = random.uniform(0, 0.2)
                await asyncio.sleep(initial_delay)
                
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
                    timeout=30.0,  # Short timeout
                    **token_param
                )
                
                content = response.choices[0].message.content
                result = self._extract_json_from_response(content, regex_pattern)
                
                if result:
                    logger.debug(f"✅ Success for pattern {regex_pattern} on attempt {attempt + 1} ({elapsed:.1f}s)")
                    return result
                else:
                    logger.warning(f"Invalid JSON for pattern {regex_pattern}, attempt {attempt + 1}/{max_retries}")
                    
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e).lower()
                
                # Detect rate limiting specifically
                is_rate_limit = any(keyword in error_msg for keyword in [
                    'rate limit', 'rate_limit', 'too many requests', 'quota', 'usage limit'
                ])
                
                if is_rate_limit:
                    # Short backoff for rate limits - max 10 seconds
                    wait_time = min(3 + (attempt * 2), 10)
                    logger.warning(f"🚫 Rate limit for {regex_pattern}, attempt {attempt + 1}, waiting {wait_time}s")
                    self.rate_limit_detected = True
                else:
                    # Very short backoff for other errors - max 5 seconds
                    wait_time = min(1 + attempt, 5)
                    logger.warning(f"{error_type} for {regex_pattern}, attempt {attempt + 1}, waiting {wait_time}s")
                
                # Check if waiting would exceed time limit
                elapsed = time.time() - start_time
                if elapsed + wait_time > max_total_time:
                    logger.warning(f"⏱️ Skipping wait to avoid time limit for {regex_pattern}")
                    break
                    
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                    
        logger.error(f"❌ Failed {regex_pattern} after {time.time() - start_time:.1f}s")
        return None

    async def process_single_pattern(self, item: Dict[str, Any], index: int, semaphore: asyncio.Semaphore) -> tuple[Optional[Dict[str, Any]], bool]:
        """Process a single regex pattern with concurrency control. Returns (result, was_skipped)"""
        async with semaphore:
            regex = item["regex"]
            description = item["description"]
            examples = item.get("examples", [])
            
            # Check if result file already exists
            temp_dir = self.output_file.parent / "temp_results"
            temp_file = temp_dir / f"result_{index:06d}.json"
            
            if temp_file.exists():
                try:
                    # Load existing result
                    async with aiofiles.open(temp_file, 'r') as f:
                        content = await f.read()
                        existing_result = json.loads(content)
                    logger.debug(f"✅ Skipping pattern {index} (already exists): {regex}")
                    return existing_result, True  # True indicates skipped
                except Exception as e:
                    logger.warning(f"Failed to load existing file {temp_file}, will regenerate: {e}")
                    # Continue to generate new result if file is corrupted
            
            logger.info(f"Processing {index}: {regex}")
            
            result = await self.generate_examples_for_regex_with_retry(regex, description, examples)
            
            if result:
                logger.info(f"✅ Generated examples for: {regex}")
                # Save individual result to avoid corruption
                await self._save_individual_result(result, index)
                return result, False  # False indicates newly generated
            else:
                logger.warning(f"❌ Failed to generate examples for: {regex}")
                return None, False

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
            
    async def process_dataset(self, max_patterns: Optional[int] = None, max_concurrent: int = 200) -> List[Dict[str, Any]]:
        """Process the entire dataset with async parallel processing"""
        logger.info("Starting async dataset processing...")
        
        # Load all regex patterns
        regex_data = self.load_regex_dataset()
        
        if max_patterns:
            regex_data = regex_data[:max_patterns]
            logger.info(f"Processing first {max_patterns} patterns")
            
        # Check for existing checkpoints to resume from
        existing_results = await self._load_latest_checkpoint()
        if existing_results:
            logger.info(f"Found existing checkpoint with {len(existing_results)} results. Resuming...")
            # Skip already processed patterns
            regex_data = regex_data[len(existing_results):]
            logger.info(f"Remaining patterns to process: {len(regex_data)}")
        else:
            existing_results = []
        
        # Initialize adaptive concurrency
        self.current_concurrency = max_concurrent
        attempt = 0
        max_attempts = 3  # Try up to 3 times with reduced concurrency
        
        while attempt < max_attempts and len(regex_data) > 0:
            attempt += 1
            
            # Reduce concurrency if we've hit rate limits before
            if self.rate_limit_detected and attempt > 1:
                self.current_concurrency = max(self.current_concurrency // 2, 50)  # Never go below 50
                logger.warning(f"🔄 Reducing concurrency to {self.current_concurrency} due to rate limits")
                self.rate_limit_detected = False  # Reset flag
            
            logger.info(f"Processing attempt {attempt}/{max_attempts} with {self.current_concurrency} concurrent requests")
            
            # Process remaining patterns
            new_results = await self._process_batch(regex_data, existing_results, self.current_concurrency)
            
            # Update results and find what's left to process
            all_results = existing_results + new_results
            completed_count = len(new_results)
            
            if completed_count == len(regex_data):
                # All done!
                logger.info("🎉 All patterns completed successfully!")
                break
            elif completed_count > 0:
                # Some progress made, continue with remaining
                logger.info(f"Completed {completed_count}/{len(regex_data)} patterns in this batch")
                existing_results = all_results
                # Find patterns that still need processing (this is a simplified approach)
                # In practice, we'd track which specific patterns failed
                if self.rate_limit_detected:
                    logger.info("Rate limits detected, will retry remaining patterns with reduced concurrency")
                    continue
                else:
                    break  # Success without rate limits
            else:
                logger.warning(f"No progress made in attempt {attempt}, will retry with reduced concurrency")
        
        # Final merge of all results
        await self._merge_temp_results()
        return all_results if 'all_results' in locals() else existing_results

    async def _scan_existing_results(self, total_patterns: int) -> int:
        """Scan for existing result files and return count"""
        temp_dir = self.output_file.parent / "temp_results"
        if not temp_dir.exists():
            return 0
            
        existing_files = list(temp_dir.glob("result_*.json"))
        existing_count = len(existing_files)
        
        if existing_count > 0:
            logger.info(f"📁 Found {existing_count} existing result files")
            logger.info(f"📊 Progress: {existing_count}/{total_patterns} ({existing_count/total_patterns*100:.1f}%) already completed")
        
        return existing_count

    async def _process_batch(self, regex_data: List[Dict[str, Any]], existing_results: List[Dict[str, Any]], max_concurrent: int) -> List[Dict[str, Any]]:
        """Process a batch of regex patterns with the given concurrency"""
        
        # Scan for existing result files
        total_patterns = len(regex_data) + len(existing_results)
        existing_file_count = await self._scan_existing_results(total_patterns)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for all patterns
        tasks = []
        for i, item in enumerate(regex_data):
            task = self.process_single_pattern(item, i + len(existing_results), semaphore)
            tasks.append(task)
        
        # Run all tasks concurrently with progress bar
        logger.info(f"Starting {len(tasks)} concurrent tasks...")
        results = []
        skipped_count = 0
        
        # Use asyncio.as_completed with tqdm for progress tracking
        completed_tasks = 0
        total_tasks = len(tasks)
        
        with tqdm(total=total_tasks, desc="Processing patterns") as pbar:
            for coro in asyncio.as_completed(tasks):
                result, was_skipped = await coro
                if result is not None:
                    results.append(result)
                    # Check if this was a skipped file vs newly generated
                    if was_skipped:
                        skipped_count += 1
                completed_tasks += 1
                pbar.update(1)
                
                # Save checkpoint every 500 completed patterns
                if completed_tasks % 500 == 0:
                    checkpoint_results = existing_results + results
                    await self._save_checkpoint(checkpoint_results, len(existing_results) + completed_tasks)
        
        logger.info(f"Successfully processed {len(results)}/{len(regex_data)} patterns in this batch")
        if skipped_count > 0:
            logger.info(f"📁 Skipped {skipped_count} patterns (files already existed)")
        
        return results

    async def _load_latest_checkpoint(self) -> List[Dict[str, Any]]:
        """Load the latest checkpoint if it exists."""
        temp_dir = self.output_file.parent / "temp_results"
        if not temp_dir.exists():
            return []
            
        checkpoint_files = sorted(temp_dir.glob("checkpoint_*.json"))
        if not checkpoint_files:
            return []
            
        latest_checkpoint = checkpoint_files[-1]
        try:
            async with aiofiles.open(latest_checkpoint, 'r') as f:
                content = await f.read()
                results = json.loads(content)
                logger.info(f"Loaded checkpoint from {latest_checkpoint}")
                return results
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {latest_checkpoint}: {e}")
            return []

    async def _save_checkpoint(self, results: List[Dict[str, Any]], completed_count: int):
        """Save a checkpoint of the current results to a temporary file."""
        temp_dir = self.output_file.parent / "temp_results"
        temp_dir.mkdir(exist_ok=True)
        checkpoint_file = temp_dir / f"checkpoint_{completed_count:06d}.json"
        
        async with aiofiles.open(checkpoint_file, 'w') as f:
            await f.write(json.dumps(results, indent=2))
        logger.info(f"Saved checkpoint at {completed_count} patterns.")

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
        "max_concurrent": 500,  # Maximum safe concurrency for high-tier accounts
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
    results = await generator.process_dataset(
        max_patterns=config["max_patterns"],
        max_concurrent=config["max_concurrent"]
    )

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
