#!/usr/bin/env python3
"""
GPT API Data Generator for Regex Golf RL/LoRA Training

This script uses GPT API to generate high-quality training examples for a 
regex golf approximation solver using RL/LoRA fine-tuning. It loads regex 
patterns from the innovatorved/regex_dataset on Hugging Face.
"""

import os
import json
import time
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from openai import OpenAI
from tqdm import tqdm
import re
from dotenv import load_dotenv
import jsonlines

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RegexGolfExample:
    """Represents a regex golf training example"""
    regex_pattern: str
    target_strings: List[str]
    non_target_strings: List[str]
    difficulty_level: str
    explanation: str
    optimization_hint: str
    character_budget: int
    expected_solution: str

class RegexGolfDataGenerator:
    """Generates high-quality regex golf training data using GPT API"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4-turbo-preview",
                 max_examples_per_regex: int = 3,
                 output_dir: str = "./generated_data"):
        """
        Initialize the data generator
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            model: GPT model to use
            max_examples_per_regex: Maximum training examples per regex pattern
            output_dir: Directory to save generated data
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_examples_per_regex = max_examples_per_regex
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests
        
    def load_regex_dataset(self) -> List[Dict[str, Any]]:
        """Load regex patterns from innovatorved/regex_dataset"""
        regex_patterns = []
        
        logger.info("Loading innovatorved/regex_dataset from Hugging Face...")
        dataset = load_dataset("innovatorved/regex_dataset", split="train")
        
        for item in dataset:
            if "regex" in item and item["regex"]:
                # Get description if available
                description = item.get("description", "No description provided")
                
                regex_patterns.append({
                    "pattern": item["regex"],
                    "description": description,
                    "source": "innovatorved/regex_dataset",
                    "metadata": {"description": description}
                })
                
        logger.info(f"Successfully loaded {len(regex_patterns)} regex patterns")
        return regex_patterns
        
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
        
    def generate_regex_golf_examples(self, regex_pattern: str, description: str) -> List[RegexGolfExample]:
        """Generate multiple regex golf training examples for a given pattern"""
        self._rate_limit()
        
        prompt = self._create_regex_golf_prompt(regex_pattern, description)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            examples = self._parse_gpt_response(content, regex_pattern)
            return examples
            
        except Exception as e:
            logger.error(f"Error generating examples for pattern {regex_pattern}: {e}")
            return []
            
    def _get_system_prompt(self) -> str:
        """Get the system prompt for GPT"""
        return """You are an expert in regex golf and reinforcement learning for code optimization. 
Your task is to generate high-quality training examples for an RL/LoRA model that learns to 
create efficient regex patterns. 

Regex golf is about finding the shortest possible regex that matches specific target strings 
while avoiding non-target strings. Focus on:

1. **Diversity**: Create varied difficulty levels and pattern types
2. **Educational Value**: Include examples that teach optimization strategies
3. **Realistic Constraints**: Use practical character budgets (10-50 characters)
4. **Clear Objectives**: Provide clear target/non-target distinctions
5. **Strategic Hints**: Include optimization strategies for RL learning

Format your response as valid JSON with multiple training examples."""

    def _create_regex_golf_prompt(self, regex_pattern: str, description: str) -> str:
        """Create a detailed prompt for generating regex golf examples"""
        
        prompt = f"""
Given this regex pattern: `{regex_pattern}`
Description: {description}

Generate {self.max_examples_per_regex} diverse regex golf training examples based on this pattern. 
Each example should be a learning scenario where an RL agent needs to discover an efficient regex.

For each example, provide:

1. **target_strings**: 5-8 strings that should match the pattern
2. **non_target_strings**: 5-8 strings that should NOT match 
3. **difficulty_level**: "beginner", "intermediate", "advanced", or "expert"
4. **explanation**: Why this is a good learning example (2-3 sentences)
5. **optimization_hint**: Specific strategy for making the regex shorter/more efficient
6. **character_budget**: Realistic limit (10-50 chars) for the solution
7. **expected_solution**: Your best regex solution within the budget

Make the examples progressively more challenging:
- **Beginner**: Simple patterns, obvious solutions
- **Intermediate**: Require some regex knowledge, multiple valid approaches  
- **Advanced**: Need optimization tricks, character classes, quantifiers
- **Expert**: Require deep regex mastery, creative shortcuts

Vary the scenarios:
- Different string lengths and patterns
- Mix of literal matches and pattern-based matches  
- Include edge cases and boundary conditions
- Create examples that teach specific regex concepts

Return ONLY a valid JSON array with this structure:
```json
[
  {{
    "target_strings": ["string1", "string2", ...],
    "non_target_strings": ["string1", "string2", ...], 
    "difficulty_level": "beginner|intermediate|advanced|expert",
    "explanation": "Educational value explanation",
    "optimization_hint": "Specific strategy for optimization",
    "character_budget": 25,
    "expected_solution": "optimized regex pattern"
  }},
  ...
]
```

Focus on creating examples that will help an RL agent learn:
- When to use character classes vs literals
- How to leverage quantifiers effectively  
- Boundary conditions and anchoring strategies
- Trade-offs between specificity and generality
- Creative use of lookarounds and groups
"""
        return prompt
        
    def _parse_gpt_response(self, content: str, original_pattern: str) -> List[RegexGolfExample]:
        """Parse GPT response into RegexGolfExample objects"""
        examples = []
        
        try:
            # Extract JSON from response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.error("No valid JSON found in GPT response")
                return examples
                
            json_str = content[json_start:json_end]
            data = json.loads(json_str)
            
            for item in data:
                try:
                    example = RegexGolfExample(
                        regex_pattern=original_pattern,
                        target_strings=item.get("target_strings", []),
                        non_target_strings=item.get("non_target_strings", []),
                        difficulty_level=item.get("difficulty_level", "intermediate"),
                        explanation=item.get("explanation", ""),
                        optimization_hint=item.get("optimization_hint", ""),
                        character_budget=item.get("character_budget", 30),
                        expected_solution=item.get("expected_solution", "")
                    )
                    
                    # Validate the example
                    if self._validate_example(example):
                        examples.append(example)
                    else:
                        logger.warning(f"Invalid example skipped for pattern: {original_pattern}")
                        
                except KeyError as e:
                    logger.error(f"Missing required field in example: {e}")
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.debug(f"Response content: {content}")
            
        return examples
        
    def _validate_example(self, example: RegexGolfExample) -> bool:
        """Validate a regex golf example"""
        # Check required fields
        if not all([
            example.target_strings,
            example.non_target_strings, 
            example.difficulty_level,
            example.expected_solution
        ]):
            return False
            
        # Check difficulty level
        valid_levels = ["beginner", "intermediate", "advanced", "expert"]
        if example.difficulty_level not in valid_levels:
            return False
            
        # Check character budget is reasonable
        if not (5 <= example.character_budget <= 100):
            return False
            
        # Test if expected solution actually works
        try:
            pattern = re.compile(example.expected_solution)
            
            # Check that it matches target strings
            for target in example.target_strings[:3]:  # Test first 3
                if not pattern.search(target):
                    logger.warning(f"Expected solution doesn't match target: {target}")
                    return False
                    
            # Check that it doesn't match non-target strings (at least some)
            non_matches = 0
            for non_target in example.non_target_strings[:3]:  # Test first 3
                if not pattern.search(non_target):
                    non_matches += 1
                    
            if non_matches == 0:
                logger.warning("Expected solution matches all non-target strings")
                return False
                
        except re.error:
            logger.warning(f"Invalid regex in expected solution: {example.expected_solution}")
            return False
            
        return True
        
    def save_examples(self, examples: List[RegexGolfExample], filename: str):
        """Save examples in multiple formats for RL training"""
        base_path = self.output_dir / filename
        
        # Save as JSON Lines (for streaming)
        jsonl_path = base_path.with_suffix('.jsonl')
        with jsonlines.open(jsonl_path, 'w') as writer:
            for example in examples:
                writer.write({
                    'regex_pattern': example.regex_pattern,
                    'target_strings': example.target_strings,
                    'non_target_strings': example.non_target_strings,
                    'difficulty_level': example.difficulty_level,
                    'explanation': example.explanation,
                    'optimization_hint': example.optimization_hint,
                    'character_budget': example.character_budget,
                    'expected_solution': example.expected_solution
                })
        
        # Save as regular JSON
        json_path = base_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump([{
                'regex_pattern': ex.regex_pattern,
                'target_strings': ex.target_strings,
                'non_target_strings': ex.non_target_strings,
                'difficulty_level': ex.difficulty_level,
                'explanation': ex.explanation,
                'optimization_hint': ex.optimization_hint,
                'character_budget': ex.character_budget,
                'expected_solution': ex.expected_solution
            } for ex in examples], f, indent=2)
            
        logger.info(f"Saved {len(examples)} examples to {base_path}.{{json,jsonl}}")
        
    def generate_dataset(self, max_patterns: Optional[int] = None) -> List[RegexGolfExample]:
        """Generate the complete training dataset"""
        logger.info("Starting regex golf dataset generation...")
        
        # Load regex patterns from innovatorved/regex_dataset
        regex_patterns = self.load_regex_dataset()
        
        if max_patterns:
            regex_patterns = regex_patterns[:max_patterns]
            
        all_examples = []
        
        for i, pattern_data in enumerate(tqdm(regex_patterns, desc="Generating examples")):
            pattern = pattern_data["pattern"]
            description = pattern_data["description"]
            
            logger.info(f"Processing pattern {i+1}/{len(regex_patterns)}: {pattern}")
            
            # Generate examples for this pattern
            examples = self.generate_regex_golf_examples(pattern, description)
            all_examples.extend(examples)
            
            # Save intermediate results every 10 patterns
            if i % 10 == 0 and all_examples:
                self.save_examples(all_examples, f"regex_golf_partial_{i}")
                
        # Save final dataset
        if all_examples:
            self.save_examples(all_examples, "regex_golf_complete")
            
        logger.info(f"Generated {len(all_examples)} total training examples")
        return all_examples
        
    def create_training_splits(self, examples: List[RegexGolfExample], 
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15):
        """Create train/validation/test splits for RL training"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # Shuffle examples
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_examples = shuffled[:n_train]
        val_examples = shuffled[n_train:n_train + n_val]
        test_examples = shuffled[n_train + n_val:]
        
        # Save splits
        self.save_examples(train_examples, "train_split")
        self.save_examples(val_examples, "val_split") 
        self.save_examples(test_examples, "test_split")
        
        logger.info(f"Created splits: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test")
        
        return train_examples, val_examples, test_examples

def main():
    """Main function to run the data generation"""
    # Configuration
    config = {
        "model": "gpt-4-turbo-preview",  # or "gpt-3.5-turbo" for faster/cheaper generation
        "max_examples_per_regex": 3,     # Number of examples per regex pattern
        "max_patterns": 50,              # Max regex patterns to process (None for all)
        "output_dir": "./generated_data"
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
        max_examples_per_regex=config["max_examples_per_regex"],
        output_dir=config["output_dir"]
    )
    
    # Generate dataset
    examples = generator.generate_dataset(max_patterns=config["max_patterns"])
    
    # Create training splits
    if examples:
        generator.create_training_splits(examples)
        
        # Print statistics
        difficulty_counts = {}
        for ex in examples:
            difficulty_counts[ex.difficulty_level] = difficulty_counts.get(ex.difficulty_level, 0) + 1
            
        logger.info("Dataset statistics:")
        for difficulty, count in difficulty_counts.items():
            logger.info(f"  {difficulty}: {count} examples")
            
        avg_budget = np.mean([ex.character_budget for ex in examples])
        logger.info(f"  Average character budget: {avg_budget:.1f}")
        
    else:
        logger.error("No examples generated! Check your OpenAI API key and quota.")

if __name__ == "__main__":
    main()
