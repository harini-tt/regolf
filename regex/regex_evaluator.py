"""
Main integration module for regex evaluation pipeline.
Combines parsing, validation, and reward calculation.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .regex_ast_validation import compile_from_model_json, ValidationError
from .model_output_parser import parse_model_output, HarmonyParseError
from .reward_calculator import RewardCalculator, EvaluationResult


@dataclass
class RegexSolution:
    """Complete solution with all metadata."""
    # Input
    model_output: str
    yes_strings: List[str]
    no_strings: List[str]
    
    # Parsed solution
    solution_json: Optional[Dict[str, Any]] = None
    
    # Compiled regex
    pattern: Optional[re.Pattern] = None
    pattern_str: Optional[str] = None
    flags: Optional[int] = None
    
    # Evaluation
    eval_result: Optional[EvaluationResult] = None
    reward: Optional[float] = None
    
    # Errors
    parse_error: Optional[str] = None
    validation_error: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if solution is valid."""
        return self.pattern is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "is_valid": self.is_valid(),
            "pattern": self.pattern_str,
            "flags": self.flags,
            "reward": self.reward,
            "accuracy": self.eval_result.accuracy if self.eval_result else 0.0,
            "pattern_length": self.eval_result.pattern_length if self.eval_result else 0,
            "is_perfect": self.eval_result.is_perfect() if self.eval_result else False,
            "parse_error": self.parse_error,
            "validation_error": self.validation_error,
        }


class RegexEvaluator:
    """Main evaluator for regex golf solutions."""
    
    def __init__(
        self,
        reward_calculator: Optional[RewardCalculator] = None,
        strict_parsing: bool = False,
        verbose: bool = False
    ):
        """
        Initialize evaluator.
        
        Args:
            reward_calculator: Custom reward calculator (uses default if None)
            strict_parsing: Require strict Harmony format
            verbose: Print debug information
        """
        self.reward_calc = reward_calculator or RewardCalculator()
        self.strict_parsing = strict_parsing
        self.verbose = verbose
    
    def evaluate_single(
        self,
        model_output: str,
        yes_strings: List[str],
        no_strings: List[str]
    ) -> RegexSolution:
        """
        Evaluate a single model output.
        
        Args:
            model_output: Raw model output (Harmony format or JSON)
            yes_strings: List of strings that should match
            no_strings: List of strings that should not match
            
        Returns:
            RegexSolution with all results
        """
        solution = RegexSolution(
            model_output=model_output,
            yes_strings=yes_strings,
            no_strings=no_strings
        )
        
        # Step 1: Parse model output
        try:
            solution.solution_json = parse_model_output(
                model_output, 
                strict=self.strict_parsing
            )
            if self.verbose:
                print(f"Parsed JSON: {json.dumps(solution.solution_json, indent=2)}")
        except HarmonyParseError as e:
            solution.parse_error = str(e)
            if self.verbose:
                print(f"Parse error: {e}")
            # Can't continue without parsed JSON
            solution.eval_result, solution.reward = self.reward_calc.evaluate_solution(
                None, "", yes_strings, no_strings, str(e)
            )
            return solution
        
        # Step 2: Validate and compile
        try:
            pattern_str, flags_val, compiled = compile_from_model_json(
                solution.solution_json
            )
            solution.pattern = compiled
            solution.pattern_str = pattern_str
            solution.flags = flags_val
            if self.verbose:
                print(f"Compiled pattern: {pattern_str} (flags: {flags_val})")
        except ValidationError as e:
            solution.validation_error = str(e)
            if self.verbose:
                print(f"Validation error: {e}")
            # Pattern is invalid
            solution.eval_result, solution.reward = self.reward_calc.evaluate_solution(
                None, "", yes_strings, no_strings, str(e)
            )
            return solution
        except Exception as e:
            solution.validation_error = f"Unexpected error: {e}"
            if self.verbose:
                print(f"Unexpected validation error: {e}")
                traceback.print_exc()
            solution.eval_result, solution.reward = self.reward_calc.evaluate_solution(
                None, "", yes_strings, no_strings, str(e)
            )
            return solution
        
        # Step 3: Evaluate against YES/NO lists
        solution.eval_result, solution.reward = self.reward_calc.evaluate_solution(
            solution.pattern,
            solution.pattern_str,
            yes_strings,
            no_strings
        )
        
        if self.verbose:
            print(f"Evaluation complete:")
            print(f"  Accuracy: {solution.eval_result.accuracy:.2f}")
            print(f"  Pattern length: {solution.eval_result.pattern_length}")
            print(f"  Reward: {solution.reward:.3f}")
        
        return solution
    
    def evaluate_batch(
        self,
        model_outputs: List[str],
        yes_strings: List[str],
        no_strings: List[str],
        max_workers: int = 4
    ) -> List[RegexSolution]:
        """
        Evaluate multiple model outputs in parallel.
        
        Args:
            model_outputs: List of model outputs
            yes_strings: YES list (same for all)
            no_strings: NO list (same for all)
            max_workers: Number of parallel workers
            
        Returns:
            List of RegexSolution objects
        """
        solutions = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self.evaluate_single, 
                    output, 
                    yes_strings, 
                    no_strings
                ): i
                for i, output in enumerate(model_outputs)
            }
            
            # Collect results in order
            results = [None] * len(model_outputs)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Create error solution
                    results[idx] = RegexSolution(
                        model_output=model_outputs[idx],
                        yes_strings=yes_strings,
                        no_strings=no_strings,
                        parse_error=f"Evaluation failed: {e}"
                    )
                    results[idx].eval_result, results[idx].reward = \
                        self.reward_calc.evaluate_solution(
                            None, "", yes_strings, no_strings, str(e)
                        )
        
        return [r for r in results if r is not None]
    
    def evaluate_problem(
        self,
        model_outputs: List[str],
        yes_strings: List[str],
        no_strings: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple candidate solutions for a single problem.
        
        Args:
            model_outputs: K candidate outputs from model
            yes_strings: YES list
            no_strings: NO list
            
        Returns:
            Dictionary with evaluation statistics
        """
        solutions = self.evaluate_batch(model_outputs, yes_strings, no_strings)
        
        # Compute statistics
        valid_solutions = [s for s in solutions if s.is_valid()]
        perfect_solutions = [s for s in valid_solutions if s.eval_result.is_perfect()]
        
        stats = {
            "num_candidates": len(solutions),
            "num_valid": len(valid_solutions),
            "num_perfect": len(perfect_solutions),
            "rewards": [s.reward for s in solutions],
            "best_reward": max([s.reward for s in solutions]) if solutions else None,
        }
        
        if perfect_solutions:
            # Find shortest perfect solution
            best = min(perfect_solutions, key=lambda s: s.eval_result.pattern_length)
            stats["best_solution"] = {
                "pattern": best.pattern_str,
                "length": best.eval_result.pattern_length,
                "reward": best.reward
            }
        elif valid_solutions:
            # Find best valid solution by reward
            best = max(valid_solutions, key=lambda s: s.reward)
            stats["best_solution"] = {
                "pattern": best.pattern_str,
                "accuracy": best.eval_result.accuracy,
                "length": best.eval_result.pattern_length,
                "reward": best.reward
            }
        
        return stats


def create_evaluator(
    length_penalty: float = 0.01,
    correctness_mode: str = "binary",
    invalid_reward: float = -10.0,
    strict_parsing: bool = False,
    verbose: bool = False
) -> RegexEvaluator:
    """
    Factory function to create evaluator with custom settings.
    
    Args:
        length_penalty: Weight for pattern length penalty
        correctness_mode: "binary" or "fractional"
        invalid_reward: Reward for invalid patterns
        strict_parsing: Require strict Harmony format
        verbose: Print debug information
        
    Returns:
        Configured RegexEvaluator
    """
    reward_calc = RewardCalculator(
        length_penalty_weight=length_penalty,
        correctness_mode=correctness_mode,
        invalid_reward=invalid_reward
    )
    
    return RegexEvaluator(
        reward_calculator=reward_calc,
        strict_parsing=strict_parsing,
        verbose=verbose
    )


if __name__ == "__main__":
    # Test the full pipeline
    evaluator = create_evaluator(verbose=True)
    
    # Example model output (Harmony format)
    model_output = """
    final: {
        "flags": "",
        "unsat": false,
        "ast": {
            "type": "seq",
            "items": [
                {"type": "anchor", "kind": "start"},
                {"type": "lit", "value": "test"},
                {"type": "anchor", "kind": "end"}
            ]
        }
    }
    """
    
    yes_list = ["test"]
    no_list = ["test123", "other", "testing"]
    
    # Evaluate single output
    solution = evaluator.evaluate_single(model_output, yes_list, no_list)
    
    print("\nFinal solution:")
    print(json.dumps(solution.to_dict(), indent=2))
    
    # Test batch evaluation
    print("\n" + "="*50)
    print("Testing batch evaluation...")
    
    outputs = [model_output] * 3  # Same output 3 times for testing
    solutions = evaluator.evaluate_batch(outputs, yes_list, no_list)
    
    print(f"Evaluated {len(solutions)} solutions")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: reward={sol.reward:.3f}, valid={sol.is_valid()}")