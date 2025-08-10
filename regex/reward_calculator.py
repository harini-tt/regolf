"""
Calculate rewards for regex golf solutions based on correctness and brevity.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class EvaluationResult:
    """Results from evaluating a regex against YES/NO lists."""
    # Correctness metrics
    yes_matches: int  # Number of YES strings matched
    yes_total: int    # Total YES strings
    no_rejects: int   # Number of NO strings rejected (not matched)
    no_total: int     # Total NO strings
    
    # Derived metrics
    accuracy: float   # Overall accuracy (correct / total)
    yes_recall: float # Fraction of YES matched
    no_precision: float # Fraction of NO rejected
    
    # Pattern info
    pattern_length: int  # Length of canonical pattern
    is_valid: bool      # Whether pattern compiled successfully
    error_message: Optional[str] = None
    
    # Performance
    eval_time_ms: float = 0.0  # Time to evaluate all strings
    
    def is_perfect(self) -> bool:
        """Check if solution perfectly solves the problem."""
        return self.yes_matches == self.yes_total and self.no_rejects == self.no_total


class RewardCalculator:
    """Calculate rewards for GRPO training."""
    
    def __init__(
        self,
        length_penalty_weight: float = 0.01,
        correctness_mode: str = "binary",  # "binary" or "fractional"
        invalid_reward: float = -10.0,
        timeout_ms: float = 100.0  # Max time per pattern evaluation
    ):
        """
        Initialize reward calculator.
        
        Args:
            length_penalty_weight: Weight for pattern length penalty (smaller = better)
            correctness_mode: "binary" (all or nothing) or "fractional" (partial credit)
            invalid_reward: Reward for invalid/non-compiling patterns
            timeout_ms: Maximum time in ms for evaluating a pattern
        """
        self.length_penalty_weight = length_penalty_weight
        self.correctness_mode = correctness_mode
        self.invalid_reward = invalid_reward
        self.timeout_ms = timeout_ms
    
    def evaluate_pattern(
        self,
        pattern: re.Pattern,
        pattern_str: str,
        yes_strings: List[str],
        no_strings: List[str]
    ) -> EvaluationResult:
        """
        Evaluate a compiled regex pattern against YES/NO lists.
        
        Args:
            pattern: Compiled regex pattern
            pattern_str: String representation (for length calculation)
            yes_strings: List of strings that should match
            no_strings: List of strings that should not match
            
        Returns:
            EvaluationResult with metrics
        """
        start_time = time.perf_counter()
        
        # Count matches for YES strings
        yes_matches = 0
        for s in yes_strings:
            try:
                if pattern.fullmatch(s):
                    yes_matches += 1
            except Exception:
                # Handle potential regex errors
                pass
        
        # Count non-matches for NO strings (we want these to NOT match)
        no_rejects = 0
        for s in no_strings:
            try:
                if not pattern.fullmatch(s):
                    no_rejects += 1
            except Exception:
                # Handle potential regex errors  
                pass
        
        eval_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Calculate metrics
        yes_total = len(yes_strings)
        no_total = len(no_strings)
        total = yes_total + no_total
        
        correct = yes_matches + no_rejects
        accuracy = correct / total if total > 0 else 0.0
        yes_recall = yes_matches / yes_total if yes_total > 0 else 1.0
        no_precision = no_rejects / no_total if no_total > 0 else 1.0
        
        return EvaluationResult(
            yes_matches=yes_matches,
            yes_total=yes_total,
            no_rejects=no_rejects,
            no_total=no_total,
            accuracy=accuracy,
            yes_recall=yes_recall,
            no_precision=no_precision,
            pattern_length=len(pattern_str),
            is_valid=True,
            eval_time_ms=eval_time_ms
        )
    
    def calculate_reward(self, eval_result: EvaluationResult) -> float:
        """
        Calculate reward from evaluation result.
        
        Lexicographic ordering: correctness >> brevity
        
        Args:
            eval_result: Result from evaluate_pattern
            
        Returns:
            Reward value for GRPO
        """
        if not eval_result.is_valid:
            return self.invalid_reward
        
        # Calculate correctness reward
        if self.correctness_mode == "binary":
            # All or nothing
            if eval_result.is_perfect():
                correctness_reward = 1.0
            else:
                correctness_reward = -1.0
        else:
            # Fractional credit based on accuracy
            correctness_reward = 2.0 * eval_result.accuracy - 1.0  # Maps [0,1] -> [-1,1]
        
        # Add length penalty (only matters when correct)
        if correctness_reward > 0:
            # Subtract small penalty for length
            length_penalty = self.length_penalty_weight * eval_result.pattern_length
            reward = correctness_reward - length_penalty
        else:
            reward = correctness_reward
        
        return reward
    
    def evaluate_solution(
        self,
        pattern: Optional[re.Pattern],
        pattern_str: str,
        yes_strings: List[str],
        no_strings: List[str],
        error: Optional[str] = None
    ) -> Tuple[EvaluationResult, float]:
        """
        Main entry point for evaluating a solution.
        
        Args:
            pattern: Compiled pattern (None if compilation failed)
            pattern_str: String representation
            yes_strings: YES list
            no_strings: NO list
            error: Error message if compilation failed
            
        Returns:
            Tuple of (EvaluationResult, reward)
        """
        if pattern is None:
            # Invalid pattern
            result = EvaluationResult(
                yes_matches=0,
                yes_total=len(yes_strings),
                no_rejects=0,
                no_total=len(no_strings),
                accuracy=0.0,
                yes_recall=0.0,
                no_precision=0.0,
                pattern_length=len(pattern_str) if pattern_str else 0,
                is_valid=False,
                error_message=error
            )
            reward = self.invalid_reward
        else:
            # Evaluate valid pattern
            result = self.evaluate_pattern(pattern, pattern_str, yes_strings, no_strings)
            reward = self.calculate_reward(result)
        
        return result, reward
    
    def batch_evaluate(
        self,
        solutions: List[Tuple[Optional[re.Pattern], str, Optional[str]]],
        yes_strings: List[str],
        no_strings: List[str]
    ) -> List[Tuple[EvaluationResult, float]]:
        """
        Evaluate multiple solutions in batch.
        
        Args:
            solutions: List of (pattern, pattern_str, error) tuples
            yes_strings: YES list
            no_strings: NO list
            
        Returns:
            List of (EvaluationResult, reward) tuples
        """
        results = []
        for pattern, pattern_str, error in solutions:
            result, reward = self.evaluate_solution(
                pattern, pattern_str, yes_strings, no_strings, error
            )
            results.append((result, reward))
        return results


def compute_grpo_rewards(
    rewards: List[float],
    normalize: bool = True
) -> List[float]:
    """
    Process rewards for GRPO update.
    
    Args:
        rewards: Raw rewards from evaluation
        normalize: Whether to normalize rewards within batch
        
    Returns:
        Processed rewards for GRPO
    """
    import numpy as np
    
    rewards = np.array(rewards)
    
    if normalize and len(rewards) > 1:
        # Normalize to zero mean and unit variance
        mean = rewards.mean()
        std = rewards.std()
        if std > 0:
            rewards = (rewards - mean) / std
    
    return rewards.tolist()


if __name__ == "__main__":
    # Test example
    import re
    
    # Create calculator
    calc = RewardCalculator(length_penalty_weight=0.01)
    
    # Test pattern
    pattern = re.compile(r"^test$")
    yes_list = ["test"]
    no_list = ["test123", "other"]
    
    # Evaluate
    result, reward = calc.evaluate_solution(
        pattern, "^test$", yes_list, no_list
    )
    
    print(f"Evaluation Result:")
    print(f"  YES matched: {result.yes_matches}/{result.yes_total}")
    print(f"  NO rejected: {result.no_rejects}/{result.no_total}")
    print(f"  Accuracy: {result.accuracy:.2f}")
    print(f"  Pattern length: {result.pattern_length}")
    print(f"  Is perfect: {result.is_perfect()}")
    print(f"  Reward: {reward:.3f}")