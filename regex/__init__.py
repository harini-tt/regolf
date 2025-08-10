"""
Regex evaluation and reward calculation for regex golf.
"""

from .regex_ast_validation import (
    compile_from_model_json,
    ValidationError,
    Node
)

from .model_output_parser import (
    parse_model_output,
    parse_harmony_output,
    HarmonyParseError
)

from .reward_calculator import (
    RewardCalculator,
    EvaluationResult,
    compute_grpo_rewards
)

from .regex_evaluator import (
    RegexEvaluator,
    RegexSolution,
    create_evaluator
)

__all__ = [
    # Main interfaces
    'RegexEvaluator',
    'RegexSolution',
    'create_evaluator',
    
    # Reward calculation
    'RewardCalculator',
    'EvaluationResult',
    'compute_grpo_rewards',
    
    # Parsing and validation
    'compile_from_model_json',
    'parse_model_output',
    'ValidationError',
    'HarmonyParseError',
]