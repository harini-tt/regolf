"""
Test script for the regex evaluation pipeline.
"""
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from regex import create_evaluator, RegexEvaluator


def test_valid_solution():
    """Test a valid regex solution."""
    print("Testing valid solution...")
    
    evaluator = create_evaluator(verbose=True)
    
    # Model output that matches "cat" or "dog"
    model_output = """{
        "flags": "i",
        "unsat": false,
        "ast": {
            "type": "alt",
            "options": [
                {"type": "lit", "value": "cat"},
                {"type": "lit", "value": "dog"}
            ]
        }
    }"""
    
    yes_strings = ["cat", "dog", "CAT", "DOG"]  # Case insensitive with flag i
    no_strings = ["bird", "cats", "doggy", ""]
    
    solution = evaluator.evaluate_single(model_output, yes_strings, no_strings)
    
    print(f"\nResults:")
    print(f"  Pattern: {solution.pattern_str}")
    print(f"  Valid: {solution.is_valid()}")
    print(f"  Accuracy: {solution.eval_result.accuracy:.2%}")
    print(f"  Perfect: {solution.eval_result.is_perfect()}")
    print(f"  Reward: {solution.reward:.3f}")
    print()


def test_invalid_solution():
    """Test an invalid solution."""
    print("Testing invalid solution...")
    
    evaluator = create_evaluator(verbose=False)
    
    # Invalid AST (missing required field)
    model_output = """{
        "flags": "",
        "unsat": false,
        "ast": {
            "type": "invalid_type"
        }
    }"""
    
    yes_strings = ["test"]
    no_strings = ["other"]
    
    solution = evaluator.evaluate_single(model_output, yes_strings, no_strings)
    
    print(f"\nResults:")
    print(f"  Valid: {solution.is_valid()}")
    print(f"  Parse error: {solution.parse_error}")
    print(f"  Validation error: {solution.validation_error}")
    print(f"  Reward: {solution.reward:.3f}")
    print()


def test_anchored_pattern():
    """Test a pattern with anchors."""
    print("Testing anchored pattern...")
    
    evaluator = create_evaluator(verbose=False)
    
    # Pattern: ^test$
    model_output = """{
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
    }"""
    
    yes_strings = ["test"]
    no_strings = ["test123", "0test", "testing", ""]
    
    solution = evaluator.evaluate_single(model_output, yes_strings, no_strings)
    
    print(f"\nResults:")
    print(f"  Pattern: {solution.pattern_str}")
    print(f"  YES matched: {solution.eval_result.yes_matches}/{solution.eval_result.yes_total}")
    print(f"  NO rejected: {solution.eval_result.no_rejects}/{solution.eval_result.no_total}")
    print(f"  Perfect: {solution.eval_result.is_perfect()}")
    print(f"  Reward: {solution.reward:.3f}")
    print()


def test_character_class():
    """Test a pattern with character classes."""
    print("Testing character class pattern...")
    
    evaluator = create_evaluator(verbose=False)
    
    # Pattern: \d{3} (three digits)
    model_output = """{
        "flags": "",
        "unsat": false,
        "ast": {
            "type": "repeat",
            "child": {"type": "bclass", "value": "d"},
            "min": 3,
            "max": 3,
            "possessive": false
        }
    }"""
    
    yes_strings = ["123", "456", "789", "000"]
    no_strings = ["12", "1234", "abc", "1a2", ""]
    
    solution = evaluator.evaluate_single(model_output, yes_strings, no_strings)
    
    print(f"\nResults:")
    print(f"  Pattern: {solution.pattern_str}")
    print(f"  Accuracy: {solution.eval_result.accuracy:.2%}")
    print(f"  Pattern length: {solution.eval_result.pattern_length}")
    print(f"  Reward: {solution.reward:.3f}")
    print()


def test_batch_evaluation():
    """Test batch evaluation of multiple candidates."""
    print("Testing batch evaluation...")
    
    evaluator = create_evaluator(
        correctness_mode="fractional",
        verbose=False
    )
    
    # Multiple candidate solutions for same problem
    candidates = [
        # Perfect but longer
        """{
            "flags": "",
            "unsat": false,
            "ast": {
                "type": "seq",
                "items": [
                    {"type": "anchor", "kind": "start"},
                    {"type": "alt", "options": [
                        {"type": "lit", "value": "yes"},
                        {"type": "lit", "value": "yep"}
                    ]},
                    {"type": "anchor", "kind": "end"}
                ]
            }
        }""",
        
        # Shorter but imperfect
        """{
            "flags": "",
            "unsat": false,
            "ast": {
                "type": "lit", "value": "ye"}
            }
        }""",
        
        # Invalid
        """{
            "flags": "",
            "unsat": false,
            "ast": {"type": "invalid"}
        }"""
    ]
    
    yes_strings = ["yes", "yep"]
    no_strings = ["no", "nope", "yeah", "ye", ""]
    
    stats = evaluator.evaluate_problem(candidates, yes_strings, no_strings)
    
    print(f"\nBatch Results:")
    print(f"  Candidates: {stats['num_candidates']}")
    print(f"  Valid: {stats['num_valid']}")
    print(f"  Perfect: {stats['num_perfect']}")
    print(f"  Rewards: {[f'{r:.3f}' for r in stats['rewards']]}")
    if 'best_solution' in stats:
        print(f"  Best pattern: {stats['best_solution']['pattern']}")
        print(f"  Best reward: {stats['best_solution']['reward']:.3f}")
    print()


def test_harmony_format():
    """Test parsing Harmony format output."""
    print("Testing Harmony format parsing...")
    
    evaluator = create_evaluator(verbose=False)
    
    # Full Harmony format
    model_output = """
    <|start|>assistant<|message|>
    analysis: Looking at the YES list ["a", "b", "c"] and NO list ["d", "e"]...
    commentary: Single character alternation works best here.
    final: {
        "flags": "",
        "unsat": false,
        "ast": {
            "type": "charclass",
            "negated": false,
            "items": [
                {"type": "char", "value": "a"},
                {"type": "char", "value": "b"},
                {"type": "char", "value": "c"}
            ]
        }
    }
    <|end|>
    """
    
    yes_strings = ["a", "b", "c"]
    no_strings = ["d", "e", "f", ""]
    
    solution = evaluator.evaluate_single(model_output, yes_strings, no_strings)
    
    print(f"\nResults:")
    print(f"  Pattern: {solution.pattern_str}")
    print(f"  Valid: {solution.is_valid()}")
    print(f"  Perfect: {solution.eval_result.is_perfect()}")
    print(f"  Reward: {solution.reward:.3f}")
    print()


if __name__ == "__main__":
    print("="*60)
    print("REGEX EVALUATION PIPELINE TEST")
    print("="*60)
    
    test_valid_solution()
    test_invalid_solution()
    test_anchored_pattern()
    test_character_class()
    test_batch_evaluation()
    test_harmony_format()
    
    print("="*60)
    print("All tests completed!")
    print("="*60)