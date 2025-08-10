"""
Parse model outputs in Harmony format to extract regex solutions.
"""
import json
import re
from typing import Dict, Any, Optional, Tuple


class HarmonyParseError(Exception):
    """Error parsing Harmony format output."""
    pass


def parse_harmony_output(model_output: str) -> Dict[str, Any]:
    """
    Parse Harmony-formatted model output to extract the JSON solution.
    
    The model output should contain channels (analysis, commentary, final).
    We extract the JSON from the final channel.
    
    Args:
        model_output: Raw model output in Harmony format
        
    Returns:
        Dictionary containing the regex solution (flags, unsat, ast)
        
    Raises:
        HarmonyParseError: If output format is invalid
    """
    # Look for content in the final channel
    # Pattern: <|start|>assistant<|message|>...final channel content...<|end|>
    # Or just look for JSON object in the output
    
    # First try to find a clean JSON object
    # Look for content between curly braces
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    # Find all potential JSON objects
    matches = re.findall(json_pattern, model_output, re.DOTALL)
    
    if not matches:
        # Try to find content after "final" channel marker
        final_pattern = r'final[^{]*(\{.*?\}(?:\s*\})*)'
        final_match = re.search(final_pattern, model_output, re.DOTALL | re.IGNORECASE)
        if final_match:
            potential_json = final_match.group(1)
        else:
            raise HarmonyParseError("No JSON object found in model output")
    else:
        # Take the last JSON object (likely the final output)
        potential_json = matches[-1]
    
    # Clean up the JSON string
    potential_json = potential_json.strip()
    
    # Try to parse the JSON
    try:
        solution = json.loads(potential_json)
    except json.JSONDecodeError as e:
        # Try to fix common issues
        # Remove trailing commas
        cleaned = re.sub(r',\s*}', '}', potential_json)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        try:
            solution = json.loads(cleaned)
        except json.JSONDecodeError:
            raise HarmonyParseError(f"Invalid JSON in model output: {e}")
    
    # Validate required fields
    if not isinstance(solution, dict):
        raise HarmonyParseError("Solution must be a JSON object")
    
    if "flags" not in solution:
        solution["flags"] = ""  # Default to no flags
    
    if "unsat" not in solution:
        solution["unsat"] = False  # Default to satisfiable
    
    # If unsat is true, ast is optional
    if not solution["unsat"] and "ast" not in solution:
        raise HarmonyParseError("Solution must contain 'ast' field when satisfiable")
    
    return solution


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Fallback method to extract JSON from arbitrary text.
    Useful when model doesn't follow exact Harmony format.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Extracted JSON object or None if not found
    """
    # Try to find JSON-like content
    # Look for the regex_solution pattern specifically
    
    # Method 1: Find content that looks like our expected schema
    ast_pattern = r'"ast"\s*:\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}'
    flags_pattern = r'"flags"\s*:\s*"[^"]*"'
    unsat_pattern = r'"unsat"\s*:\s*(?:true|false)'
    
    # Try to reconstruct if we find the parts
    has_ast = re.search(ast_pattern, text)
    has_flags = re.search(flags_pattern, text) 
    has_unsat = re.search(unsat_pattern, text)
    
    if has_flags or has_unsat or has_ast:
        # Try to extract the full object
        start_idx = text.find('{')
        if start_idx == -1:
            return None
            
        # Find matching closing brace
        depth = 0
        end_idx = start_idx
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > start_idx:
            potential_json = text[start_idx:end_idx]
            try:
                return json.loads(potential_json)
            except:
                pass
    
    return None


def parse_model_output(output: str, strict: bool = False) -> Dict[str, Any]:
    """
    Main entry point for parsing model output.
    
    Args:
        output: Raw model output
        strict: If True, require proper Harmony format. If False, try fallback extraction.
        
    Returns:
        Parsed solution dictionary
        
    Raises:
        HarmonyParseError: If parsing fails
    """
    try:
        return parse_harmony_output(output)
    except HarmonyParseError:
        if strict:
            raise
        
        # Try fallback extraction
        solution = extract_json_from_text(output)
        if solution is None:
            raise HarmonyParseError("Could not extract valid solution from model output")
        
        return solution


if __name__ == "__main__":
    # Test with example output
    test_output = """
    <|start|>assistant<|message|>
    analysis: Looking at the YES and NO lists...
    commentary: The pattern needs to match...
    final: {
        "flags": "i",
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
    <|end|>
    """
    
    result = parse_model_output(test_output)
    print("Parsed solution:", json.dumps(result, indent=2))