import re

def validate_regex(expr, yes, no):
    """
    Validates a regex expression against positive and negative test cases.
    
    Args:
        expr (str): Regular expression pattern
        yes (list): List of strings that should match the regex
        no (list): List of strings that should NOT match the regex
    
    Returns:
        bool: True if regex matches all 'yes' strings and none of 'no' strings
    """
    try:
        pattern = re.compile(expr)
    except re.error:
        return False
    
    # Check that all 'yes' strings match
    for string in yes:
        if not pattern.search(string):
            return False
    
    # Check that no 'no' strings match
    for string in no:
        if pattern.search(string):
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    expr = r"\d+"
    yes = ["123", "abc456def", "7"]
    no = ["abc", "xyz", ""]
    
    result = validate_regex(expr, yes, no)
    print(f"Regex '{expr}' is valid: {result}")