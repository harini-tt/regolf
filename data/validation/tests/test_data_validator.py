"""
Test the data validator to ensure it correctly filters YES/NO lists.
"""
import json
import tempfile
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_validator import DataValidator, ValidationStats


def test_basic_validation():
    """Test basic validation functionality."""
    
    # Create test data
    test_data = [
        {
            "expr": r"^test\d+$",  # Matches "test" followed by one or more digits
            "yes": [
                "test123",      # Valid
                "test1",        # Valid
                "test99999",    # Valid
                "test",         # Invalid - no digits
                "test123abc",   # Invalid - extra characters
                "123test",      # Invalid - wrong order
            ],
            "no": [
                "abc",          # Valid NO
                "test",         # Valid NO
                "123",          # Valid NO
                "test123",      # Invalid - actually matches
                "test1",        # Invalid - actually matches
                "other",        # Valid NO
            ]
        },
        {
            "expr": r"[a-z]+",  # One or more lowercase letters
            "yes": [
                "abc",          # Valid
                "xyz",          # Valid
                "ABC",          # Invalid - uppercase
                "123",          # Invalid - digits
                "ab12",         # Invalid - contains digits
            ],
            "no": [
                "123",          # Valid NO
                "ABC",          # Valid NO
                "!@#",          # Valid NO
                "abc",          # Invalid - actually matches
                "xyz",          # Invalid - actually matches
            ]
        }
    ]
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        Path(input_dir).mkdir()
        
        # Save test data
        test_file = os.path.join(input_dir, "test_data.json")
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Run validator
        validator = DataValidator(input_dir=input_dir, output_dir=output_dir)
        all_stats = validator.process_all_files()
        
        # Load validated data
        validated_file = os.path.join(output_dir, "test_data.json")
        with open(validated_file, 'r') as f:
            validated_data = json.load(f)
        
        # Check first entry
        entry1 = validated_data[0]
        assert entry1["expr"] == r"^test\d+$"
        assert set(entry1["yes"]) == {"test123", "test1", "test99999"}
        assert set(entry1["no"]) == {"abc", "test", "123", "other"}
        
        # Check second entry
        entry2 = validated_data[1]
        assert entry2["expr"] == r"[a-z]+"
        assert set(entry2["yes"]) == {"abc", "xyz"}
        assert set(entry2["no"]) == {"123", "ABC", "!@#"}
        
        # Check statistics
        stats = all_stats["test_data.json"]
        assert stats["entries"] == 2
        assert stats["total_removed_yes"] == 3 + 3  # 3 from each entry
        assert stats["total_removed_no"] == 2 + 2   # 2 from each entry
        
        print("✓ Basic validation test passed")


def test_invalid_regex():
    """Test handling of invalid regex patterns."""
    
    test_data = [
        {
            "expr": r"[invalid(regex",  # Invalid - unclosed bracket
            "yes": ["test"],
            "no": ["other"]
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        Path(input_dir).mkdir()
        
        # Save test data
        test_file = os.path.join(input_dir, "test_invalid.json")
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Run validator
        validator = DataValidator(input_dir=input_dir, output_dir=output_dir)
        all_stats = validator.process_all_files()
        
        # Load validated data
        validated_file = os.path.join(output_dir, "test_invalid.json")
        with open(validated_file, 'r') as f:
            validated_data = json.load(f)
        
        # Check that invalid regex results in empty lists
        entry = validated_data[0]
        assert entry["yes"] == []
        assert entry["no"] == []
        assert "_validation_error" in entry
        
        print("✓ Invalid regex test passed")


def test_empty_lists():
    """Test handling of empty YES/NO lists."""
    
    test_data = [
        {
            "expr": r"test",
            "yes": [],
            "no": []
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        Path(input_dir).mkdir()
        
        # Save test data
        test_file = os.path.join(input_dir, "test_empty.json")
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Run validator
        validator = DataValidator(input_dir=input_dir, output_dir=output_dir)
        all_stats = validator.process_all_files()
        
        # Load validated data
        validated_file = os.path.join(output_dir, "test_empty.json")
        with open(validated_file, 'r') as f:
            validated_data = json.load(f)
        
        # Check that empty lists remain empty
        entry = validated_data[0]
        assert entry["yes"] == []
        assert entry["no"] == []
        
        print("✓ Empty lists test passed")


def test_validation_stats():
    """Test validation statistics calculation."""
    
    stats = ValidationStats(
        original_yes_count=10,
        original_no_count=20,
        filtered_yes_count=8,
        filtered_no_count=18,
        removed_yes_count=2,
        removed_no_count=2
    )
    
    assert stats.yes_accuracy == 0.8
    assert stats.no_accuracy == 0.9
    assert abs(stats.overall_accuracy - 26/30) < 0.001
    
    # Test with zero counts
    stats_zero = ValidationStats(
        original_yes_count=0,
        original_no_count=0,
        filtered_yes_count=0,
        filtered_no_count=0,
        removed_yes_count=0,
        removed_no_count=0
    )
    
    assert stats_zero.yes_accuracy == 1.0
    assert stats_zero.no_accuracy == 1.0
    assert stats_zero.overall_accuracy == 1.0
    
    print("✓ Validation stats test passed")


if __name__ == "__main__":
    print("Running data validator tests...")
    test_basic_validation()
    test_invalid_regex()
    test_empty_lists()
    test_validation_stats()
    print("\nAll tests passed! ✅")