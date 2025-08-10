#!/usr/bin/env python
"""
Convenience script to run data validation from project root.
"""
import sys
import os

# Add validation module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data', 'validation'))

from data_validator import DataValidator

def main():
    """Run data validation with correct paths."""
    validator = DataValidator(
        input_dir="data/generated_data",
        output_dir="data/validated_data"
    )
    
    print("Starting data validation...")
    print(f"Input directory: {validator.input_dir}")
    print(f"Output directory: {validator.output_dir}")
    
    # Process all files
    all_stats = validator.process_all_files()
    
    # Print summary
    validator.print_summary(all_stats)
    
    # Save statistics to file
    import json
    stats_file = os.path.join(validator.output_dir, "_validation_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\nValidation statistics saved to {stats_file}")
    print("\nValidation complete!")

if __name__ == "__main__":
    main()