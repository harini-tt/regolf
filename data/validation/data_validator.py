"""
Data validator for regex golf datasets.
Filters YES/NO lists to ensure they correctly match/don't match the given regex.
"""
import json
import re
import os
import glob
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import traceback


@dataclass
class ValidationStats:
    """Statistics for a single entry validation."""
    original_yes_count: int
    original_no_count: int
    filtered_yes_count: int
    filtered_no_count: int
    removed_yes_count: int  # False positives (didn't actually match)
    removed_no_count: int   # False negatives (actually matched)
    
    @property
    def yes_accuracy(self) -> float:
        """Percentage of YES strings that were correct."""
        if self.original_yes_count == 0:
            return 1.0
        return self.filtered_yes_count / self.original_yes_count
    
    @property
    def no_accuracy(self) -> float:
        """Percentage of NO strings that were correct."""
        if self.original_no_count == 0:
            return 1.0
        return self.filtered_no_count / self.original_no_count
    
    @property
    def overall_accuracy(self) -> float:
        """Overall accuracy of the original data."""
        total = self.original_yes_count + self.original_no_count
        if total == 0:
            return 1.0
        correct = self.filtered_yes_count + self.filtered_no_count
        return correct / total


class DataValidator:
    """Validates and cleans regex golf datasets."""
    
    def __init__(self, input_dir: str = "../generated_data", output_dir: str = "../validated_data"):
        """
        Initialize validator.
        
        Args:
            input_dir: Directory containing JSON files to validate
            output_dir: Directory to save validated files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def validate_entry(self, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], ValidationStats]:
        """
        Validate a single entry by filtering YES/NO lists.
        
        Args:
            entry: Dictionary with 'expr', 'yes', and 'no' fields
            
        Returns:
            Tuple of (validated_entry, stats)
        """
        # Extract fields
        regex_str = entry.get('expr', '')
        yes_list = entry.get('yes', [])
        no_list = entry.get('no', [])
        
        # Track original counts
        original_yes = len(yes_list)
        original_no = len(no_list)
        
        # Try to compile regex
        try:
            pattern = re.compile(regex_str)
        except re.error as e:
            # If regex is invalid, return empty lists
            print(f"Invalid regex pattern: {regex_str}")
            print(f"Error: {e}")
            validated_entry = {
                'expr': regex_str,
                'yes': [],
                'no': [],
                '_validation_error': str(e)
            }
            stats = ValidationStats(
                original_yes_count=original_yes,
                original_no_count=original_no,
                filtered_yes_count=0,
                filtered_no_count=0,
                removed_yes_count=original_yes,
                removed_no_count=original_no
            )
            return validated_entry, stats
        
        # Filter YES list - keep only strings that match
        filtered_yes = []
        for s in yes_list:
            try:
                if pattern.fullmatch(s):
                    filtered_yes.append(s)
            except Exception as e:
                # Skip strings that cause errors
                print(f"Error matching YES string '{s}': {e}")
        
        # Filter NO list - keep only strings that DON'T match
        filtered_no = []
        for s in no_list:
            try:
                if not pattern.fullmatch(s):
                    filtered_no.append(s)
            except Exception as e:
                # Skip strings that cause errors
                print(f"Error matching NO string '{s}': {e}")
        
        # Create validated entry
        validated_entry = {
            'expr': regex_str,
            'yes': filtered_yes,
            'no': filtered_no
        }
        
        # Calculate statistics
        stats = ValidationStats(
            original_yes_count=original_yes,
            original_no_count=original_no,
            filtered_yes_count=len(filtered_yes),
            filtered_no_count=len(filtered_no),
            removed_yes_count=original_yes - len(filtered_yes),
            removed_no_count=original_no - len(filtered_no)
        )
        
        return validated_entry, stats
    
    def validate_file(self, filepath: str) -> Tuple[List[Dict[str, Any]], List[ValidationStats]]:
        """
        Validate all entries in a JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Tuple of (validated_entries, stats_list)
        """
        print(f"\nValidating {filepath}...")
        
        # Load data
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return [], []
        
        # Handle both list and single entry formats
        if not isinstance(data, list):
            data = [data]
        
        validated_entries = []
        stats_list = []
        
        # Process each entry
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                print(f"Skipping non-dict entry {i}")
                continue
            
            # Validate entry
            validated_entry, stats = self.validate_entry(entry)
            validated_entries.append(validated_entry)
            stats_list.append(stats)
            
            # Print progress for problematic entries
            if stats.removed_yes_count > 0 or stats.removed_no_count > 0:
                print(f"  Entry {i}: Removed {stats.removed_yes_count} YES (false positives), "
                      f"{stats.removed_no_count} NO (false negatives)")
        
        return validated_entries, stats_list
    
    def process_all_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Process all JSON files in the input directory (including subdirectories).
        
        Returns:
            Dictionary with statistics for each file
        """
        # Find all JSON files recursively
        pattern = os.path.join(self.input_dir, "**", "*.json")
        json_files = glob.glob(pattern, recursive=True)
        
        if not json_files:
            print(f"No JSON files found in {self.input_dir}")
            return {}
        
        print(f"Found {len(json_files)} JSON files to process")
        
        all_stats = {}
        
        # Process each file
        for filepath in json_files:
            filename = os.path.basename(filepath)
            
            # Validate file
            validated_entries, stats_list = self.validate_file(filepath)
            
            if not validated_entries:
                print(f"  No valid entries in {filename}")
                continue
            
            # Save validated data
            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, 'w') as f:
                json.dump(validated_entries, f, indent=2)
            
            # Calculate aggregate statistics
            total_original_yes = sum(s.original_yes_count for s in stats_list)
            total_original_no = sum(s.original_no_count for s in stats_list)
            total_filtered_yes = sum(s.filtered_yes_count for s in stats_list)
            total_filtered_no = sum(s.filtered_no_count for s in stats_list)
            total_removed_yes = sum(s.removed_yes_count for s in stats_list)
            total_removed_no = sum(s.removed_no_count for s in stats_list)
            
            file_stats = {
                'entries': len(validated_entries),
                'total_original_yes': total_original_yes,
                'total_original_no': total_original_no,
                'total_filtered_yes': total_filtered_yes,
                'total_filtered_no': total_filtered_no,
                'total_removed_yes': total_removed_yes,
                'total_removed_no': total_removed_no,
                'yes_accuracy': total_filtered_yes / total_original_yes if total_original_yes > 0 else 1.0,
                'no_accuracy': total_filtered_no / total_original_no if total_original_no > 0 else 1.0,
            }
            
            all_stats[filename] = file_stats
            
            print(f"  Saved to {output_path}")
            print(f"  Stats: {len(validated_entries)} entries, "
                  f"YES accuracy: {file_stats['yes_accuracy']:.1%}, "
                  f"NO accuracy: {file_stats['no_accuracy']:.1%}")
        
        return all_stats
    
    def print_summary(self, all_stats: Dict[str, Dict[str, Any]]):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        for filename, stats in all_stats.items():
            print(f"\n{filename}:")
            print(f"  Entries: {stats['entries']}")
            print(f"  YES strings: {stats['total_original_yes']} → {stats['total_filtered_yes']} "
                  f"(removed {stats['total_removed_yes']} false positives)")
            print(f"  NO strings: {stats['total_original_no']} → {stats['total_filtered_no']} "
                  f"(removed {stats['total_removed_no']} false negatives)")
            print(f"  YES accuracy: {stats['yes_accuracy']:.1%}")
            print(f"  NO accuracy: {stats['no_accuracy']:.1%}")
        
        # Overall statistics
        if all_stats:
            total_entries = sum(s['entries'] for s in all_stats.values())
            total_yes_removed = sum(s['total_removed_yes'] for s in all_stats.values())
            total_no_removed = sum(s['total_removed_no'] for s in all_stats.values())
            
            print(f"\nOVERALL:")
            print(f"  Files processed: {len(all_stats)}")
            print(f"  Total entries: {total_entries}")
            print(f"  Total false positives removed: {total_yes_removed}")
            print(f"  Total false negatives removed: {total_no_removed}")


def main():
    """Main entry point for data validation."""
    validator = DataValidator()
    
    print("Starting data validation...")
    print(f"Input directory: {validator.input_dir}")
    print(f"Output directory: {validator.output_dir}")
    
    # Process all files
    all_stats = validator.process_all_files()
    
    # Print summary
    validator.print_summary(all_stats)
    
    # Save statistics to file
    stats_file = os.path.join(validator.output_dir, "_validation_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\nValidation statistics saved to {stats_file}")
    print("\nValidation complete!")


if __name__ == "__main__":
    main()