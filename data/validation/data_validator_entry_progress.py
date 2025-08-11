"""
Data validator with entry-level progress tracking using tqdm.
Optimized for small number of files with many entries each.
"""
import json
import re
import os
import glob
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import time

from data_validator import ValidationStats, DataValidator


class EntryProgressValidator(DataValidator):
    """Validator with entry-level progress tracking."""
    
    def __init__(self, input_dir: str = "../generated_data", 
                 output_dir: str = "../validated_data"):
        """
        Initialize validator with entry-level progress.
        
        Args:
            input_dir: Directory containing JSON files to validate
            output_dir: Directory to save validated files
        """
        super().__init__(input_dir, output_dir)
    
    def count_total_entries(self, json_files: List[str]) -> Tuple[int, Dict[str, int]]:
        """
        Count total entries without loading all data into memory.
        
        Returns:
            Tuple of (total_entries, dict of filepath -> entry_count)
        """
        print("\n📊 Analyzing dataset structure...")
        total = 0
        file_counts = {}
        
        for filepath in json_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    count = 1
                else:
                    count = len(data)
                    
                file_counts[filepath] = count
                total += count
                
                filename = os.path.basename(filepath)
                print(f"  • {filename}: {count:,} entries")
                
            except Exception as e:
                print(f"  ✗ Error reading {os.path.basename(filepath)}: {e}")
                file_counts[filepath] = 0
        
        return total, file_counts
    
    def process_all_files_with_entry_progress(self) -> Dict[str, Dict[str, Any]]:
        """
        Process all JSON files with entry-level progress tracking.
        
        Returns:
            Dictionary with statistics for each file
        """
        # Find all JSON files recursively
        pattern = os.path.join(self.input_dir, "**", "*.json")
        json_files = glob.glob(pattern, recursive=True)
        
        # Filter out checkpoint files if they exist
        main_files = [f for f in json_files if 'temp_results' not in f]
        temp_files = [f for f in json_files if 'temp_results' in f]
        
        if main_files and temp_files:
            print(f"\n⚠️  Found {len(temp_files)} checkpoint files in temp_results/")
            response = input("Include checkpoint files? They may be duplicates. (y/N): ").strip().lower()
            if response != 'y':
                json_files = main_files
        
        if not json_files:
            print(f"No JSON files found in {self.input_dir}")
            return {}
        
        # Count entries without preloading data
        total_entries, file_counts = self.count_total_entries(json_files)
        
        print(f"\n{'='*70}")
        print(f"📁 Files to process: {len(json_files)}")
        print(f"📝 Total entries: {total_entries:,}")
        print(f"{'='*70}\n")
        
        all_stats = {}
        all_errors = []
        
        # Create a single progress bar for all entries
        with tqdm(total=total_entries, 
                 desc="Validating entries", 
                 unit=" entries",
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                 colour='green') as pbar:
            
            # Process each file (loading one at a time to save memory)
            for filepath in json_files:
                filename = os.path.basename(filepath)
                entry_count = file_counts[filepath]
                
                if entry_count == 0:
                    continue
                
                # Load this file's data
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                except Exception as e:
                    print(f"\n✗ Error loading {filename}: {e}")
                    pbar.update(entry_count)
                    continue
                
                # Update progress bar with current file
                pbar.set_postfix_str(f"Current: {filename[:30]}...")
                
                # Process entries in this file
                validated_entries = []
                stats_list = []
                file_errors = []
                
                for i, entry in enumerate(data):
                    if not isinstance(entry, dict):
                        pbar.update(1)
                        continue
                    
                    # Validate entry
                    validated_entry, stats = self.validate_entry(entry)
                    validated_entries.append(validated_entry)
                    stats_list.append(stats)
                    
                    # Track errors
                    if stats.removed_yes_count > 0 or stats.removed_no_count > 0:
                        file_errors.append({
                            'entry_idx': i,
                            'regex': entry.get('expr', '')[:50],
                            'removed_yes': stats.removed_yes_count,
                            'removed_no': stats.removed_no_count
                        })
                    
                    # Update progress
                    pbar.update(1)
                    
                    # Update description periodically with stats
                    if (i + 1) % 100 == 0 or (i + 1) == len(data):
                        # Calculate running accuracy for this file
                        total_yes = sum(s.original_yes_count for s in stats_list)
                        total_no = sum(s.original_no_count for s in stats_list)
                        kept_yes = sum(s.filtered_yes_count for s in stats_list)
                        kept_no = sum(s.filtered_no_count for s in stats_list)
                        
                        if total_yes > 0 and total_no > 0:
                            yes_acc = (kept_yes / total_yes) * 100
                            no_acc = (kept_no / total_no) * 100
                            pbar.set_postfix_str(
                                f"{filename[:25]}... [{i+1}/{len(data)}] YES:{yes_acc:.0f}% NO:{no_acc:.0f}%"
                            )
                
                # Save validated data
                if validated_entries:
                    output_path = os.path.join(self.output_dir, filename)
                    with open(output_path, 'w') as f:
                        json.dump(validated_entries, f, indent=2)
                    
                    # Calculate file statistics
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
                    
                    # Print file summary if errors found
                    if file_errors:
                        tqdm.write(f"\n📄 {filename}: {len(file_errors)} entries with errors")
                        tqdm.write(f"   Removed {total_removed_yes} false positives, {total_removed_no} false negatives")
                        all_errors.extend([(filename, err) for err in file_errors])
        
        return all_stats
    
    def print_summary(self, all_stats: Dict[str, Dict[str, Any]]):
        """Print summary with better formatting."""
        if not all_stats:
            print("\n❌ No files were processed successfully.")
            return
        
        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70)
        
        # Overall statistics
        total_entries = sum(s['entries'] for s in all_stats.values())
        total_original = sum(s['total_original_yes'] + s['total_original_no'] for s in all_stats.values())
        total_kept = sum(s['total_filtered_yes'] + s['total_filtered_no'] for s in all_stats.values())
        total_removed = total_original - total_kept
        
        print(f"\n📊 Summary:")
        print(f"  Files processed: {len(all_stats)}")
        print(f"  Total entries: {total_entries:,}")
        print(f"  Total strings validated: {total_original:,}")
        print(f"  Strings kept: {total_kept:,} ({(total_kept/total_original)*100:.1f}%)")
        print(f"  Strings removed: {total_removed:,} ({(total_removed/total_original)*100:.1f}%)")
        
        # Per-file details
        print(f"\n📁 File Details:")
        for filename, stats in sorted(all_stats.items()):
            yes_acc = stats['yes_accuracy'] * 100
            no_acc = stats['no_accuracy'] * 100
            avg_acc = (yes_acc + no_acc) / 2
            
            print(f"  {filename[:40]:40} | "
                  f"Entries: {stats['entries']:4} | "
                  f"Accuracy: {avg_acc:5.1f}% "
                  f"(YES: {yes_acc:.1f}%, NO: {no_acc:.1f}%)")
        
        # Save stats
        stats_file = os.path.join(self.output_dir, '_validation_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n💾 Statistics saved to {stats_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate regex datasets with entry-level progress')
    parser.add_argument('--input', default='../generated_data', 
                        help='Input directory with JSON files')
    parser.add_argument('--output', default='../validated_data',
                        help='Output directory for validated files')
    
    args = parser.parse_args()
    
    print("🚀 Starting validation with entry-level progress tracking...")
    print("   Optimized for small number of files with many entries each")
    
    validator = EntryProgressValidator(args.input, args.output)
    
    start_time = time.perf_counter()
    all_stats = validator.process_all_files_with_entry_progress()
    elapsed = time.perf_counter() - start_time
    
    # Print performance metrics
    if all_stats:
        total_entries = sum(s['entries'] for s in all_stats.values())
        print(f"\n⏱️  Completed in {elapsed:.2f} seconds")
        print(f"   Speed: {total_entries/elapsed:.1f} entries/second")
    
    validator.print_summary(all_stats)


if __name__ == "__main__":
    main()