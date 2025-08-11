"""
Parallel data validator for regex golf datasets with improved progress tracking.
Shows progress by total entries processed, not just files.
"""
import json
import re
import os
import glob
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
import multiprocessing
from tqdm import tqdm
import time
from queue import Queue
import threading

from data_validator import ValidationStats, DataValidator


class ParallelDataValidator(DataValidator):
    """Parallel version of DataValidator with better progress tracking."""
    
    def __init__(self, input_dir: str = "../generated_data", 
                 output_dir: str = "../validated_data",
                 max_workers: int = None):
        """
        Initialize parallel validator.
        
        Args:
            input_dir: Directory containing JSON files to validate
            output_dir: Directory to save validated files
            max_workers: Maximum number of parallel workers (None = number of CPUs)
        """
        super().__init__(input_dir, output_dir)
        self.max_workers = max_workers or multiprocessing.cpu_count()
    
    def count_total_entries(self, json_files: List[str]) -> Tuple[int, Dict[str, int]]:
        """
        Count total entries across all files for progress tracking.
        
        Returns:
            Tuple of (total_entries, file_entry_counts)
        """
        print("Counting total entries for progress tracking...")
        total = 0
        file_counts = {}
        
        for filepath in tqdm(json_files, desc="Scanning files"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    count = len(data)
                else:
                    count = 1
                file_counts[filepath] = count
                total += count
            except Exception as e:
                print(f"Error reading {os.path.basename(filepath)}: {e}")
                file_counts[filepath] = 0
        
        return total, file_counts
    
    def process_all_files_parallel(self) -> Dict[str, Dict[str, Any]]:
        """
        Process all JSON files in parallel with entry-level progress tracking.
        
        Returns:
            Dictionary with statistics for each file
        """
        # Find all JSON files recursively
        pattern = os.path.join(self.input_dir, "**", "*.json")
        json_files = glob.glob(pattern, recursive=True)
        
        # Filter out checkpoint files if main dataset exists
        main_files = [f for f in json_files if 'temp_results' not in f]
        temp_files = [f for f in json_files if 'temp_results' in f]
        
        if main_files and temp_files:
            print(f"Found {len(main_files)} main files and {len(temp_files)} checkpoint files")
            response = input("Include checkpoint files? They may be duplicates. (y/N): ").strip().lower()
            if response != 'y':
                json_files = main_files
                print(f"Processing {len(json_files)} main files only")
        
        if not json_files:
            print(f"No JSON files found in {self.input_dir}")
            return {}
        
        # Count total entries
        total_entries, file_entry_counts = self.count_total_entries(json_files)
        
        print(f"\n{'='*60}")
        print(f"Found {len(json_files)} files with {total_entries:,} total entries")
        print(f"Using {self.max_workers} parallel workers")
        print(f"{'='*60}\n")
        
        all_stats = {}
        
        # Process files in parallel with entry-level progress
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {}
            for filepath in json_files:
                future = executor.submit(
                    self._process_single_file_with_progress, 
                    filepath,
                    file_entry_counts[filepath]
                )
                future_to_file[future] = filepath
            
            # Main progress bar for total entries
            with tqdm(total=total_entries, desc="Processing entries", unit="entry") as pbar:
                completed_entries = 0
                completed_files = 0
                
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    filename = os.path.basename(filepath)
                    entries_in_file = file_entry_counts[filepath]
                    
                    try:
                        file_stats, entry_progress = future.result()
                        if file_stats:
                            all_stats[filename] = file_stats
                            
                            # Update progress
                            pbar.update(entries_in_file)
                            completed_entries += entries_in_file
                            completed_files += 1
                            
                            # Show file completion status
                            accuracy_str = f"YES: {file_stats['yes_accuracy']*100:.1f}% NO: {file_stats['no_accuracy']*100:.1f}%"
                            pbar.set_postfix_str(
                                f"Files: {completed_files}/{len(json_files)} | "
                                f"Last: {filename[:20]}... | {accuracy_str}"
                            )
                            
                            # Print significant findings
                            if file_stats['total_removed_yes'] > 0 or file_stats['total_removed_no'] > 0:
                                print(f"\n✓ {filename}: Removed {file_stats['total_removed_yes']} false positives, "
                                      f"{file_stats['total_removed_no']} false negatives")
                    except Exception as e:
                        print(f"\n✗ Error processing {filename}: {e}")
                        pbar.update(entries_in_file)
                        completed_files += 1
        
        return all_stats
    
    def _process_single_file_with_progress(self, filepath: str, expected_entries: int) -> Tuple[Dict[str, Any], List[int]]:
        """
        Process a single file with progress tracking.
        
        Args:
            filepath: Path to JSON file
            expected_entries: Expected number of entries (for progress)
            
        Returns:
            Tuple of (statistics dict, list of entries processed at each step)
        """
        filename = os.path.basename(filepath)
        progress_updates = []
        
        # Load data
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, []
        
        # Handle both list and single entry formats
        if not isinstance(data, list):
            data = [data]
        
        validated_entries = []
        stats_list = []
        
        # Process each entry with internal progress tracking
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            
            # Validate entry
            validated_entry, stats = self.validate_entry(entry)
            validated_entries.append(validated_entry)
            stats_list.append(stats)
            progress_updates.append(i + 1)
        
        if not validated_entries:
            return None, progress_updates
        
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
        
        return file_stats, progress_updates
    
    def print_detailed_summary(self, all_stats: Dict[str, Dict[str, Any]]):
        """Print detailed summary with better formatting."""
        if not all_stats:
            print("No files were processed successfully.")
            return
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        # Overall statistics
        total_entries = sum(s['entries'] for s in all_stats.values())
        total_original_yes = sum(s['total_original_yes'] for s in all_stats.values())
        total_original_no = sum(s['total_original_no'] for s in all_stats.values())
        total_filtered_yes = sum(s['total_filtered_yes'] for s in all_stats.values())
        total_filtered_no = sum(s['total_filtered_no'] for s in all_stats.values())
        total_removed_yes = sum(s['total_removed_yes'] for s in all_stats.values())
        total_removed_no = sum(s['total_removed_no'] for s in all_stats.values())
        
        overall_yes_accuracy = total_filtered_yes / total_original_yes if total_original_yes > 0 else 1.0
        overall_no_accuracy = total_filtered_no / total_original_no if total_original_no > 0 else 1.0
        overall_accuracy = (total_filtered_yes + total_filtered_no) / (total_original_yes + total_original_no) if (total_original_yes + total_original_no) > 0 else 1.0
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total files processed: {len(all_stats)}")
        print(f"  Total entries: {total_entries:,}")
        print(f"  Total strings validated: {total_original_yes + total_original_no:,}")
        
        print(f"\n✅ YES Strings:")
        print(f"  Original: {total_original_yes:,}")
        print(f"  Kept: {total_filtered_yes:,}")
        print(f"  Removed (false positives): {total_removed_yes:,}")
        print(f"  Accuracy: {overall_yes_accuracy*100:.2f}%")
        
        print(f"\n❌ NO Strings:")
        print(f"  Original: {total_original_no:,}")
        print(f"  Kept: {total_filtered_no:,}")
        print(f"  Removed (false negatives): {total_removed_no:,}")
        print(f"  Accuracy: {overall_no_accuracy*100:.2f}%")
        
        print(f"\n📈 Overall Accuracy: {overall_accuracy*100:.2f}%")
        
        # File-by-file summary (sorted by accuracy)
        print(f"\n📁 Per-file Summary (sorted by accuracy):")
        sorted_files = sorted(all_stats.items(), 
                            key=lambda x: (x[1]['yes_accuracy'] + x[1]['no_accuracy'])/2)
        
        for filename, stats in sorted_files[:10]:  # Show worst 10
            avg_accuracy = (stats['yes_accuracy'] + stats['no_accuracy']) / 2
            print(f"  {filename[:40]:40} | "
                  f"Entries: {stats['entries']:4} | "
                  f"Accuracy: {avg_accuracy*100:5.1f}% | "
                  f"Removed: YES={stats['total_removed_yes']:3} NO={stats['total_removed_no']:3}")
        
        if len(sorted_files) > 10:
            print(f"  ... and {len(sorted_files) - 10} more files")
        
        # Save summary to file
        stats_file = os.path.join(self.output_dir, '_validation_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n💾 Detailed statistics saved to {stats_file}")


def main():
    """Main entry point for parallel data validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate regex golf datasets with better progress tracking')
    parser.add_argument('--input', default='../generated_data', 
                        help='Input directory with JSON files')
    parser.add_argument('--output', default='../validated_data',
                        help='Output directory for validated files')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: number of CPUs)')
    
    args = parser.parse_args()
    
    # Use parallel validator with improved progress
    validator = ParallelDataValidator(args.input, args.output, args.workers)
    
    print("🚀 Starting parallel validation with entry-level progress tracking...")
    start_time = time.perf_counter()
    
    all_stats = validator.process_all_files_parallel()
    
    elapsed = time.perf_counter() - start_time
    print(f"\n⏱️  Processing completed in {elapsed:.2f} seconds")
    print(f"   Speed: {sum(s['entries'] for s in all_stats.values())/elapsed:.1f} entries/second")
    
    # Print detailed summary
    validator.print_detailed_summary(all_stats)


if __name__ == "__main__":
    main()