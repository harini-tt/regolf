"""
Parallel data validator for regex golf datasets.
Optimized for processing large numbers of files.
"""
import json
import re
import os
import glob
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import time

from data_validator import ValidationStats, DataValidator


class ParallelDataValidator(DataValidator):
    """Parallel version of DataValidator for processing many files."""
    
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
    
    def process_all_files_parallel(self) -> Dict[str, Dict[str, Any]]:
        """
        Process all JSON files in parallel.
        
        Returns:
            Dictionary with statistics for each file
        """
        # Find all JSON files
        pattern = os.path.join(self.input_dir, "*.json")
        json_files = glob.glob(pattern)
        
        if not json_files:
            print(f"No JSON files found in {self.input_dir}")
            return {}
        
        print(f"Found {len(json_files)} JSON files to process")
        print(f"Using {self.max_workers} parallel workers")
        
        all_stats = {}
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, filepath): filepath
                for filepath in json_files
            }
            
            # Process results as they complete with progress bar
            with tqdm(total=len(json_files), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    filename = os.path.basename(filepath)
                    
                    try:
                        file_stats = future.result()
                        if file_stats:
                            all_stats[filename] = file_stats
                    except Exception as e:
                        print(f"\nError processing {filename}: {e}")
                    
                    pbar.update(1)
        
        return all_stats
    
    def _process_single_file(self, filepath: str) -> Dict[str, Any]:
        """
        Process a single file (used by parallel workers).
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Statistics dictionary for the file
        """
        filename = os.path.basename(filepath)
        
        # Validate file
        validated_entries, stats_list = self.validate_file(filepath)
        
        if not validated_entries:
            return None
        
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
        
        return file_stats
    
    def benchmark(self, sample_size: int = 10) -> Dict[str, float]:
        """
        Benchmark sequential vs parallel processing.
        
        Args:
            sample_size: Number of files to use for benchmarking
            
        Returns:
            Dictionary with timing results
        """
        # Find files
        pattern = os.path.join(self.input_dir, "*.json")
        json_files = glob.glob(pattern)[:sample_size]
        
        if not json_files:
            print("No files to benchmark")
            return {}
        
        print(f"Benchmarking with {len(json_files)} files...")
        
        # Sequential timing
        start = time.perf_counter()
        for filepath in json_files:
            self._process_single_file(filepath)
        sequential_time = time.perf_counter() - start
        
        # Parallel timing
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_single_file, f) for f in json_files]
            for future in as_completed(futures):
                future.result()
        parallel_time = time.perf_counter() - start
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        
        results = {
            'files': len(json_files),
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'workers': self.max_workers
        }
        
        print(f"\nBenchmark Results:")
        print(f"  Sequential: {sequential_time:.2f}s")
        print(f"  Parallel ({self.max_workers} workers): {parallel_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        return results


def main():
    """Main entry point for parallel data validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate regex golf datasets')
    parser.add_argument('--input', default='../generated_data', 
                        help='Input directory with JSON files')
    parser.add_argument('--output', default='../validated_data',
                        help='Output directory for validated files')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: number of CPUs)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark comparison')
    parser.add_argument('--sequential', action='store_true',
                        help='Use sequential processing instead of parallel')
    
    args = parser.parse_args()
    
    if args.sequential:
        # Use original sequential validator
        validator = DataValidator(args.input, args.output)
        print("Using sequential processing...")
        all_stats = validator.process_all_files()
    else:
        # Use parallel validator
        validator = ParallelDataValidator(args.input, args.output, args.workers)
        
        if args.benchmark:
            validator.benchmark()
            return
        
        print("Using parallel processing...")
        start_time = time.perf_counter()
        all_stats = validator.process_all_files_parallel()
        elapsed = time.perf_counter() - start_time
        print(f"\nProcessing completed in {elapsed:.2f} seconds")
    
    # Print summary
    validator.print_summary(all_stats)
    
    # Save statistics to file
    stats_file = os.path.join(args.output, "_validation_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\nValidation statistics saved to {stats_file}")
    print("\nValidation complete!")


if __name__ == "__main__":
    main()