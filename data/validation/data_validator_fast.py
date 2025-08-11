"""
Fast data validator with proper timeout handling using multiprocessing.
"""
import json
import re
import os
import glob
import warnings
import time
from multiprocessing import Process, Queue, TimeoutError
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Suppress regex compilation warnings for speed
warnings.filterwarnings('ignore', category=FutureWarning)
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from data_validator import ValidationStats, DataValidator


def validate_entry_worker(entry: Dict[str, Any], result_queue: Queue):
    """Worker function to validate a single entry in a separate process."""
    try:
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
            # Invalid regex
            result = {
                'validated_entry': {
                    'expr': regex_str,
                    'yes': [],
                    'no': [],
                    '_validation_error': str(e)
                },
                'stats': {
                    'original_yes': original_yes,
                    'original_no': original_no,
                    'filtered_yes': 0,
                    'filtered_no': 0
                }
            }
            result_queue.put(result)
            return
        
        # Filter YES list - keep only strings that match
        filtered_yes = []
        for s in yes_list:
            try:
                if pattern.fullmatch(s):
                    filtered_yes.append(s)
            except:
                pass  # Skip problematic strings
        
        # Filter NO list - keep only strings that DON'T match  
        filtered_no = []
        for s in no_list:
            try:
                if not pattern.fullmatch(s):
                    filtered_no.append(s)
            except:
                pass  # Skip problematic strings
        
        result = {
            'validated_entry': {
                'expr': regex_str,
                'yes': filtered_yes,
                'no': filtered_no
            },
            'stats': {
                'original_yes': original_yes,
                'original_no': original_no,
                'filtered_yes': len(filtered_yes),
                'filtered_no': len(filtered_no)
            }
        }
        result_queue.put(result)
        
    except Exception as e:
        # Catch-all for any other errors
        result = {
            'validated_entry': {
                'expr': entry.get('expr', ''),
                'yes': [],
                'no': [],
                '_validation_error': str(e)
            },
            'stats': {
                'original_yes': len(entry.get('yes', [])),
                'original_no': len(entry.get('no', [])),
                'filtered_yes': 0,
                'filtered_no': 0
            }
        }
        result_queue.put(result)


class FastValidator:
    """Fast validator with hard timeouts using multiprocessing."""
    
    def __init__(self, input_dir: str = "../generated_data", 
                 output_dir: str = "../validated_data",
                 entry_timeout: float = 1.0):
        """
        Initialize fast validator.
        
        Args:
            input_dir: Directory containing JSON files to validate
            output_dir: Directory to save validated files
            entry_timeout: Maximum seconds per entry before hard kill
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.entry_timeout = entry_timeout
        self.timed_out_entries = []
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def validate_entry_with_timeout(self, entry: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """
        Validate entry with hard timeout using separate process.
        """
        result_queue = Queue()
        process = Process(target=validate_entry_worker, args=(entry, result_queue))
        
        process.start()
        process.join(timeout=self.entry_timeout)
        
        if process.is_alive():
            # Timeout - kill the process
            process.terminate()
            process.join()
            
            regex_str = entry.get('expr', '')[:50] + '...'
            self.timed_out_entries.append(regex_str)
            
            # Return timeout result
            return {
                'expr': entry.get('expr', ''),
                'yes': [],
                'no': [],
                '_validation_error': f'Timeout after {self.entry_timeout}s'
            }, {
                'original_yes': len(entry.get('yes', [])),
                'original_no': len(entry.get('no', [])),
                'filtered_yes': 0,
                'filtered_no': 0
            }
        
        # Get result from queue
        try:
            result = result_queue.get_nowait()
            return result['validated_entry'], result['stats']
        except:
            # No result available
            return {
                'expr': entry.get('expr', ''),
                'yes': [],
                'no': [],
                '_validation_error': 'Process failed'
            }, {
                'original_yes': len(entry.get('yes', [])),
                'original_no': len(entry.get('no', [])),
                'filtered_yes': 0,
                'filtered_no': 0
            }
    
    def process_all_files(self):
        """Process all JSON files with progress tracking."""
        # Find all JSON files
        pattern = os.path.join(self.input_dir, "**", "*.json")
        json_files = glob.glob(pattern, recursive=True)
        
        # Filter out checkpoint files
        main_files = [f for f in json_files if 'temp_results' not in f]
        
        if not main_files:
            print(f"No JSON files found in {self.input_dir}")
            return {}
        
        print(f"\n📊 Found {len(main_files)} files to process")
        
        # Count total entries
        total_entries = 0
        file_data = []
        for filepath in main_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
                count = len(data)
                total_entries += count
                file_data.append((filepath, data))
                print(f"  • {os.path.basename(filepath)}: {count:,} entries")
            except Exception as e:
                print(f"  ✗ Error reading {os.path.basename(filepath)}: {e}")
        
        print(f"\n📝 Total entries: {total_entries:,}")
        print(f"⚡ Using {self.entry_timeout}s timeout per entry")
        print("="*60)
        
        all_stats = {}
        
        # Process all entries with single progress bar
        with tqdm(total=total_entries, desc="Validating", unit=" entries") as pbar:
            for filepath, data in file_data:
                filename = os.path.basename(filepath)
                validated_entries = []
                file_stats = {
                    'total_original_yes': 0,
                    'total_original_no': 0,
                    'total_filtered_yes': 0,
                    'total_filtered_no': 0,
                    'total_removed_yes': 0,
                    'total_removed_no': 0
                }
                
                for i, entry in enumerate(data):
                    if not isinstance(entry, dict):
                        pbar.update(1)
                        continue
                    
                    # Validate with timeout
                    validated_entry, stats = self.validate_entry_with_timeout(entry)
                    validated_entries.append(validated_entry)
                    
                    # Update stats
                    file_stats['total_original_yes'] += stats['original_yes']
                    file_stats['total_original_no'] += stats['original_no']
                    file_stats['total_filtered_yes'] += stats['filtered_yes']
                    file_stats['total_filtered_no'] += stats['filtered_no']
                    file_stats['total_removed_yes'] += stats['original_yes'] - stats['filtered_yes']
                    file_stats['total_removed_no'] += stats['original_no'] - stats['filtered_no']
                    
                    # Update progress
                    pbar.update(1)
                    if (i + 1) % 20 == 0:
                        pbar.set_postfix_str(f"{filename[:30]}... [{i+1}/{len(data)}]")
                
                # Save validated data
                output_path = os.path.join(self.output_dir, filename)
                with open(output_path, 'w') as f:
                    json.dump(validated_entries, f, indent=2)
                
                # Calculate accuracy
                if file_stats['total_original_yes'] > 0:
                    file_stats['yes_accuracy'] = file_stats['total_filtered_yes'] / file_stats['total_original_yes']
                else:
                    file_stats['yes_accuracy'] = 1.0
                    
                if file_stats['total_original_no'] > 0:
                    file_stats['no_accuracy'] = file_stats['total_filtered_no'] / file_stats['total_original_no']
                else:
                    file_stats['no_accuracy'] = 1.0
                
                file_stats['entries'] = len(validated_entries)
                all_stats[filename] = file_stats
                
                # Report file completion
                tqdm.write(f"✓ {filename}: {len(validated_entries)} entries "
                          f"(YES: {file_stats['yes_accuracy']*100:.1f}%, NO: {file_stats['no_accuracy']*100:.1f}%)")
        
        return all_stats
    
    def print_summary(self, all_stats):
        """Print validation summary."""
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)
        
        if self.timed_out_entries:
            print(f"\n⚠️  {len(self.timed_out_entries)} entries timed out:")
            for regex in self.timed_out_entries[:5]:
                print(f"  • {regex}")
            if len(self.timed_out_entries) > 5:
                print(f"  ... and {len(self.timed_out_entries) - 5} more")
        
        # Overall stats
        total_entries = sum(s['entries'] for s in all_stats.values())
        total_removed = sum(s['total_removed_yes'] + s['total_removed_no'] for s in all_stats.values())
        
        print(f"\n📊 Summary:")
        print(f"  Files: {len(all_stats)}")
        print(f"  Entries: {total_entries:,}")
        print(f"  Strings removed: {total_removed:,}")
        
        # Save stats
        stats_file = os.path.join(self.output_dir, '_validation_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n💾 Stats saved to {stats_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast regex validator with hard timeouts')
    parser.add_argument('--timeout', type=float, default=1.0,
                        help='Timeout per entry in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    print("🚀 Fast Validator with Hard Timeouts")
    validator = FastValidator(entry_timeout=args.timeout)
    
    start_time = time.perf_counter()
    all_stats = validator.process_all_files()
    elapsed = time.perf_counter() - start_time
    
    validator.print_summary(all_stats)
    
    if all_stats:
        total_entries = sum(s['entries'] for s in all_stats.values())
        print(f"\n⏱️  Completed in {elapsed:.1f} seconds")
        print(f"   Speed: {total_entries/elapsed:.1f} entries/second")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  # Required for macOS
    main()