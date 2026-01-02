"""Analyze test_big.py results and generate comprehensive data table.

This script parses all log files from the test_big.py experiment suite,
extracts performance metrics, and creates a structured analysis table
suitable for LaTeX and thesis documentation.

INSTANCE GENERATION PATTERNS (from test_big.py):
==================================================

Pattern 1 (Uniform Ratios): 
    Items with consistent duration-to-price ratios (1.5-3.0).
    Durations: 3-15, prices calculated from ratio.
    Creates balanced, predictable instances with similar cost-efficiency across jobs.

Pattern 2 (High Variance):
    Alternates between cheap-long jobs (duration 10-20, price 1-3) 
    and expensive-short jobs (duration 2-5, price 8-15).
    Diverse job portfolios with extreme cost/time tradeoffs.

Pattern 3 (Increasing Complexity):
    Systematic progression where duration = 3+j*2, price = 2+j.
    Jobs become progressively longer and more expensive.
    Structured, predictable difficulty gradient.

Pattern 4 (Random Uniform):
    Completely random within realistic bounds (duration 4-18, price 2-12).
    No special structure, tests general algorithm behavior.

Pattern 5 (Extreme Cases):
    30% probability: very long & very cheap (duration 20-30, price 1-3)
    70% probability: normal range (duration 3-12, price 3-10)
    Tests edge cases and unusual configurations.

==================================================
"""

import os
import re
import json
from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

LOG_DIR = Path("logs/test_big")
CACHE_FILE = Path("enumeration_results_cache_big.json")
OUTPUT_CSV = Path("test_results_analysis.csv")


# ============================================================
# PATTERN DESCRIPTIONS
# ============================================================

PATTERN_NAMES = {
    0: "Pattern 1: Uniform Ratios",
    1: "Pattern 2: High Variance",
    2: "Pattern 3: Increasing",
    3: "Pattern 4: Random Uniform",
    4: "Pattern 5: Extreme Cases"
}


# ============================================================
# PARSING FUNCTIONS
# ============================================================

def parse_instance_name(filename):
    """Extract instance details from filename.
    
    Example: Complex_001_J6_M3_B12_20251221_101759.log
    Returns: (instance_id, n_jobs, n_machines, budget, instance_name)
    """
    pattern = r"Complex_(\d+)_J(\d+)_M(\d+)_B(\d+)"
    match = re.search(pattern, filename)
    if match:
        instance_id = int(match.group(1))
        n_jobs = int(match.group(2))
        n_machines = int(match.group(3))
        budget = int(match.group(4))
        instance_name = f"Complex_{instance_id:03d}_J{n_jobs}_M{n_machines}_B{budget}"
        return instance_id, n_jobs, n_machines, budget, instance_name
    return None, None, None, None, None


def determine_pattern(instance_id):
    """Determine generation pattern from instance ID.
    
    Pattern is based on (instance_id - 1) % 5.
    """
    pattern_index = (instance_id - 1) % 5
    return PATTERN_NAMES[pattern_index]


def parse_log_file(log_path):
    """Parse a single log file and extract all metrics.
    
    Extracts from both the Branch-and-bound completed section and ENUMERATION COMPARISON section.
    """
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract metrics using regex
        data = {}
        
        # Extract from "Branch-and-bound completed" section
        # These are more accurate than the comparison section
        match = re.search(r"Nodes explored:\s*(\d+)", content)
        data['bnb_nodes_explored'] = int(match.group(1)) if match else None
        
        match = re.search(r"Nodes pruned:\s*(\d+)", content)
        data['bnb_nodes_pruned_bound'] = int(match.group(1)) if match else None
        
        match = re.search(r"Nodes evaluated:\s*(\d+)", content)
        data['bnb_nodes_evaluated'] = int(match.group(1)) if match else None
        
        match = re.search(r"Total runtime:\s*([\d.]+) seconds", content)
        data['bnb_runtime_sec'] = float(match.group(1)) if match else None
        
        # Find the enumeration comparison section for remaining metrics
        comparison_section = content.split("ENUMERATION COMPARISON")[-1]
        
        # BnB makespan from comparison section
        match = re.search(r"BnB makespan:\s*([\d.]+)", comparison_section)
        data['bnb_makespan'] = float(match.group(1)) if match else None
        
        # Enumeration metrics
        match = re.search(r"Enumeration nodes evaluated:\s*(\d+)", comparison_section)
        data['enum_nodes_checked'] = int(match.group(1)) if match else None
        
        match = re.search(r"Enumeration runtime:\s*([\d.]+)s", comparison_section)
        data['enum_runtime_sec'] = float(match.group(1)) if match else None
        
        match = re.search(r"Enumeration makespan:\s*([\d.]+)", comparison_section)
        data['enum_makespan'] = float(match.group(1)) if match else None
        
        # Comparison metrics
        match = re.search(r"Speedup \(Enum/BnB\):\s*([\d.]+)x", comparison_section)
        data['speedup_factor'] = float(match.group(1)) if match else None
        
        match = re.search(r"Match:\s*(YES|NO)", comparison_section)
        data['results_match'] = (match.group(1) == "YES") if match else None
        
        match = re.search(r"Timed out:\s*(True|False)", comparison_section)
        timed_out = (match.group(1) == "True") if match else False
        
        # Determine verification status
        if timed_out:
            data['verification_status'] = "TIMEOUT"
            data['enum_status'] = "TIMEOUT"
        elif data.get('results_match') is False:
            data['verification_status'] = "WARNING"
            data['enum_status'] = "VERIFIED"
        else:
            data['verification_status'] = "VERIFIED"
            data['enum_status'] = "VERIFIED"
        
        return data
        
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None


def parse_metrics_json(json_path):
    """Parse metrics JSON file for additional BnB status info."""
    try:
        with open(json_path, 'r') as f:
            metrics = json.load(f)
        
        perf = metrics.get('performance', {})
        
        data = {
            'bnb_status': 'SUCCESS'  # Default, will update if needed
        }
        
        # Check if node limit was reached
        nodes_explored = perf.get('nodes_explored', 0)
        if nodes_explored >= 100000:
            data['bnb_status'] = 'NODE_LIMIT'
        
        return data
        
    except Exception as e:
        print(f"Error parsing {json_path}: {e}")
        return None


def load_enumeration_cache():
    """Load enumeration cache for cross-reference."""
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        return cache
    except Exception as e:
        print(f"Warning: Could not load enumeration cache: {e}")
        return {}


# ============================================================
# MAIN ANALYSIS
# ============================================================

def collect_all_data():
    """Collect data from all log files and build comprehensive dataset."""
    
    print("="*70)
    print("ANALYZING TEST_BIG.PY RESULTS")
    print("="*70)
    
    # Find all log files
    log_files = sorted(LOG_DIR.glob("Complex_*_*.log"))
    print(f"\nFound {len(log_files)} log files")
    
    # Load enumeration cache
    print("Loading enumeration cache...")
    cache = load_enumeration_cache()
    
    # Collect data
    all_data = []
    
    for log_file in log_files:
        # Parse instance name
        instance_id, n_jobs, n_machines, budget, instance_name = parse_instance_name(log_file.name)
        
        if instance_id is None:
            print(f"Warning: Could not parse filename {log_file.name}")
            continue
        
        # Parse log file
        log_data = parse_log_file(log_file)
        if log_data is None:
            print(f"Warning: Could not parse log file {log_file.name}")
            continue
        
        # Parse metrics JSON
        json_file = log_file.with_suffix('').name + "_metrics.json"
        json_path = LOG_DIR / json_file
        metrics_data = parse_metrics_json(json_path)
        
        # Combine all data
        row = {
            'instance_id': instance_id,
            'instance_name': instance_name,
            'pattern_type': determine_pattern(instance_id),
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'budget': budget,
        }
        
        # Add log data
        row.update(log_data)
        
        # Add metrics data
        if metrics_data:
            row.update(metrics_data)
        
        all_data.append(row)
        
        if instance_id % 10 == 0:
            print(f"  Processed {instance_id}/100 instances...")
    
    print(f"\nSuccessfully parsed {len(all_data)} instances")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by instance ID
    df = df.sort_values('instance_id')
    
    # Reorder columns for better readability
    column_order = [
        'instance_id',
        'instance_name',
        'pattern_type',
        'n_jobs',
        'n_machines',
        'budget',
        'bnb_runtime_sec',
        'bnb_nodes_explored',
        'bnb_nodes_pruned_bound',
        'bnb_nodes_evaluated',
        'bnb_makespan',
        'bnb_status',
        'enum_runtime_sec',
        'enum_nodes_checked',
        'enum_makespan',
        'enum_status',
        'speedup_factor',
        'results_match',
        'verification_status'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    return df


def display_summary_statistics(df):
    """Display summary statistics for key metrics."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Pattern distribution
    print("\nInstance Distribution by Pattern:")
    pattern_counts = df['pattern_type'].value_counts().sort_index()
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} instances")
    
    # Verification status
    print("\nVerification Status:")
    status_counts = df['verification_status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count} instances")
    
    # Runtime statistics
    print("\nRuntime Statistics:")
    print(f"  BnB Runtime:  min={df['bnb_runtime_sec'].min():.4f}s, "
          f"max={df['bnb_runtime_sec'].max():.4f}s, "
          f"mean={df['bnb_runtime_sec'].mean():.4f}s")
    
    verified = df[df['verification_status'] == 'VERIFIED']
    if len(verified) > 0:
        print(f"  Enum Runtime (verified only): min={verified['enum_runtime_sec'].min():.4f}s, "
              f"max={verified['enum_runtime_sec'].max():.4f}s, "
              f"mean={verified['enum_runtime_sec'].mean():.4f}s")
    
    # Speedup statistics (only for verified instances)
    if len(verified) > 0:
        print(f"\nSpeedup Statistics (verified instances only):")
        print(f"  min={verified['speedup_factor'].min():.2f}x, "
              f"max={verified['speedup_factor'].max():.2f}x, "
              f"mean={verified['speedup_factor'].mean():.2f}x, "
              f"median={verified['speedup_factor'].median():.2f}x")
    
    # Node statistics
    print(f"\nNode Statistics:")
    print(f"  BnB Nodes Explored:  min={df['bnb_nodes_explored'].min()}, "
          f"max={df['bnb_nodes_explored'].max()}, "
          f"mean={df['bnb_nodes_explored'].mean():.0f}")
    
    if len(verified) > 0:
        print(f"  Enum Nodes Checked (verified): min={verified['enum_nodes_checked'].min()}, "
              f"max={verified['enum_nodes_checked'].max()}, "
              f"mean={verified['enum_nodes_checked'].mean():.0f}")
    
    # Results match rate
    match_rate = (df['results_match'].sum() / len(df)) * 100
    print(f"\nResults Match Rate: {match_rate:.1f}% ({df['results_match'].sum()}/{len(df)} instances)")


def main():
    """Main analysis workflow."""
    
    # Collect all data
    df = collect_all_data()
    
    # Display summary statistics
    display_summary_statistics(df)
    
    # Display full table
    print("\n" + "="*70)
    print("FULL DATA TABLE (first 20 rows)")
    print("="*70)
    
    # Configure pandas display options for better visibility
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print(df.head(20).to_string(index=False))
    
    # Export to CSV
    print("\n" + "="*70)
    print("EXPORTING DATA")
    print("="*70)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nData exported to: {OUTPUT_CSV}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Display file info
    file_size = OUTPUT_CSV.stat().st_size / 1024
    print(f"File size: {file_size:.2f} KB")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the CSV file in VSCode or Excel")
    print("  2. Import into LaTeX using \\csvreader or similar")
    print("  3. Copy to Thesis_bilevel_tu_bs folder for inclusion in thesis")
    
    return df


if __name__ == "__main__":
    df = main()
