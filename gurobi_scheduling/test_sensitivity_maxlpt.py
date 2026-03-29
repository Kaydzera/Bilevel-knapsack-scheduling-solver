"""
Sensitivity Analysis: Ceiling vs Max-LPT Bound Comparison

This script tests how the relative performance of ceiling vs Max-LPT bounds
changes when varying problem parameters:
- Number of job types (n)
- Number of machines (m)
- Budget multiplier (relative to average job cost)

For each parameter value:
1. Generate multiple random instances until we find one that is not solved at the root node and both bounds solve within time limit
2. Run BnB with BOTH ceiling and maxlpt bounds
3. Compare metrics: runtime, nodes explored, pruning effectiveness
4. Save results to CSV for analysis

Usage:
    python test_sensitivity_maxlpt.py --parameter jobs --start 4 --step 2 --repetitions 5
    python test_sensitivity_maxlpt.py --parameter machines --start 2 --step 1 --repetitions 5
    python test_sensitivity_maxlpt.py --parameter budget --start 2.0 --step 0.5 --repetitions 5
"""

import argparse
import csv
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

import random

from models import MainProblem
from bnb import run_bnb_classic
from logger import create_logger



# Baseline configuration
BASELINE = {
    'n_jobs': 20,
    'm_machines': 10,
    'budget_multiplier': 2.5,  # Budget = 2.5x average job cost * number of machines
    'max_nodes': 1_000_000_000,  # Increased node limit to 1 billion
    'time_limit': 3600.0,  # 1 hour per instance
}

# Leaf timeout in seconds (15 minutes)
LEAF_TIMEOUT = 900.0

# Stopping criteria
MAX_MACHINES = 50
TIMEOUT_THRESHOLD = 3600.0


def generate_instance(n_jobs: int, m_machines: int, budget_multiplier: float, seed: int) -> MainProblem:
    """Generate a test instance with specified parameters."""
    random.seed(seed)
    
    prices = [random.randint(10, 100) for _ in range(n_jobs)]
    durations = [random.randint(5, 50) for _ in range(n_jobs)]

    avg_job_cost = sum(prices) / n_jobs
    budget = int(avg_job_cost * m_machines * budget_multiplier)

    return MainProblem(
        prices=prices,
        durations=durations,
        anzahl_maschinen=m_machines,
        budget_total=budget
    )


def run_single_bound(problem: MainProblem, bound_type: str, instance_name: str,
                    max_nodes: int, time_limit: float, log_dir: str) -> Dict:
    """
    Run BnB with specified bound and collect performance metrics.
    
    Args:
        problem: MainProblem instance
        bound_type: 'ceiling' or 'maxlpt'
        instance_name: Name for logging
        max_nodes: Node limit
        time_limit: Time limit in seconds
        log_dir: Directory for logs
        
    Returns:
        Dictionary with metrics
    """
    # Logging is disabled; collect metrics directly from BnB result
    start_time = time.time()
    try:
        result = run_bnb_classic(
            problem=problem,
            max_nodes=max_nodes,
            logger=None,  # No logger
            instance_name=instance_name,
            bound_type=bound_type,
            verbose=False,
            time_limit=time_limit,
            leaf_timeout=LEAF_TIMEOUT  # Pass leaf timeout to BnB if supported
        )
        runtime = time.time() - start_time
        status = 'success'

        # Try to extract metrics directly from result
        metrics = {
            'runtime': runtime,
            'nodes_explored': result.get('nodes_explored', 0),
            'pruned_bound': result.get('pruned_bound', 0),
            'best_makespan': result.get('best_obj', float('inf')),
            'status': status,
            'leaf_timeout_hit': result.get('leaf_timeout_hit', False),
            'leaf_nodes_evaluated': result.get('nodes_evaluated', 0),
        }

        total_nodes = metrics['nodes_explored']
        pruned_bound = metrics['pruned_bound']
        if total_nodes > 0:
            metrics['pruning_rate'] = pruned_bound / total_nodes
        else:
            metrics['pruning_rate'] = 0.0

    except TimeoutError:
        runtime = time.time() - start_time
        metrics = {
            'runtime': runtime,
            'nodes_explored': 0,
            'pruned_bound': 0,
            'best_makespan': float('inf'),
            'status': 'timeout',
            'pruning_rate': 0.0,
            'leaf_timeout_hit': False,
        }
    except Exception as e:
        runtime = time.time() - start_time
        metrics = {
            'runtime': runtime,
            'nodes_explored': 0,
            'pruned_bound': 0,
            'best_makespan': float('inf'),
            'status': f'error: {str(e)}',
            'pruning_rate': 0.0,
            'leaf_timeout_hit': False,
        }

    return metrics


def run_comparison_test(n_jobs: int, m_machines: int, budget_multiplier: float,
                       seed: int, max_nodes: int, time_limit: float, log_dir: str) -> Dict:
    """
    Run BnB with BOTH bounds on the same instance and compare.
    
    Returns:
        Dictionary with metrics for both bounds and comparison
    """
    # Generate instance once
    problem = generate_instance(n_jobs, m_machines, budget_multiplier, seed)
    
    instance_name = f"sensitivity_{n_jobs}j_{m_machines}m_{budget_multiplier:.1f}b_s{seed}"

    # Run ceiling bound (no log_dir)
    ceiling_metrics = run_single_bound(
        problem, 'ceiling', instance_name, max_nodes, time_limit, None
    )

    maxlpt_metrics = run_single_bound(
        problem, 'maxlpt', instance_name, max_nodes, time_limit, None
    )
    
    # Combine results
    combined = {
        'n_jobs': n_jobs,
        'm_machines': m_machines,
        'budget_multiplier': budget_multiplier,
        'seed': seed,

        # Ceiling metrics
        'ceiling_runtime': ceiling_metrics['runtime'],
        'ceiling_nodes': ceiling_metrics['nodes_explored'],
        'ceiling_pruned_bound': ceiling_metrics['pruned_bound'],
        'ceiling_pruning_rate': ceiling_metrics['pruning_rate'],
        'ceiling_makespan': ceiling_metrics['best_makespan'],
        'ceiling_status': ceiling_metrics['status'],
        'ceiling_leaf_timeout': ceiling_metrics.get('leaf_timeout_hit', False),
        'ceiling_leaf_nodes': ceiling_metrics.get('leaf_nodes_evaluated', 0),

        # Max-LPT metrics
        'maxlpt_runtime': maxlpt_metrics['runtime'],
        'maxlpt_nodes': maxlpt_metrics['nodes_explored'],
        'maxlpt_pruned_bound': maxlpt_metrics['pruned_bound'],
        'maxlpt_pruning_rate': maxlpt_metrics['pruning_rate'],
        'maxlpt_makespan': maxlpt_metrics['best_makespan'],
        'maxlpt_status': maxlpt_metrics['status'],
        'maxlpt_leaf_timeout': maxlpt_metrics.get('leaf_timeout_hit', False),
        'maxlpt_leaf_nodes': maxlpt_metrics.get('leaf_nodes_evaluated', 0),

        # Comparison metrics
        'match_makespan': ceiling_metrics['best_makespan'] == maxlpt_metrics['best_makespan'],
    }
    
    # Calculate relative improvements (only if both succeeded)
    if (ceiling_metrics['status'] == 'success' and 
        maxlpt_metrics['status'] == 'success' and
        ceiling_metrics['nodes_explored'] > 0):
        
        combined['node_reduction_pct'] = 100 * (
            1 - maxlpt_metrics['nodes_explored'] / ceiling_metrics['nodes_explored']
        )
        combined['speedup'] = ceiling_metrics['runtime'] / maxlpt_metrics['runtime']
    else:
        combined['node_reduction_pct'] = None
        combined['speedup'] = None
    
    return combined


def run_sensitivity_analysis(parameter: str, start_value: float, step: float,
                            repetitions: int, output_dir: str, log_dir: str,
                            resume_rep: int = 0):
    """
    Run sensitivity analysis comparing ceiling vs maxlpt bounds.
    
    Args:
        parameter: Parameter to vary ('jobs', 'machines', 'budget')
        start_value: Starting value for parameter
        step: Step size for parameter increments
        repetitions: Number of repetitions per value
        output_dir: Directory for results CSV (with timestamp)
        log_dir: Directory for log files (with timestamp)
        resume_rep: Which repetition to start from (0 = start fresh, >0 = resume)
    """
    # Create results directory only
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Output CSV file path
    output_file = f"{output_dir}/sensitivity_{parameter}.csv"
    
    # Check if we're resuming
    file_exists = Path(output_file).exists()
    mode = 'a' if file_exists else 'w'
    
    print(f"\n{'='*80}")
    if resume_rep > 0:
        print(f"RESUMING SENSITIVITY ANALYSIS: CEILING vs MAX-LPT - {parameter.upper()}")
    else:
        print(f"SENSITIVITY ANALYSIS: CEILING vs MAX-LPT - {parameter.upper()}")
    print(f"{'='*80}")
    print(f"Parameter: {parameter}")
    print(f"Starting value: {start_value}")
    if resume_rep > 0:
        print(f"Resuming from repetition: {resume_rep}")
    print(f"Step size: {step}")
    print(f"Repetitions per value: {repetitions}")
    print(f"Results directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Stopping criteria:")
    if parameter == 'machines':
        print(f"  - First timeout (>{TIMEOUT_THRESHOLD}s) on EITHER bound OR")
        print(f"  - Machines > {MAX_MACHINES}")
    else:
        print(f"  - First timeout (>{TIMEOUT_THRESHOLD}s) on EITHER bound")
    print(f"Output file: {output_file}")
    print(f"{'='*80}\n")
    
    # CSV fieldnames
    fieldnames = [
        'parameter', 'value', 'repetition', 'seed',
        'n_jobs', 'm_machines', 'budget_multiplier',

        'ceiling_runtime', 'ceiling_nodes', 'ceiling_pruned_bound',
        'ceiling_pruning_rate', 'ceiling_makespan', 'ceiling_status', 'ceiling_leaf_timeout', 'ceiling_leaf_nodes',

        'maxlpt_runtime', 'maxlpt_nodes', 'maxlpt_pruned_bound',
        'maxlpt_pruning_rate', 'maxlpt_makespan', 'maxlpt_status', 'maxlpt_leaf_timeout', 'maxlpt_leaf_nodes',

        'match_makespan', 'node_reduction_pct', 'speedup',
        'root_solved_skipped',
        'ceiling_timeout', 'maxlpt_timeout'
    ]
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        completed = 0
        value = start_value
        should_stop = False

        while not should_stop:
            print(f"\n--- Testing {parameter} = {value} ---")

            # Set parameters
            if parameter == 'jobs':
                n_jobs = int(value)
                m_machines = BASELINE['m_machines']
                budget_multiplier = BASELINE['budget_multiplier']
            elif parameter == 'machines':
                n_jobs = BASELINE['n_jobs']
                m_machines = int(value)
                budget_multiplier = BASELINE['budget_multiplier']
                if m_machines > MAX_MACHINES:
                    print(f"\n*** Reached machine limit ({MAX_MACHINES}). Stopping. ***")
                    break
            elif parameter == 'budget':
                n_jobs = BASELINE['n_jobs']
                m_machines = BASELINE['m_machines']
                budget_multiplier = value
            else:
                raise ValueError(f"Unknown parameter: {parameter}")

            # Try up to 25 instances to get one valid repetition
            max_attempts = 25
            attempt = 0
            found_valid = False
            root_solved_count = 0
            while attempt < max_attempts and not found_valid:
                seed = 1000 * int(value * 10) + attempt
                print(f"  Attempt {attempt+1}/{max_attempts} (seed={seed})...", end=' ', flush=True)

                metrics = run_comparison_test(
                    n_jobs=n_jobs,
                    m_machines=m_machines,
                    budget_multiplier=budget_multiplier,
                    seed=seed,
                    max_nodes=BASELINE['max_nodes'],
                    time_limit=BASELINE['time_limit'],
                    log_dir=None
                )

                # Check if solved at root node (nodes_explored == 0 for both bounds)
                if metrics['ceiling_nodes'] == 0 and metrics['maxlpt_nodes'] == 0:
                    root_solved_count += 1
                    print("Solved at root node. Skipping.")
                    attempt += 1
                    continue

                # If either bound explores at least one node, record as valid
                if metrics['ceiling_nodes'] > 0 or metrics['maxlpt_nodes'] > 0:
                    found_valid = True
                    metrics['parameter'] = parameter
                    metrics['value'] = value
                    metrics['repetition'] = 0
                    metrics['root_solved_skipped'] = root_solved_count
                    metrics['ceiling_timeout'] = metrics['ceiling_runtime'] >= TIMEOUT_THRESHOLD or metrics.get('ceiling_leaf_timeout', False)
                    metrics['maxlpt_timeout'] = metrics['maxlpt_runtime'] >= TIMEOUT_THRESHOLD or metrics.get('maxlpt_leaf_timeout', False)
                    writer.writerow(metrics)
                    csvfile.flush()
                    completed += 1
                    c_status = metrics['ceiling_status']
                    m_status = metrics['maxlpt_status']
                    c_nodes = metrics['ceiling_nodes']
                    m_nodes = metrics['maxlpt_nodes']
                    node_red = metrics.get('node_reduction_pct')
                    timeout_info = f"C_timeout:{metrics['ceiling_timeout']} | M_timeout:{metrics['maxlpt_timeout']}"
                    if node_red is not None:
                        print(f"Done! C:{c_status} ({c_nodes}n) | M:{m_status} ({m_nodes}n) | Red:{node_red:.1f}% | Skipped:{root_solved_count} | {timeout_info}")
                    else:
                        print(f"Done! C:{c_status} ({c_nodes}n) | M:{m_status} ({m_nodes}n) | Skipped:{root_solved_count} | {timeout_info}")
                else:
                    print("Timeout or invalid. Skipping.")
                    attempt += 1

            # If not found, print info
            if not found_valid:
                print(f"No valid instance found for {parameter}={value} after {max_attempts} attempts. Skipped {root_solved_count} root-solved instances.")

            # Stopping criteria: if either bound hit timeout or leaf timeout in the valid instance, stop
            if found_valid:
                if (metrics['ceiling_runtime'] >= TIMEOUT_THRESHOLD or metrics.get('ceiling_leaf_timeout', False)) and \
                   (metrics['maxlpt_runtime'] >= TIMEOUT_THRESHOLD or metrics.get('maxlpt_leaf_timeout', False)):
                    print(f"\n*** Timeout or leaf timeout reached for both bounds at {parameter}={value}. Stopping. ***")
                    should_stop = True
            # Next parameter value
            if parameter in ['jobs', 'machines']:
                value = int(value + step)
            else:
                value = round(value + step, 2)
    
    print(f"\n{'='*80}")
    print(f"Sensitivity analysis complete!")
    print(f"Total comparisons: {completed}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Sensitivity analysis comparing ceiling vs Max-LPT bounds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_sensitivity_maxlpt.py --parameter jobs --start 4 --step 2 --repetitions 10
  python test_sensitivity_maxlpt.py --parameter machines --start 2 --step 1 --repetitions 10
  python test_sensitivity_maxlpt.py --parameter budget --start 1.2 --step 0.2 --repetitions 10
        """
    )
    
    parser.add_argument('--parameter', type=str, required=True,
                       choices=['jobs', 'machines', 'budget'],
                       help='Parameter to vary')
    parser.add_argument('--start', type=float, required=True,
                       help='Starting value')
    parser.add_argument('--step', type=float, required=True,
                       help='Step size')
    parser.add_argument('--repetitions', type=int, default=10,
                       help='Repetitions per value (default: 10)')
    parser.add_argument('--output-dir', type=str, default='results/sensitivity_maxlpt',
                       help='Output directory (default: results/sensitivity_maxlpt)')
    parser.add_argument('--resume-rep', type=int, default=0,
                       help='Resume from this repetition (0 = start fresh, >0 = resume)')
    parser.add_argument('--resume-timestamp', type=str, default=None,
                       help='Use existing timestamp folder instead of creating new one (for resuming)')
    
    args = parser.parse_args()
    
    # Create timestamp for this run or use existing one for resume
    if args.resume_timestamp:
        timestamp = args.resume_timestamp
        print(f"Resuming analysis with timestamp: {timestamp}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create timestamped directories for this sensitivity run
    base_output_dir = args.output_dir if args.output_dir else 'results/sensitivity_maxlpt'
    output_dir = f"{base_output_dir}_{args.parameter}_{timestamp}"
    run_sensitivity_analysis(
        parameter=args.parameter,
        start_value=args.start,
        step=args.step,
        repetitions=args.repetitions,
        output_dir=output_dir,
        log_dir=None,
        resume_rep=args.resume_rep
    )
    
    # Output file path for user reference
    output_file = f"{output_dir}/sensitivity_{args.parameter}.csv"
    
    print("\nAnalyze results with pandas:")
    print(f"  df = pd.read_csv('{output_file}')")
    print(f"  df.groupby('value')[['node_reduction_pct', 'speedup']].mean()")


if __name__ == '__main__':
    main()
