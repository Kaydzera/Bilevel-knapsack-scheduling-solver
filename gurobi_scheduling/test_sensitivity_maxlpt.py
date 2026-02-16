"""
Sensitivity Analysis: Ceiling vs Max-LPT Bound Comparison

This script tests how the relative performance of ceiling vs Max-LPT bounds
changes when varying problem parameters:
- Number of job types (n)
- Number of machines (m)
- Budget multiplier (relative to minimum cost)

For each parameter value:
1. Generate multiple random instances
2. Run BnB with BOTH ceiling and maxlpt bounds
3. Compare metrics: runtime, nodes explored, pruning effectiveness
4. Save results to CSV for analysis

Usage:
    python test_sensitivity_maxlpt.py --parameter jobs --start 4 --step 2 --repetitions 10
    python test_sensitivity_maxlpt.py --parameter machines --start 2 --step 1 --repetitions 10
    python test_sensitivity_maxlpt.py --parameter budget --start 1.2 --step 0.2 --repetitions 10
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
    'n_jobs': 8,
    'm_machines': 4,
    'budget_multiplier': 2.0,
    'max_nodes': 500000,
    'time_limit': 300.0,
}

# Stopping criteria
MAX_MACHINES = 50
TIMEOUT_THRESHOLD = 300.0


def generate_instance(n_jobs: int, m_machines: int, budget_multiplier: float, seed: int) -> MainProblem:
    """Generate a test instance with specified parameters."""
    random.seed(seed)
    
    durations = [random.randint(5, 20) for _ in range(n_jobs)]
    prices = [random.randint(2, 15) for _ in range(n_jobs)]
    
    min_cost = sum(prices)
    budget = int(budget_multiplier * min_cost)
    
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
    logger = create_logger(
        instance_name=f"{instance_name}_{bound_type}", 
        log_dir=log_dir
    )
    
    start_time = time.time()
    try:
        result = run_bnb_classic(
            problem=problem,
            max_nodes=max_nodes,
            logger=logger,
            instance_name=instance_name,
            bound_type=bound_type,
            verbose=False
        )
        runtime = time.time() - start_time
        status = 'success'
        
        # Read metrics from logger's JSON
        metrics_file = Path(log_dir) / f"{logger.run_id}_metrics.json"
        with open(metrics_file, 'r') as f:
            logger_metrics = json.load(f)
        
        pruning_reasons = logger_metrics.get('pruning_reasons', {})
        pruned_bound = pruning_reasons.get('bound_dominated', 0)
        
        metrics = {
            'runtime': runtime,
            'nodes_explored': result.get('nodes_explored', 0),
            'pruned_bound': pruned_bound,
            'best_makespan': result.get('best_obj', float('inf')),
            'status': status,
        }
        
        total_nodes = metrics['nodes_explored']
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
    
    # Run ceiling bound
    ceiling_metrics = run_single_bound(
        problem, 'ceiling', instance_name, max_nodes, time_limit, log_dir
    )
    
    # Run maxlpt bound
    maxlpt_metrics = run_single_bound(
        problem, 'maxlpt', instance_name, max_nodes, time_limit, log_dir
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
        
        # Max-LPT metrics
        'maxlpt_runtime': maxlpt_metrics['runtime'],
        'maxlpt_nodes': maxlpt_metrics['nodes_explored'],
        'maxlpt_pruned_bound': maxlpt_metrics['pruned_bound'],
        'maxlpt_pruning_rate': maxlpt_metrics['pruning_rate'],
        'maxlpt_makespan': maxlpt_metrics['best_makespan'],
        'maxlpt_status': maxlpt_metrics['status'],
        
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
                            repetitions: int, output_file: str):
    """
    Run sensitivity analysis comparing ceiling vs maxlpt bounds.
    """
    log_dir = f"logs/sensitivity_maxlpt_{parameter}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS: CEILING vs MAX-LPT - {parameter.upper()}")
    print(f"{'='*80}")
    print(f"Parameter: {parameter}")
    print(f"Starting value: {start_value}")
    print(f"Step size: {step}")
    print(f"Repetitions per value: {repetitions}")
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
        'ceiling_pruning_rate', 'ceiling_makespan', 'ceiling_status',
        
        'maxlpt_runtime', 'maxlpt_nodes', 'maxlpt_pruned_bound',
        'maxlpt_pruning_rate', 'maxlpt_makespan', 'maxlpt_status',
        
        'match_makespan', 'node_reduction_pct', 'speedup'
    ]
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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
            
            hit_timeout = False
            
            # Run repetitions
            for rep in range(repetitions):
                seed = 1000 * int(value * 10) + rep
                
                print(f"  Rep {rep+1}/{repetitions} (seed={seed})...", end=' ', flush=True)
                
                metrics = run_comparison_test(
                    n_jobs=n_jobs,
                    m_machines=m_machines,
                    budget_multiplier=budget_multiplier,
                    seed=seed,
                    max_nodes=BASELINE['max_nodes'],
                    time_limit=BASELINE['time_limit'],
                    log_dir=log_dir
                )
                
                # Add metadata
                metrics['parameter'] = parameter
                metrics['value'] = value
                metrics['repetition'] = rep
                
                writer.writerow(metrics)
                csvfile.flush()
                
                completed += 1
                
                # Print summary
                c_status = metrics['ceiling_status']
                m_status = metrics['maxlpt_status']
                c_nodes = metrics['ceiling_nodes']
                m_nodes = metrics['maxlpt_nodes']
                node_red = metrics.get('node_reduction_pct')
                
                if node_red is not None:
                    print(f"Done! C:{c_status} ({c_nodes}n) | M:{m_status} ({m_nodes}n) | Red:{node_red:.1f}%")
                else:
                    print(f"Done! C:{c_status} ({c_nodes}n) | M:{m_status} ({m_nodes}n)")
                
                # Check timeout on EITHER bound
                if (metrics['ceiling_runtime'] >= TIMEOUT_THRESHOLD or 
                    metrics['maxlpt_runtime'] >= TIMEOUT_THRESHOLD):
                    hit_timeout = True
            
            # Check stopping criteria
            if hit_timeout:
                print(f"\n*** First timeout reached at {parameter}={value}. Stopping. ***")
                should_stop = True
            else:
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
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{args.output_dir}/sensitivity_{args.parameter}_{timestamp}.csv"
    
    run_sensitivity_analysis(
        parameter=args.parameter,
        start_value=args.start,
        step=args.step,
        repetitions=args.repetitions,
        output_file=output_file
    )
    
    print("\nAnalyze results with pandas:")
    print(f"  df = pd.read_csv('{output_file}')")
    print(f"  df.groupby('value')[['node_reduction_pct', 'speedup']].mean()")


if __name__ == '__main__':
    main()
