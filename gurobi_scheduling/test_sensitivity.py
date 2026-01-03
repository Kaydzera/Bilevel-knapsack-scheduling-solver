"""
Sensitivity Analysis for BnB Algorithm Performance

This script tests how BnB performance changes when varying a single parameter:
- Number of job types (n)
- Number of machines (m)  
- Budget multiplier (relative to minimum cost)

For each parameter, we:
1. Fix all other parameters at baseline values
2. Vary the target parameter across a range
3. Run BnB multiple times (with different random seeds) for statistical reliability
4. Collect performance metrics: runtime, nodes, pruning rate, solution quality
5. Save results to CSV for analysis/plotting

Usage:
    python test_sensitivity.py --parameter jobs --min 4 --max 16 --step 2 --repetitions 10
    python test_sensitivity.py --parameter machines --min 2 --max 10 --step 1 --repetitions 10
    python test_sensitivity.py --parameter budget --min 1.2 --max 4.0 --step 0.2 --repetitions 10
"""

import argparse
import csv
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import random

from models import MainProblem
from bnb import run_bnb_classic
from logger import create_logger


# Baseline configuration (used when parameter is not being varied)
BASELINE = {
    'n_jobs': 8,           # Number of job types
    'm_machines': 4,       # Number of machines
    'budget_multiplier': 2.0,  # Budget as multiple of minimum cost
    'max_nodes': 500000,   # BnB node limit (increased from 100k)
    'time_limit': 300.0,   # 5 minutes per instance
}

# Stopping criteria
MAX_MACHINES = 500  # Hard limit on number of machines
TIMEOUT_THRESHOLD = 300.0  # Stop increasing parameter if we hit this runtime


def generate_instance(n_jobs: int, m_machines: int, budget_multiplier: float, seed: int) -> MainProblem:
    """
    Generate a test instance with specified parameters.
    
    Uses uniform random generation for durations and prices.
    Budget is set as a multiple of the minimum total cost.
    """
    random.seed(seed)
    
    # Generate durations: uniform between 5 and 20
    durations = [random.randint(5, 20) for _ in range(n_jobs)]
    
    # Generate prices: uniform between 2 and 15
    prices = [random.randint(2, 15) for _ in range(n_jobs)]
    
    # Calculate budget as multiple of minimum total cost
    min_cost = sum(prices)
    budget = int(budget_multiplier * min_cost)
    
    return MainProblem(
        prices=prices,
        durations=durations,
        anzahl_maschinen=m_machines,
        budget_total=budget
    )


def run_single_test(n_jobs: int, m_machines: int, budget_multiplier: float, 
                   seed: int, max_nodes: int, time_limit: float, log_dir: str = "logs/sensitivity") -> Dict:
    """
    Run BnB on a single instance and collect performance metrics.
    
    Returns:
        Dictionary with metrics: runtime, nodes_explored, pruned_bound, 
        pruning_rate, best_makespan, status
    """
    # Generate instance
    problem = generate_instance(n_jobs, m_machines, budget_multiplier, seed)
    
    # Create logger (suppress output during test) 
    instance_name = f"sensitivity_{n_jobs}j_{m_machines}m_{budget_multiplier}b_s{seed}"
    logger = create_logger(instance_name=instance_name, log_dir=log_dir)
    
    # Run BnB
    start_time = time.time()
    try:
        result = run_bnb_classic(
            problem=problem,
            max_nodes=max_nodes,
            logger=logger,
            instance_name=instance_name,
            verbose=False
        )
        runtime = time.time() - start_time
        status = 'success'
        
        # Read metrics from the logger's saved JSON file
        metrics_file = Path(log_dir) / f"{logger.run_id}_metrics.json"
        with open(metrics_file, 'r') as f:
            logger_metrics = json.load(f)
        
        # Extract only bound_dominated pruning (the algorithmic contribution)
        pruning_reasons = logger_metrics.get('pruning_reasons', {})
        pruned_bound = pruning_reasons.get('bound_dominated', 0)
        
        # Extract metrics
        metrics = {
            'runtime': runtime,
            'nodes_explored': result.get('nodes_explored', 0),
            'pruned_bound': pruned_bound,
            'best_makespan': result.get('best_obj', float('inf')),
            'status': status,
            'n_jobs': n_jobs,
            'm_machines': m_machines,
            'budget_multiplier': budget_multiplier,
            'seed': seed,
        }
        
        # Calculate pruning rate (only bound-dominated pruning)
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
            'n_jobs': n_jobs,
            'm_machines': m_machines,
            'budget_multiplier': budget_multiplier,
            'seed': seed,
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
            'n_jobs': n_jobs,
            'm_machines': m_machines,
            'budget_multiplier': budget_multiplier,
            'seed': seed,
        }
    
    return metrics


def run_sensitivity_analysis(parameter: str, start_value: float, step: float,
                            repetitions: int, output_file: str):
    """
    Run sensitivity analysis by varying a single parameter.
    
    Continues increasing the parameter until:
    - Timeout threshold is reached (5 minutes), OR
    - For machines: MAX_MACHINES limit is reached
    
    Args:
        parameter: Which parameter to vary ('jobs', 'machines', or 'budget')
        start_value: Starting value for the parameter
        step: Increment step for the parameter
        repetitions: Number of random instances to test per value
        output_file: CSV file path to save results
    """
    # Set up parameter-specific log directory
    log_dir = f"logs/sensitivity_{parameter}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"SENSITIVITY ANALYSIS: {parameter.upper()}")
    print(f"{'='*70}")
    print(f"Parameter: {parameter}")
    print(f"Starting value: {start_value}")
    print(f"Step size: {step}")
    print(f"Repetitions per value: {repetitions}")
    print(f"Log directory: {log_dir}")
    print(f"Stopping criteria:")
    if parameter == 'machines':
        print(f"  - First timeout (>{TIMEOUT_THRESHOLD}s) OR")
        print(f"  - Machines > {MAX_MACHINES}")
    else:
        print(f"  - First timeout (>{TIMEOUT_THRESHOLD}s)")
    print(f"Output file: {output_file}")
    print(f"{'='*70}\n")
    
    # Prepare CSV file
    fieldnames = [
        'parameter', 'value', 'repetition', 'seed',
        'n_jobs', 'm_machines', 'budget_multiplier',
        'runtime', 'nodes_explored', 'pruned_bound', 
        'pruning_rate', 'best_makespan', 'status'
    ]
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        completed = 0
        value = start_value
        should_stop = False
        
        # Iterate through parameter values until stopping criteria met
        while not should_stop:
            print(f"\n--- Testing {parameter} = {value} ---")
            
            # Set parameters based on which one we're varying
            if parameter == 'jobs':
                n_jobs = int(value)
                m_machines = BASELINE['m_machines']
                budget_multiplier = BASELINE['budget_multiplier']
            elif parameter == 'machines':
                n_jobs = BASELINE['n_jobs']
                m_machines = int(value)
                budget_multiplier = BASELINE['budget_multiplier']
                # Check machine limit
                if m_machines > MAX_MACHINES:
                    print(f"\n*** Reached machine limit ({MAX_MACHINES}). Stopping. ***")
                    break
            elif parameter == 'budget':
                n_jobs = BASELINE['n_jobs']
                m_machines = BASELINE['m_machines']
                budget_multiplier = value
            else:
                raise ValueError(f"Unknown parameter: {parameter}")
            
            # Track if any repetition hit timeout for this value
            hit_timeout = False
            
            # Run multiple repetitions with different seeds
            for rep in range(repetitions):
                seed = 1000 * int(value * 10) + rep  # Deterministic seed based on value and rep
                
                print(f"  Repetition {rep+1}/{repetitions} (seed={seed})...", end=' ', flush=True)
                
                metrics = run_single_test(
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
                
                # Write to CSV
                writer.writerow(metrics)
                csvfile.flush()
                
                completed += 1
                print(f"Done! ({metrics['status']}, {metrics['runtime']:.2f}s, {metrics['nodes_explored']} nodes)")
                
                # Check if we hit timeout threshold
                if metrics['runtime'] >= TIMEOUT_THRESHOLD:
                    hit_timeout = True
            
            # Check stopping criteria after completing all repetitions for this value
            if hit_timeout:
                print(f"\n*** First timeout reached at {parameter}={value}. Stopping. ***")
                should_stop = True
            else:
                # Increment for next iteration
                if parameter in ['jobs', 'machines']:
                    value = int(value + step)
                else:
                    value = round(value + step, 2)
    
    print(f"\n{'='*70}")
    print(f"Sensitivity analysis complete!")
    print(f"Total tests completed: {completed}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run sensitivity analysis for BnB algorithm performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Vary number of jobs starting from 4, incrementing by 2 until timeout
  python test_sensitivity.py --parameter jobs --start 4 --step 2 --repetitions 10
  
  # Vary number of machines starting from 2, incrementing by 1 until timeout or 500 machines
  python test_sensitivity.py --parameter machines --start 2 --step 1 --repetitions 10
  
  # Vary budget multiplier starting from 1.2, incrementing by 0.2 until timeout
  python test_sensitivity.py --parameter budget --start 1.2 --step 0.2 --repetitions 10
  
  # Run all three analyses
  python test_sensitivity.py --parameter jobs --start 4 --step 2 --repetitions 10
  python test_sensitivity.py --parameter machines --start 2 --step 1 --repetitions 10
  python test_sensitivity.py --parameter budget --start 1.2 --step 0.2 --repetitions 10
  
Note: Script automatically stops when first timeout (5min) is reached or 500 machines limit
        """
    )
    
    parser.add_argument('--parameter', type=str, required=True,
                       choices=['jobs', 'machines', 'budget'],
                       help='Parameter to vary (jobs, machines, or budget)')
    parser.add_argument('--start', type=float, required=True,
                       help='Starting value for the parameter')
    parser.add_argument('--step', type=float, required=True,
                       help='Step size to increment the parameter')
    parser.add_argument('--repetitions', type=int, default=10,
                       help='Number of random instances per value (default: 10)')
    parser.add_argument('--output-dir', type=str, default='results/sensitivity',
                       help='Directory to save results (default: results/sensitivity)')
    
    args = parser.parse_args()
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{args.output_dir}/sensitivity_{args.parameter}_{timestamp}.csv"
    
    # Run the analysis
    run_sensitivity_analysis(
        parameter=args.parameter,
        start_value=args.start,
        step=args.step,
        repetitions=args.repetitions,
        output_file=output_file
    )
    
    print("\nYou can now analyze the results using pandas:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_csv('{output_file}')")
    print(f"  df.groupby('value')[['runtime', 'nodes_explored', 'pruning_rate']].mean()")


if __name__ == '__main__':
    main()
