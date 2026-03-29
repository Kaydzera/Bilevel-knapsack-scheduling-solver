"""
Grid Sensitivity Analysis: Testing combined effect of job types and machines
with proportional budget scaling.

This script tests the solver performance across a grid of:
- m_machines: 2 to 12, step by 2 (6 values: 2, 4, 6, 8, 10, 12)
- n_jobs: 4 to 24, step by 4 (6 values: 4, 8, 12, 16, 20, 24)
- budget_multiplier: 1.3 (default, configurable via --budget-multiplier)
- 5 repetitions per configuration

Total: 6 × 6 × 1 × 5 = 180 tests per budget level

Budget scales proportionally with machines:
  budget = (sum(prices) / n_jobs) × m_machines × multiplier
So 10% more machines → 10% more budget.
"""

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from bnb import run_bnb_classic, LeafTimeoutError
from models import MainProblem


def generate_random_instance(n_jobs, m_machines, budget_multiplier, seed):
    """
    Generate a random problem instance with specified parameters.
    Budget scales proportionally with number of machines.
    
    Args:
        n_jobs: Number of job types
        m_machines: Number of machines
        budget_multiplier: Multiplier for budget calculation
        seed: Random seed for reproducibility
    
    Returns:
        MainProblem instance
    """
    random.seed(seed)
    
    # Generate random prices and durations
    prices = [random.randint(10, 100) for _ in range(n_jobs)]
    durations = [random.randint(5, 50) for _ in range(n_jobs)]
    
    # Calculate budget: scales proportionally with machines
    base_cost = sum(prices) / n_jobs  # Average price per job type
    budget = base_cost * m_machines * budget_multiplier
    
    return MainProblem(
        prices=prices,
        durations=durations,
        anzahl_maschinen=m_machines,
        budget_total=budget
    )


def run_grid_sensitivity(
    output_dir="results/sensitivity_grid",
    machines_range=(2, 12, 2),  # start, stop (inclusive), step
    jobs_range=(4, 24, 4),  # start, stop (inclusive), step
    budget_multipliers=(1.3,),  # Single budget per run (comma needed for tuple)
    repetitions=5,
    time_limit=3600.0,
    node_limit=10000000,
    resume_machines=None,
    resume_jobs=None,
    resume_multiplier=None,
    resume_rep=None,
    resume_timestamp=None
):
    """
    Run grid sensitivity analysis varying machines and jobs independently.
    
    Args:
        output_dir: Base directory for results
        machines_range: (start, stop, step) tuple for m_machines (stop inclusive)
        jobs_range: (start, stop, step) for n_jobs (stop inclusive)
        budget_multipliers: Tuple of budget multipliers to test
        repetitions: Number of repetitions per configuration
        time_limit: Time limit in seconds per test
        node_limit: Node limit per bound type
        resume_*: Parameters for resuming interrupted runs
        resume_timestamp: Timestamp of previous run to resume
    """
    # Create output directory with timestamp
    if resume_timestamp:
        timestamp = resume_timestamp
        # Handle case where timestamp already includes "grid_" prefix
        if timestamp.startswith("grid_"):
            output_path = Path(output_dir) / timestamp
        else:
            output_path = Path(output_dir) / f"grid_{timestamp}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"grid_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_path / "sensitivity_grid.csv"
    
    # Determine if we're resuming or starting fresh
    file_exists = csv_file.exists()
    mode = 'a' if file_exists else 'w'
    
    # Load already completed tests to avoid duplicates
    completed_tests = set()
    if file_exists:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Track (m_machines, n_jobs, budget_multiplier, repetition) tuples
                key = (int(row['m_machines']), int(row['n_jobs']), 
                       float(row['budget_multiplier']), int(row['repetition']))
                completed_tests.add(key)
        print(f"Loaded {len(completed_tests)} already completed tests from CSV\n")
    
    # Generate parameter ranges
    machines_values = list(range(machines_range[0], machines_range[1] + 1, machines_range[2]))
    jobs_values = list(range(jobs_range[0], jobs_range[1] + 1, jobs_range[2]))
    
    print(f"\n{'='*80}")
    print(f"GRID SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    # Show clean timestamp (without grid_ prefix) for display
    display_timestamp = timestamp.replace("grid_", "") if timestamp.startswith("grid_") else timestamp
    print(f"Timestamp: {display_timestamp}")
    print(f"Output: {output_path}")
    print(f"\nParameters:")
    print(f"  Machines: {machines_values[0]} to {machines_values[-1]} ({len(machines_values)} values)")
    print(f"  Job types: {jobs_values[0]} to {jobs_values[-1]} ({len(jobs_values)} values)")
    print(f"  Budget multipliers: {budget_multipliers}")
    print(f"  Repetitions: {repetitions}")
    print(f"  Time limit: {time_limit}s")
    print(f"  Node limit: {node_limit:,}")
    print(f"\nTotal tests: {len(machines_values)} x {len(jobs_values)} x {len(budget_multipliers)} x {repetitions}")
    print(f"            = {len(machines_values) * len(jobs_values) * len(budget_multipliers) * repetitions:,} tests")
    print(f"{'='*80}\n")
    
    if resume_machines is not None:
        print(f"RESUMING from: m={resume_machines}, n_jobs={resume_jobs}, "
              f"multiplier={resume_multiplier}, rep={resume_rep}\n")
    
    # Open CSV file
    with open(csv_file, mode, newline='') as f:
        writer = csv.writer(f)
        
        # Write header only for new files
        if not file_exists:
            writer.writerow([
                'timestamp', 'm_machines', 'n_jobs', 'budget_multiplier', 
                'budget', 'repetition', 'seed',
                'ceiling_status', 'ceiling_nodes', 'ceiling_time',
                'ceiling_initial', 'ceiling_final', 'ceiling_improvement',
                'maxlpt_status', 'maxlpt_nodes', 'maxlpt_time',
                'maxlpt_initial', 'maxlpt_final', 'maxlpt_improvement'
            ])
        
        test_count = 0
        start_time = time.time()
        
        # Grid iteration
        for budget_mult in budget_multipliers:
            for m_machines in machines_values:
                for n_jobs in jobs_values:
                    # Handle resume logic
                    start_rep = 0
                    if (resume_machines is not None and 
                        (budget_mult < resume_multiplier or
                         (budget_mult == resume_multiplier and m_machines < resume_machines) or
                         (budget_mult == resume_multiplier and m_machines == resume_machines and n_jobs < resume_jobs))):
                        continue  # Skip completed configurations
                    elif (resume_machines is not None and 
                          budget_mult == resume_multiplier and 
                          m_machines == resume_machines and 
                          n_jobs == resume_jobs):
                        start_rep = resume_rep  # Resume from this repetition
                    
                    for rep in range(start_rep, repetitions):
                        # Skip if already completed (check CSV)
                        test_key = (m_machines, n_jobs, budget_mult, rep + 1)
                        if test_key in completed_tests:
                            continue
                        
                        test_count += 1
                        elapsed = time.time() - start_time
                        
                        print(f"\n[Test {test_count}] "
                              f"Multiplier={budget_mult:.1f}, "
                              f"Machines={m_machines}, "
                              f"Jobs={n_jobs}, "
                              f"Rep={rep+1}/{repetitions}")
                        print(f"Elapsed time: {elapsed/3600:.2f}h")
                        
                        # Generate instance
                        seed = hash((n_jobs, m_machines, budget_mult, rep)) % (2**32)
                        problem = generate_random_instance(
                            n_jobs=n_jobs,
                            m_machines=m_machines,
                            budget_multiplier=budget_mult,
                            seed=seed
                        )
                        
                        print(f"Budget: {problem.budget_total:.2f}")
                        print(f"Prices: {problem.prices}")
                        print(f"Durations: {problem.durations}")

                        # Initialize default values so we can always persist progress,
                        # even if one solver times out/errors.
                        ceil_status = 'not_run'
                        ceil_nodes = 0
                        ceil_time = 0.0
                        ceil_initial = None
                        ceil_final = None
                        ceil_improvement = 0.0

                        maxlpt_status = 'not_run'
                        maxlpt_nodes = 0
                        maxlpt_time = 0.0
                        maxlpt_initial = None
                        maxlpt_final = None
                        maxlpt_improvement = 0.0
                        
                        # Test with ceiling bound
                        print(f"  [Ceiling] Running...")
                        ceil_start = time.time()
                        try:
                            result_ceil = run_bnb_classic(
                                problem=problem,
                                max_nodes=node_limit,
                                verbose=False,
                                logger=None,
                                enable_logging=False,
                                bound_type='ceiling',
                                time_limit=time_limit,
                                leaf_timeout=900.0  # 15 minutes
                            )
                            ceil_time = time.time() - ceil_start
                            
                            ceil_nodes = result_ceil.get('nodes_explored', 0)
                            ceil_initial = result_ceil.get('initial_makespan', None)
                            ceil_final = result_ceil.get('best_obj', None)
                            ceil_improvement = (ceil_initial - ceil_final) if (ceil_initial and ceil_final) else 0
                            
                            # Determine status
                            if result_ceil.get('proven_optimal', False):
                                ceil_status = 'optimal'
                            elif ceil_nodes >= node_limit:
                                ceil_status = 'node_limit'
                            elif ceil_time >= time_limit:
                                ceil_status = 'timeout'
                            else:
                                ceil_status = 'completed'
                                
                        except LeafTimeoutError as e:
                            # Persist partial result instead of skipping this test.
                            ceil_time = time.time() - ceil_start
                            ceil_status = 'leaf_timeout'
                            print(f"  [Ceiling] LEAF TIMEOUT - {str(e)}")

                        except Exception as e:
                            ceil_time = time.time() - ceil_start
                            ceil_status = f"error:{type(e).__name__}"
                            print(f"  [Ceiling] ERROR - {e}")
                        
                        # Format output safely
                        ceil_initial_str = f"{ceil_initial:.2f}" if ceil_initial is not None else "N/A"
                        ceil_final_str = f"{ceil_final:.2f}" if ceil_final is not None else "N/A"
                        
                        print(f"  [Ceiling] {ceil_status} | Nodes: {ceil_nodes:,} | "
                            f"Time: {ceil_time:.2f}s | Initial: {ceil_initial_str} -> Final: {ceil_final_str} "
                            f"(delta={ceil_improvement:.2f})")
                        
                        # Test with MaxLPT bound even if Ceiling hit leaf timeout.
                        print(f"  [MaxLPT] Running...")
                        maxlpt_start = time.time()
                        try:
                            result_maxlpt = run_bnb_classic(
                                problem=problem,
                                max_nodes=node_limit,
                                verbose=False,
                                logger=None,
                                enable_logging=False,
                                bound_type='maxlpt',
                                time_limit=time_limit,
                                leaf_timeout=900.0  # 15 minutes
                            )
                            maxlpt_time = time.time() - maxlpt_start

                            maxlpt_nodes = result_maxlpt.get('nodes_explored', 0)
                            maxlpt_initial = result_maxlpt.get('initial_makespan', None)
                            maxlpt_final = result_maxlpt.get('best_obj', None)
                            maxlpt_improvement = (maxlpt_initial - maxlpt_final) if (maxlpt_initial and maxlpt_final) else 0

                            # Determine status
                            if result_maxlpt.get('proven_optimal', False):
                                maxlpt_status = 'optimal'
                            elif maxlpt_nodes >= node_limit:
                                maxlpt_status = 'node_limit'
                            elif maxlpt_time >= time_limit:
                                maxlpt_status = 'timeout'
                            else:
                                maxlpt_status = 'completed'

                        except LeafTimeoutError as e:
                            maxlpt_time = time.time() - maxlpt_start
                            maxlpt_nodes = 0
                            maxlpt_initial = ceil_initial  # Use ceiling initial as fallback
                            maxlpt_final = ceil_initial
                            maxlpt_improvement = 0
                            maxlpt_status = 'leaf_timeout'
                            print(f"  [MaxLPT] LEAF TIMEOUT - {str(e)}")

                        except Exception as e:
                            maxlpt_time = time.time() - maxlpt_start
                            maxlpt_status = f"error:{type(e).__name__}"
                            maxlpt_initial = ceil_initial
                            maxlpt_final = ceil_final
                            print(f"  [MaxLPT] ERROR - {e}")
                        
                        # Format output safely
                        maxlpt_initial_str = f"{maxlpt_initial:.2f}" if maxlpt_initial is not None else "N/A"
                        maxlpt_final_str = f"{maxlpt_final:.2f}" if maxlpt_final is not None else "N/A"
                        
                        print(f"  [MaxLPT] {maxlpt_status} | Nodes: {maxlpt_nodes:,} | "
                            f"Time: {maxlpt_time:.2f}s | Initial: {maxlpt_initial_str} -> Final: {maxlpt_final_str} "
                            f"(delta={maxlpt_improvement:.2f})")
                        
                        # Write result to CSV
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            m_machines, n_jobs, budget_mult, 
                            f"{problem.budget_total:.2f}", rep + 1, seed,
                            ceil_status, ceil_nodes, f"{ceil_time:.4f}",
                            ceil_initial, ceil_final, f"{ceil_improvement:.4f}",
                            maxlpt_status, maxlpt_nodes, f"{maxlpt_time:.4f}",
                            maxlpt_initial, maxlpt_final, f"{maxlpt_improvement:.4f}"
                        ])
                        f.flush()  # Ensure data is written immediately
                        
                        # Mark test as completed to avoid re-running in this session
                        completed_tests.add(test_key)
    
    total_elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"GRID SENSITIVITY ANALYSIS COMPLETE")
    print(f"Total tests run: {test_count}")
    print(f"Total time: {total_elapsed/3600:.2f}h")
    print(f"Results saved to: {csv_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run grid sensitivity analysis varying machines and jobs"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/sensitivity_grid',
        help='Output directory for results (default: results/sensitivity_grid)'
    )
    parser.add_argument(
        '--machines-min',
        type=int,
        default=2,
        help='Minimum number of machines (default: 2)'
    )
    parser.add_argument(
        '--machines-max',
        type=int,
        default=12,
        help='Maximum number of machines (default: 12)'
    )
    parser.add_argument(
        '--jobs-min',
        type=int,
        default=4,
        help='Minimum number of job types (default: 4)'
    )
    parser.add_argument(
        '--jobs-max',
        type=int,
        default=24,
        help='Maximum number of job types (default: 24)'
    )
    parser.add_argument(
        '--jobs-step',
        type=int,
        default=4,
        help='Step size for job types (default: 4)'
    )
    parser.add_argument(
        '--machines-step',
        type=int,
        default=2,
        help='Step size for machines (default: 2)'
    )
    parser.add_argument(
        '--budget-multiplier',
        type=float,
        default=1.3,
        help='Budget multiplier (default: 1.3)'
    )
    parser.add_argument(
        '--repetitions',
        type=int,
        default=5,
        help='Number of repetitions per configuration (default: 5)'
    )
    parser.add_argument(
        '--time-limit',
        type=float,
        default=3600.0,
        help='Time limit in seconds per test (default: 3600.0)'
    )
    parser.add_argument(
        '--node-limit',
        type=int,
        default=1000000,
        help='Node limit per bound type (default: 1,000,000)'
    )
    parser.add_argument(
        '--resume-machines',
        type=int,
        help='Resume from this number of machines'
    )
    parser.add_argument(
        '--resume-jobs',
        type=int,
        help='Resume from this number of job types'
    )
    parser.add_argument(
        '--resume-multiplier',
        type=float,
        help='Resume from this budget multiplier'
    )
    parser.add_argument(
        '--resume-rep',
        type=int,
        help='Resume from this repetition (0-indexed)'
    )
    parser.add_argument(
        '--resume-timestamp',
        type=str,
        help='Timestamp of previous run to resume (format: YYYYMMDD_HHMMSS)'
    )
    
    args = parser.parse_args()
    
    # Validate resume parameters
    if args.resume_timestamp:
        if not all([
            args.resume_machines is not None,
            args.resume_jobs is not None,
            args.resume_multiplier is not None,
            args.resume_rep is not None
        ]):
            print("Error: When using --resume-timestamp, you must also provide:")
            print("  --resume-machines, --resume-jobs, --resume-multiplier, --resume-rep")
            sys.exit(1)
    
    run_grid_sensitivity(
        output_dir=args.output_dir,
        machines_range=(args.machines_min, args.machines_max, args.machines_step),
        jobs_range=(args.jobs_min, args.jobs_max, args.jobs_step),
        budget_multipliers=(args.budget_multiplier,),
        repetitions=args.repetitions,
        time_limit=args.time_limit,
        node_limit=args.node_limit,
        resume_machines=args.resume_machines,
        resume_jobs=args.resume_jobs,
        resume_multiplier=args.resume_multiplier,
        resume_rep=args.resume_rep,
        resume_timestamp=args.resume_timestamp
    )


if __name__ == '__main__':
    main()
