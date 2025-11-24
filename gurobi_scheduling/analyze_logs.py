"""Utility script to analyze log files from BnB runs.

Usage:
    python analyze_logs.py <metrics_file.json>
    python analyze_logs.py logs/sample_bilevel_20251124_104713_metrics.json
"""

import json
import sys
from pathlib import Path


def analyze_metrics(metrics_file):
    """Analyze and summarize metrics from a BnB run."""
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print("=" * 70)
    print(f"ANALYSIS: {metrics['instance_name']}")
    print(f"Run ID: {metrics['timestamp']}")
    print("=" * 70)
    
    # Basic statistics
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Total runtime: {metrics['total_runtime']:.3f} seconds")
    print(f"Nodes explored: {metrics['nodes_explored']:,}")
    print(f"Nodes pruned: {metrics['nodes_pruned']:,}")
    print(f"Nodes evaluated (leaves): {metrics['nodes_evaluated']:,}")
    
    if metrics['nodes_explored'] > 0:
        prune_rate = 100 * metrics['nodes_pruned'] / metrics['nodes_explored']
        eval_rate = 100 * metrics['nodes_evaluated'] / metrics['nodes_explored']
        print(f"Pruning rate: {prune_rate:.2f}%")
        print(f"Evaluation rate: {eval_rate:.2f}%")
        print(f"Nodes per second: {metrics['nodes_explored'] / metrics['total_runtime']:.1f}")
    
    # Problem characteristics
    if 'problem_data' in metrics:
        print("\n--- PROBLEM CHARACTERISTICS ---")
        prob = metrics['problem_data']
        print(f"Job types: {prob['n_job_types']}")
        print(f"Machines: {prob['machines']}")
        print(f"Budget: {prob['budget']}")
        print(f"Prices: {prob['prices']}")
        print(f"Durations: {prob['durations']}")
    
    # Solution quality progression
    if metrics['best_bound_updates']:
        print("\n--- SOLUTION PROGRESSION ---")
        for i, update in enumerate(metrics['best_bound_updates']):
            print(f"Update {i+1}: makespan={update['incumbent']:.1f} "
                  f"at node {update['node_count']} "
                  f"({update['timestamp']:.3f}s)")
        
        final = metrics['best_bound_updates'][-1]
        print(f"\nFinal solution: makespan={final['incumbent']:.1f}")
        print(f"Selection: {final['selection']}")
    
    # Bound computation statistics
    if metrics['bound_computations']:
        print("\n--- BOUND COMPUTATION STATISTICS ---")
        bounds = metrics['bound_computations']
        total_bound_time = sum(b['time'] for b in bounds if b['time'])
        avg_bound_time = total_bound_time / len(bounds) if bounds else 0
        
        print(f"Total bounds computed: {len(bounds):,}")
        print(f"Total time in bounds: {total_bound_time:.3f}s "
              f"({100*total_bound_time/metrics['total_runtime']:.1f}% of runtime)")
        print(f"Average bound time: {avg_bound_time*1000:.2f}ms")
        
        # Bound quality by depth
        depth_bounds = {}
        for b in bounds[:100]:  # Sample first 100
            depth = b['depth']
            if depth not in depth_bounds:
                depth_bounds[depth] = []
            depth_bounds[depth].append(b['value'])
        
        print("\nBound values by depth (first 100 bounds):")
        for depth in sorted(depth_bounds.keys()):
            avg_val = sum(depth_bounds[depth]) / len(depth_bounds[depth])
            print(f"  Depth {depth}: avg={avg_val:.1f} (n={len(depth_bounds[depth])})")
    
    # Pruning reasons
    if metrics.get('pruning_reasons'):
        print("\n--- PRUNING REASONS ---")
        for reason, count in metrics['pruning_reasons'].items():
            pct = 100 * count / metrics['nodes_pruned'] if metrics['nodes_pruned'] > 0 else 0
            print(f"{reason}: {count:,} ({pct:.1f}%)")
    
    # Final result
    if 'final_result' in metrics:
        print("\n--- FINAL RESULT ---")
        result = metrics['final_result']
        print(f"Best makespan: {result['best_obj']}")
        print(f"Best selection: {result['best_selection']}")
        if 'best_schedule' in result and result['best_schedule']:
            sched = result['best_schedule']
            if 'machine_assignments' in sched:
                print("\nMachine assignments:")
                for m, jobs in sched['machine_assignments'].items():
                    load = sched['machine_loads'][int(m)]
                    print(f"  Machine {m}: {len(jobs)} jobs, load={load:.1f}")
    
    print("\n" + "=" * 70)


def compare_runs(metrics_files):
    """Compare multiple BnB runs."""
    
    runs = []
    for file in metrics_files:
        with open(file, 'r') as f:
            runs.append(json.load(f))
    
    print("=" * 70)
    print(f"COMPARING {len(runs)} RUNS")
    print("=" * 70)
    
    print(f"\n{'Instance':<30} {'Runtime':>10} {'Nodes':>10} {'Prune%':>8} {'Makespan':>10}")
    print("-" * 70)
    
    for run in runs:
        name = run['instance_name'][:28]
        runtime = run['total_runtime']
        nodes = run['nodes_explored']
        prune_pct = 100 * run['nodes_pruned'] / nodes if nodes > 0 else 0
        
        if run['best_bound_updates']:
            makespan = run['best_bound_updates'][-1]['incumbent']
        else:
            makespan = "N/A"
        
        print(f"{name:<30} {runtime:>9.2f}s {nodes:>10,} {prune_pct:>7.1f}% {makespan:>10}")
    
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <metrics_file.json> [<more_files>...]")
        print("\nExample:")
        print("  python analyze_logs.py logs/sample_bilevel_20251124_104713_metrics.json")
        sys.exit(1)
    
    files = [Path(f) for f in sys.argv[1:]]
    
    # Check all files exist
    for f in files:
        if not f.exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)
    
    if len(files) == 1:
        # Analyze single run
        analyze_metrics(files[0])
    else:
        # Compare multiple runs
        compare_runs(files)
