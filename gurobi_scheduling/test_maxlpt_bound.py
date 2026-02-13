"""Test Max-LPT upper bound integration with branch-and-bound.

This script tests the Max-LPT bound on small instances and compares
it with the ceiling bound to verify correctness and analyze performance.
"""

from models import MainProblem
from bnb import run_bnb_classic
import time


def test_small_instance():
    """Test both bound types on a small instance."""
    print("=" * 70)
    print("TEST: Small Instance with Both Bound Types")
    print("=" * 70)
    
    # Small instance: 4 jobs, 2 machines, uniform ratios
    durations = [4, 6, 8, 10]
    prices = [2, 3, 4, 5]
    m = 2
    budget = 12
    
    print(f"\nProblem:")
    print(f"  Durations: {durations}")
    print(f"  Prices:    {prices}")
    print(f"  Machines:  {m}")
    print(f"  Budget:    {budget}")
    print(f"  Ratios:    {[d/p for d, p in zip(durations, prices)]}")
    
    problem = MainProblem(prices, durations, m, budget)
    
    # Test 1: Ceiling bound
    print("\n" + "-" * 70)
    print("Running BnB with CEILING bound...")
    print("-" * 70)
    start_time = time.time()
    result_ceiling = run_bnb_classic(
        problem, 
        max_nodes=10000, 
        verbose=False,
        enable_logging=False,
        bound_type='ceiling'
    )
    ceiling_time = time.time() - start_time
    
    print(f"\nResults (Ceiling Bound):")
    print(f"  Best makespan:    {result_ceiling['best_obj']}")
    print(f"  Best selection:   {result_ceiling['best_selection']}")
    print(f"  Nodes explored:   {result_ceiling['nodes_explored']}")
    print(f"  Runtime:          {ceiling_time:.4f}s")
    
    # Test 2: Max-LPT bound
    print("\n" + "-" * 70)
    print("Running BnB with MAX-LPT bound...")
    print("-" * 70)
    start_time = time.time()
    result_maxlpt = run_bnb_classic(
        problem, 
        max_nodes=10000, 
        verbose=False,
        enable_logging=False,
        bound_type='maxlpt'
    )
    maxlpt_time = time.time() - start_time
    
    print(f"\nResults (Max-LPT Bound):")
    print(f"  Best makespan:    {result_maxlpt['best_obj']}")
    print(f"  Best selection:   {result_maxlpt['best_selection']}")
    print(f"  Nodes explored:   {result_maxlpt['nodes_explored']}")
    print(f"  Runtime:          {maxlpt_time:.4f}s")
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    # Check if optimal solutions match
    solutions_match = abs(result_ceiling['best_obj'] - result_maxlpt['best_obj']) < 1e-6
    print(f"Optimal solutions match: {solutions_match}")
    if solutions_match:
        print(f"  [OK] Both methods found optimal makespan: {result_ceiling['best_obj']}")
    else:
        print(f"  [FAIL] MISMATCH!")
        print(f"    Ceiling: {result_ceiling['best_obj']}")
        print(f"    Max-LPT: {result_maxlpt['best_obj']}")
    
    # Compare nodes explored
    print(f"\nNodes explored:")
    print(f"  Ceiling:  {result_ceiling['nodes_explored']}")
    print(f"  Max-LPT:  {result_maxlpt['nodes_explored']}")
    if result_maxlpt['nodes_explored'] < result_ceiling['nodes_explored']:
        reduction = 100 * (1 - result_maxlpt['nodes_explored'] / result_ceiling['nodes_explored'])
        print(f"  --> Max-LPT pruned {reduction:.1f}% more nodes")
    elif result_maxlpt['nodes_explored'] > result_ceiling['nodes_explored']:
        increase = 100 * (result_maxlpt['nodes_explored'] / result_ceiling['nodes_explored'] - 1)
        print(f"  --> Max-LPT explored {increase:.1f}% more nodes")
    else:
        print(f"  --> Same number of nodes explored")
    
    # Compare runtime
    print(f"\nRuntime:")
    print(f"  Ceiling:  {ceiling_time:.4f}s")
    print(f"  Max-LPT:  {maxlpt_time:.4f}s")
    if maxlpt_time < ceiling_time:
        speedup = ceiling_time / maxlpt_time
        print(f"  --> Max-LPT is {speedup:.2f}x faster")
    else:
        slowdown = maxlpt_time / ceiling_time
        print(f"  --> Max-LPT is {slowdown:.2f}x slower")


def test_multiple_instances():
    """Test on multiple instances with varying characteristics."""
    print("\n" + "=" * 70)
    print("TEST: Multiple Instances")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "Tiny (3 items, 2 machines)",
            "durations": [5, 7, 9],
            "prices": [3, 4, 5],
            "m": 2,
            "budget": 10
        },
        {
            "name": "Small uniform (4 items, 2 machines)",
            "durations": [4, 6, 8, 10],
            "prices": [2, 3, 4, 5],
            "m": 2,
            "budget": 12
        },
        {
            "name": "Small skewed (4 items, 3 machines)",
            "durations": [10, 5, 8, 3],
            "prices": [2, 3, 4, 5],
            "m": 3,
            "budget": 15
        },
        {
            "name": "Medium uniform (5 items, 3 machines)",
            "durations": [6, 8, 10, 12, 14],
            "prices": [3, 4, 5, 6, 7],
            "m": 3,
            "budget": 20
        },
        {
            "name": "Medium skewed (5 items, 2 machines)",
            "durations": [20, 5, 18, 3, 12],
            "prices": [4, 6, 9, 2, 7],
            "m": 2,
            "budget": 25
        },
        {
            "name": "Tight budget (6 items, 4 machines)",
            "durations": [7, 9, 11, 6, 8, 10],
            "prices": [5, 6, 7, 4, 5, 6],
            "m": 4,
            "budget": 12
        },
        {
            "name": "Loose budget (6 items, 2 machines)",
            "durations": [3, 5, 7, 9, 11, 13],
            "prices": [1, 2, 2, 3, 3, 4],
            "m": 2,
            "budget": 40
        },
        {
            "name": "Uniform ratio (4 items, 4 machines)",
            "durations": [4, 8, 12, 16],
            "prices": [2, 4, 6, 8],
            "m": 4,
            "budget": 24
        },
        {
            "name": "Single dominant item (5 items, 3 machines)",
            "durations": [50, 6, 7, 8, 9],
            "prices": [5, 6, 7, 8, 9],
            "m": 3,
            "budget": 30
        },
        {
            "name": "Many small items (7 items, 3 machines)",
            "durations": [2, 3, 4, 5, 6, 7, 8],
            "prices": [1, 1, 2, 2, 3, 3, 4],
            "m": 3,
            "budget": 18
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'-' * 70}")
        print(f"Instance: {test_case['name']}")
        print(f"{'-' * 70}")
        
        problem = MainProblem(
            test_case['prices'],
            test_case['durations'],
            test_case['m'],
            test_case['budget']
        )
        
        # Run with ceiling bound
        start = time.time()
        res_ceiling = run_bnb_classic(
            problem, max_nodes=10000, verbose=False,
            enable_logging=False, bound_type='ceiling'
        )
        time_ceiling = time.time() - start
        
        # Run with Max-LPT bound
        start = time.time()
        res_maxlpt = run_bnb_classic(
            problem, max_nodes=10000, verbose=False,
            enable_logging=False, bound_type='maxlpt'
        )
        time_maxlpt = time.time() - start
        
        # Record results
        result = {
            'name': test_case['name'],
            'optimal': res_ceiling['best_obj'],
            'match': abs(res_ceiling['best_obj'] - res_maxlpt['best_obj']) < 1e-6,
            'nodes_ceiling': res_ceiling['nodes_explored'],
            'nodes_maxlpt': res_maxlpt['nodes_explored'],
            'time_ceiling': time_ceiling,
            'time_maxlpt': time_maxlpt
        }
        results.append(result)
        
        print(f"Optimal makespan: {result['optimal']}")
        print(f"Solutions match:  {result['match']}")
        print(f"Nodes (C/M):      {result['nodes_ceiling']} / {result['nodes_maxlpt']}")
        print(f"Time (C/M):       {result['time_ceiling']:.4f}s / {result['time_maxlpt']:.4f}s")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Instance':<30} {'Nodes C':<10} {'Nodes M':<10} {'Time C':<10} {'Time M':<10} {'Match':<8}")
    print("-" * 70)
    for r in results:
        match_str = "[OK]" if r['match'] else "[FAIL]"
        print(f"{r['name']:<30} {r['nodes_ceiling']:<10} {r['nodes_maxlpt']:<10} "
              f"{r['time_ceiling']:<10.4f} {r['time_maxlpt']:<10.4f} {match_str:<8}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING MAX-LPT BOUND INTEGRATION")
    print("=" * 70)
    
    # Test 1: Single small instance with detailed output
    test_small_instance()
    
    # Test 2: Multiple instances
    test_multiple_instances()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
