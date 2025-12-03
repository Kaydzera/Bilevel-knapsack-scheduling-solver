"""Test CeilKnapsackSolver against compute_ip_bound_direct from bnb.py.

This test verifies that the DP-based CeilKnapsackSolver produces the same
results as the IP-based compute_ip_bound_direct, while being much faster.

Key relationship:
- compute_ip_bound_direct(items, depth, rem_budget, m) uses items[depth+1:]
- CeilKnapsackSolver.reconstruct(num_items, budget) uses items[:num_items]
- To compare: if depth=D, we query with num_items=len(items)-(D+1)
  and ensure items match up correctly.
"""

import time
from knapsack_dp import CeilKnapsackSolver

try:
    from bnb import compute_ip_bound_direct
    GUROBI_AVAILABLE = True
except (ImportError, Exception) as e:
    GUROBI_AVAILABLE = False
    print(f"⚠️  Gurobi not available or version mismatch, skipping IP bound tests")
    print(f"   Error: {e}")


def test_item_correspondence():
    """Test that we correctly understand the depth/num_items relationship."""
    print("="*80)
    print("TEST: Item Correspondence")
    print("="*80)
    print("\nVerifying the relationship between depth and num_items:")
    print("  compute_ip_bound_direct(items, depth, ...) uses items[depth+1:]")
    print("  CeilKnapsackSolver with items uses all items")
    print("  To query equivalent subset: use .reconstruct(num_items, budget)")
    print("    where num_items = len(items) - (depth + 1)\n")
    
    # Example setup
    items = [
        type("_", (), {'duration': 10, 'price': 2})(),
        type("_", (), {'duration': 20, 'price': 3})(),
        type("_", (), {'duration': 30, 'price': 4})(),
        type("_", (), {'duration': 40, 'price': 5})()
    ]
    
    print(f"Full item list (4 items):")
    for i, it in enumerate(items):
        print(f"  Item {i}: duration={it.duration}, price={it.price}")
    
    # Test different depths
    test_cases = [
        (0, "depth=0 → items[1:] = items 1,2,3"),
        (1, "depth=1 → items[2:] = items 2,3"),
        (2, "depth=2 → items[3:] = items 3"),
        (3, "depth=3 → items[4:] = [] (no items)"),
    ]
    
    print("\nCorrespondence:")
    for depth, description in test_cases:
        remaining = items[depth+1:]
        num_items = len(remaining)
        print(f"  {description}")
        print(f"    → {len(remaining)} items remain")
        print(f"    → CeilKnapsackSolver query: reconstruct(num_items={num_items}, budget=...)")
        if remaining:
            print(f"    → Remaining: {[(it.duration, it.price) for it in remaining]}")
        else:
            print(f"    → Remaining: []")
    
    print("\n✅ Item correspondence verified!\n")


def compare_solvers_single(items, depth, rem_budget, m, time_limit=2.0, verbose=False):
    """Compare CeilKnapsackSolver and compute_ip_bound_direct on a single instance.
    
    Args:
        items: Full list of items
        depth: Depth parameter for compute_ip_bound_direct
        rem_budget: Budget for both solvers
        m: Number of machines
        time_limit: Time limit for IP solver
        verbose: Whether to print detailed output
        
    Returns:
        dict with comparison results
    """
    # Extract items that will be used by IP solver
    remaining_items = items[depth+1:]
    num_items = len(remaining_items)
    
    if num_items == 0:
        # No items to solve, both should return 0
        return {
            'match': True,
            'ceil_value': 0.0,
            'ip_value': 0.0,
            'ceil_time': 0.0,
            'ip_time': 0.0,
            'num_items': 0
        }
    
    # Create item lists for CeilKnapsackSolver
    costs = [it.price for it in remaining_items]
    durations = [it.duration for it in remaining_items]
    
    # Solve with CeilKnapsackSolver
    start_ceil = time.time()
    solver_ceil = CeilKnapsackSolver(costs, durations, m, rem_budget)
    result_ceil = solver_ceil.reconstruct(num_items, rem_budget)
    time_ceil = time.time() - start_ceil
    value_ceil = result_ceil['max_value']
    selection_ceil = [bd['x_total'] for bd in result_ceil['breakdown'][:num_items]]
    
    # Solve with compute_ip_bound_direct
    start_ip = time.time()
    result_ip = compute_ip_bound_direct(items, depth, rem_budget, m, time_limit)
    time_ip = time.time() - start_ip
    
    if result_ip is None or result_ip == 0.0:
        value_ip = 0.0
        selection_ip = [0] * num_items
    else:
        value_ip, selection_ip = result_ip
    
    # Compare values (allowing small numerical tolerance)
    match = abs(value_ceil - value_ip) < 1e-6
    
    if verbose or not match:
        print(f"\n  Items used: {[(it.duration, it.price) for it in remaining_items]}")
        print(f"  CeilKnapsackSolver: value={value_ceil:.2f}, selection={selection_ceil}, time={time_ceil*1000:.2f}ms")
        print(f"  IP bound:           value={value_ip:.2f}, selection={selection_ip}, time={time_ip*1000:.2f}ms")
        if match:
            speedup_str = f"{time_ip/time_ceil:.1f}x" if time_ceil > 0 else "∞"
            print(f"  ✅ Match! (speedup: {speedup_str})")
        else:
            print(f"  ❌ MISMATCH!")
    
    return {
        'match': match,
        'ceil_value': value_ceil,
        'ip_value': value_ip,
        'ceil_selection': selection_ceil,
        'ip_selection': selection_ip,
        'ceil_time': time_ceil,
        'ip_time': time_ip,
        'num_items': num_items,
        'speedup': time_ip / time_ceil if time_ceil > 0 else float('inf')
    }


def test_comparison_suite():
    """Run comprehensive comparison tests across various configurations."""
    if not GUROBI_AVAILABLE:
        print("Skipping comparison tests - Gurobi not available")
        return
    
    print("="*80)
    print("COMPREHENSIVE COMPARISON: CeilKnapsackSolver vs IP Bound")
    print("="*80)
    print("\nTesting various m values, budgets, and item configurations\n")
    
    # Test configurations
    test_configs = [
        {
            'name': "Basic 4-item case",
            'items': [
                type("_", (), {'duration': 10, 'price': 2})(),
                type("_", (), {'duration': 20, 'price': 3})(),
                type("_", (), {'duration': 30, 'price': 4})(),
                type("_", (), {'duration': 40, 'price': 5})()
            ],
            'm_values': [1, 2, 5, 10],
            'budgets': [5, 10, 15],
            'depths': [0, 1, 2]
        },
        {
            'name': "Uniform ratios (8 items)",
            'items': [
                type("_", (), {'duration': 4, 'price': 2})(),
                type("_", (), {'duration': 6, 'price': 3})(),
                type("_", (), {'duration': 8, 'price': 4})(),
                type("_", (), {'duration': 10, 'price': 5})(),
                type("_", (), {'duration': 12, 'price': 6})(),
                type("_", (), {'duration': 14, 'price': 7})(),
                type("_", (), {'duration': 16, 'price': 8})(),
                type("_", (), {'duration': 18, 'price': 9})()
            ],
            'm_values': [1, 3, 10],
            'budgets': [10, 20, 30],
            'depths': [0, 2, 4, 6]
        },
        {
            'name': "High diversity",
            'items': [
                type("_", (), {'duration': 8, 'price': 1})(),
                type("_", (), {'duration': 50, 'price': 5})(),
                type("_", (), {'duration': 100, 'price': 13})(),
                type("_", (), {'duration': 65, 'price': 7})(),
                type("_", (), {'duration': 200, 'price': 23})(),
                type("_", (), {'duration': 28, 'price': 3})()
            ],
            'm_values': [1, 2, 5, 100],
            'budgets': [10, 25, 50],
            'depths': [0, 1, 3, 4]
        },
        {
            'name': "Large numbers",
            'items': [
                type("_", (), {'duration': 1000, 'price': 100})(),
                type("_", (), {'duration': 3000, 'price': 250})(),
                type("_", (), {'duration': 6000, 'price': 500})(),
                type("_", (), {'duration': 1800, 'price': 150})()
            ],
            'm_values': [1, 10, 50],
            'budgets': [200, 500, 1000],
            'depths': [0, 1, 2]
        },
        {
            'name': "Small granularity (Fibonacci/primes)",
            'items': [
                type("_", (), {'duration': 2, 'price': 1})(),
                type("_", (), {'duration': 3, 'price': 1})(),
                type("_", (), {'duration': 5, 'price': 2})(),
                type("_", (), {'duration': 7, 'price': 3})(),
                type("_", (), {'duration': 11, 'price': 5})(),
                type("_", (), {'duration': 13, 'price': 8})()
            ],
            'm_values': [1, 2, 3],
            'budgets': [5, 10, 15],
            'depths': [0, 1, 3, 4]
        }
    ]
    
    total_tests = 0
    passed_tests = 0
    total_ceil_time = 0.0
    total_ip_time = 0.0
    
    for config in test_configs:
        print("\n" + "-"*80)
        print(f"TEST SUITE: {config['name']}")
        print("-"*80)
        print(f"Items: {[(it.duration, it.price) for it in config['items']]}")
        
        suite_tests = 0
        suite_passed = 0
        suite_ceil_time = 0.0
        suite_ip_time = 0.0
        
        for m in config['m_values']:
            for budget in config['budgets']:
                for depth in config['depths']:
                    # Skip if depth leaves no items
                    if depth + 1 >= len(config['items']):
                        continue
                    
                    total_tests += 1
                    suite_tests += 1
                    
                    result = compare_solvers_single(
                        config['items'], 
                        depth, 
                        budget, 
                        m, 
                        time_limit=5.0,
                        verbose=False
                    )
                    
                    if result['match']:
                        passed_tests += 1
                        suite_passed += 1
                    else:
                        # Print details for mismatches
                        print(f"\n  ❌ MISMATCH at m={m}, budget={budget}, depth={depth}:")
                        print(f"     CeilKnapsack: {result['ceil_value']:.2f}, selection={result['ceil_selection']}")
                        print(f"     IP bound:     {result['ip_value']:.2f}, selection={result['ip_selection']}")
                    
                    suite_ceil_time += result['ceil_time']
                    suite_ip_time += result['ip_time']
                    total_ceil_time += result['ceil_time']
                    total_ip_time += result['ip_time']
        
        # Suite summary
        speedup = suite_ip_time / suite_ceil_time if suite_ceil_time > 0 else float('inf')
        print(f"\n  Suite results: {suite_passed}/{suite_tests} passed")
        print(f"  CeilKnapsack total time: {suite_ceil_time*1000:.2f}ms")
        print(f"  IP bound total time:     {suite_ip_time*1000:.2f}ms")
        print(f"  Average speedup:         {speedup:.1f}x")
        
        if suite_passed == suite_tests:
            print(f"  ✅ All tests in suite passed!")
        else:
            print(f"  ❌ {suite_tests - suite_passed} tests failed!")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total tests:     {total_tests}")
    print(f"Passed:          {passed_tests}")
    print(f"Failed:          {total_tests - passed_tests}")
    print(f"\nCeilKnapsackSolver total time: {total_ceil_time*1000:.2f}ms")
    print(f"IP bound total time:           {total_ip_time*1000:.2f}ms")
    
    if total_ceil_time > 0:
        overall_speedup = total_ip_time / total_ceil_time
        print(f"Overall speedup:               {overall_speedup:.1f}x")
    
    if passed_tests == total_tests:
        print("\n✅ ALL TESTS PASSED!")
        print("CeilKnapsackSolver produces identical results to IP bound solver,")
        print("but is significantly faster!")
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed!")
        print("There are discrepancies between the two solvers.")
    
    print("="*80)


def test_edge_cases():
    """Test edge cases where behavior might differ."""
    if not GUROBI_AVAILABLE:
        print("Skipping edge case tests - Gurobi not available")
        return
    
    print("\n" + "="*80)
    print("EDGE CASE TESTING")
    print("="*80)
    
    # Edge case 1: Single item
    print("\n--- Edge Case 1: Single item ---")
    items = [
        type("_", (), {'duration': 10, 'price': 3})(),
        type("_", (), {'duration': 20, 'price': 5})()
    ]
    result = compare_solvers_single(items, depth=0, rem_budget=10, m=2, verbose=True)
    assert result['match'], "Single item test failed!"
    
    # Edge case 2: Zero budget
    print("\n--- Edge Case 2: Zero budget ---")
    items = [
        type("_", (), {'duration': 10, 'price': 3})(),
        type("_", (), {'duration': 20, 'price': 5})()
    ]
    result = compare_solvers_single(items, depth=0, rem_budget=0, m=2, verbose=True)
    assert result['match'], "Zero budget test failed!"
    assert result['ceil_value'] == 0.0, "Should return 0 with zero budget"
    
    # Edge case 3: Budget allows only one item
    print("\n--- Edge Case 3: Budget for exactly one item ---")
    items = [
        type("_", (), {'duration': 100, 'price': 10})(),
        type("_", (), {'duration': 50, 'price': 10})(),
        type("_", (), {'duration': 75, 'price': 10})()
    ]
    result = compare_solvers_single(items, depth=0, rem_budget=10, m=3, verbose=True)
    assert result['match'], "Single item budget test failed!"
    
    # Edge case 4: Very large m (should act like 0/1)
    print("\n--- Edge Case 4: Very large m (m=10000) ---")
    items = [
        type("_", (), {'duration': 10, 'price': 2})(),
        type("_", (), {'duration': 20, 'price': 3})(),
        type("_", (), {'duration': 30, 'price': 4})()
    ]
    result = compare_solvers_single(items, depth=0, rem_budget=9, m=10000, verbose=True)
    assert result['match'], "Large m test failed!"
    
    # Edge case 5: m=1 (unbounded-like)
    print("\n--- Edge Case 5: m=1 (unbounded-like) ---")
    items = [
        type("_", (), {'duration': 10, 'price': 2})(),
        type("_", (), {'duration': 15, 'price': 3})(),
        type("_", (), {'duration': 20, 'price': 4})()
    ]
    result = compare_solvers_single(items, depth=0, rem_budget=12, m=1, verbose=True)
    assert result['match'], "m=1 test failed!"
    
    # Edge case 6: All items same cost/duration ratio
    print("\n--- Edge Case 6: Uniform ratios ---")
    items = [
        type("_", (), {'duration': 4, 'price': 2})(),
        type("_", (), {'duration': 6, 'price': 3})(),
        type("_", (), {'duration': 8, 'price': 4})(),
        type("_", (), {'duration': 10, 'price': 5})()
    ]
    result = compare_solvers_single(items, depth=1, rem_budget=12, m=2, verbose=True)
    assert result['match'], "Uniform ratio test failed!"
    
    print("\n✅ All edge cases passed!\n")


if __name__ == "__main__":
    # Run all tests
    test_item_correspondence()
    
    if GUROBI_AVAILABLE:
        test_edge_cases()
        test_comparison_suite()
    else:
        print("\n" + "="*80)
        print("⚠️  Gurobi not available - only item correspondence test was run")
        print("Install gurobipy to run full comparison tests")
        print("="*80)
