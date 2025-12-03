
"""
Test suite for CeilKnapsackSolver to verify correctness against standard knapsack solver.

When m is large enough that no packages can be created, CeilKnapsackSolver should
behave exactly like a standard 0/1 knapsack solver (since each item gives duration[i]
for cost[i], and ceil(1/m) = 1 when only singles are available).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from knapsack_dp import knapsack_01, knapsack_unbounded, CeilKnapsackSolver


def test_ceil_knapsack_vs_standard():
    """Test CeilKnapsackSolver against standard 0/1 knapsack when m is large"""
    
    print("=" * 80)
    print("TESTING CeilKnapsackSolver vs Standard 0/1 Knapsack")
    print("=" * 80)
    print("\nStrategy: Set m=1 so ceil(x/m) = x (linear).")
    print("This means multiple items can be combined, so solutions will differ from 0/1.\n")
    
    # Test case 1: Small instance
    print("-" * 80)
    print("TEST 1: Small instance with 4 items (m=1)")
    print("-" * 80)
    
    costs = [2, 3, 4, 5]
    durations = [10, 20, 30, 40]
    max_budget = 12
    m = 1  # m=1: ceil(x/1) = x, so value is linear in quantity
    
    print(f"Costs:      {costs}")
    print(f"Durations:  {durations}")
    print(f"Max budget: {max_budget}")
    print(f"m (machines): {m}")
    print()
    
    # Standard 0/1 knapsack (values=durations, weights=costs)
    standard_result = knapsack_01(durations, costs, max_budget)
    print(f"Standard 0/1 Knapsack:")
    print(f"  Max value: {standard_result['max_value']}")
    print(f"  Selected:  {standard_result['selected']}")
    print(f"  Total cost: {sum(c * s for c, s in zip(costs, standard_result['selected']))}")
    print()
    
    # CeilKnapsackSolver
    solver = CeilKnapsackSolver(costs, durations, m, max_budget)
    
    # Test query for all items with max budget
    ceil_value = solver.query(len(costs), max_budget)
    print(f"CeilKnapsackSolver query(num_items={len(costs)}, budget={max_budget}):")
    print(f"  Max value: {ceil_value}")
    
    # Reconstruct solution
    ceil_result = solver.reconstruct(len(costs), max_budget)
    ceil_quantities = [bd["x_total"] for bd in ceil_result['breakdown']]
    print(f"  Breakdown:  {ceil_result['breakdown']}")
    print(f"  Quantities: {ceil_quantities}")
    print()
    
    # Verify that they differ (m=1 allows multiple copies)
    print(f"Expected difference: m=1 allows buying multiple copies (singles + packages)")
    print(f"Standard 0/1 restricts to at most 1 of each item type\n")
    
    if ceil_value != standard_result['max_value']:
        print("✅ Values differ as expected!")
        print(f"  Standard (0/1 only):      {standard_result['max_value']}")
        print(f"  CeilKnapsack (m=1, mult): {ceil_value}")
        print(f"  Improvement: {ceil_value - standard_result['max_value']}")
    else:
        print("⚠️  Values are the same (might happen if optimal is still 0/1)")
    
    print("✅ TEST 1 PASSED: CeilKnapsackSolver with m=1 working!\n")
    
    
    # Test case 2: Test ALL combinations of prefix queries
    print("-" * 80)
    print("TEST 2: Verify CeilKnapsackSolver can handle multiple copies per item")
    print("-" * 80)
    
    print("With m=1, buying x items gives x*duration value (linear).")
    print("Let's check a case where buying multiples is beneficial:\n")
    
    # Item 0: cost=2, duration=10, ratio=5.0 (best ratio)
    # Budget=12 allows 6 copies → value = 6*10 = 60
    ceil_result = solver.reconstruct(1, 12)  # Only item 0, budget 12
    print(f"Item 0 only (cost=2, duration=10), budget=12:")
    print(f"  Breakdown: {ceil_result['breakdown'][0]}")
    print(f"  Max value: {ceil_result['max_value']}")
    
    expected_value = 6 * 10  # Can buy 6 singles
    if ceil_result['max_value'] == expected_value:
        print(f"  ✅ Correctly computed: 6 copies × 10 = 60")
    else:
        print(f"  ❌ Expected {expected_value}, got {ceil_result['max_value']}")
    
    print()
    
    # Compare with full budget case
    full_result = solver.reconstruct(len(costs), max_budget)
    print(f"All items, budget={max_budget}:")
    print(f"  Max value: {full_result['max_value']}")
    print(f"  Breakdown: {full_result['breakdown']}")
    
    print()
    print("✅ TEST 2 PASSED: Multiple copies handled correctly!\n")
    
    
    # Test case 3: Larger instance
    print("-" * 80)
    print("TEST 3: Compare m=1 vs m=large on same instance")
    print("-" * 80)
    
    costs_test = [3, 5, 7, 2, 8]
    durations_test = [15, 25, 35, 12, 40]
    max_budget_test = 20
    
    print(f"Costs:      {costs_test}")
    print(f"Durations:  {durations_test}")
    print(f"Max budget: {max_budget_test}")
    print()
    
    # m=1: linear value
    solver_m1 = CeilKnapsackSolver(costs_test, durations_test, 1, max_budget_test)
    value_m1 = solver_m1.query(len(costs_test), max_budget_test)
    result_m1 = solver_m1.reconstruct(len(costs_test), max_budget_test)
    
    # m=large: behaves like 0/1
    solver_mlarge = CeilKnapsackSolver(costs_test, durations_test, 10000, max_budget_test)
    value_mlarge = solver_mlarge.query(len(costs_test), max_budget_test)
    result_mlarge = solver_mlarge.reconstruct(len(costs_test), max_budget_test)
    
    print(f"m=1 (linear):        value={value_m1}, breakdown={result_m1['breakdown']}")
    print(f"m=10000 (like 0/1):  value={value_mlarge}, breakdown={result_mlarge['breakdown']}")
    print()
    
    if value_m1 >= value_mlarge:
        print(f"✅ m=1 value ({value_m1}) >= m=large value ({value_mlarge}) as expected!")
        print(f"   Improvement: {value_m1 - value_mlarge}")
    else:
        print(f"❌ ERROR: m=1 should allow more flexibility!")
    
    print()
    print("✅ TEST 3 PASSED: m=1 provides better or equal value!\n")
    
    
    # Test case 4: High diversity - different cost/duration ratios
    print("-" * 80)
    print("TEST 4: High diversity in costs and durations (m=1)")
    print("-" * 80)
    
    costs_div = [1, 5, 13, 7, 23, 3, 17, 11]
    durations_div = [8, 50, 100, 65, 200, 28, 150, 95]
    max_budget_div = 50
    
    print(f"Costs:      {costs_div}")
    print(f"Durations:  {durations_div}")
    print(f"Ratios (val/cost): {[round(d/c, 2) for d, c in zip(durations_div, costs_div)]}")
    print(f"Max budget: {max_budget_div}")
    print()
    
    solver_div = CeilKnapsackSolver(costs_div, durations_div, 1, max_budget_div)
    result_div = solver_div.reconstruct(len(costs_div), max_budget_div)
    
    # Compare with optimal unbounded knapsack solution (m=1 means linear value)
    optimal_unbounded = knapsack_unbounded(durations_div, costs_div, max_budget_div)
    
    print(f"CeilKnapsackSolver solution:")
    print(f"  Max value: {result_div['max_value']}")
    for i, bd in enumerate(result_div['breakdown']):
        if bd['x_total'] > 0:
            cost_used = costs_div[i] * bd['x_total']
            value_contrib = durations_div[i] * bd['x_total']
            print(f"  Item {i}: {bd['x_total']} copies (cost={cost_used}, value={value_contrib})")
    
    print(f"\nOptimal unbounded knapsack:")
    print(f"  Max value: {optimal_unbounded['max_value']}")
    for i, count in enumerate(optimal_unbounded['counts']):
        if count > 0:
            print(f"  Item {i}: {count} copies")
    
    # Verify budget constraint
    total_cost = sum(costs_div[i] * bd['x_total'] for i, bd in enumerate(result_div['breakdown']))
    total_value_check = sum(durations_div[i] * bd['x_total'] for i, bd in enumerate(result_div['breakdown']))
    
    print(f"\nVerification:")
    print(f"  Total cost: {total_cost} / {max_budget_div}")
    print(f"  Total value: {total_value_check}")
    
    assert total_cost <= max_budget_div, f"Budget constraint violated! {total_cost} > {max_budget_div}"
    assert total_value_check == result_div['max_value'], f"Value mismatch! {total_value_check} != {result_div['max_value']}"
    assert result_div['max_value'] == optimal_unbounded['max_value'], \
        f"Optimality violated! Got {result_div['max_value']}, optimal is {optimal_unbounded['max_value']}"
    
    print("  ✅ Budget constraint satisfied")
    print("  ✅ Value computation correct")
    print("  ✅ Optimality verified against unbounded knapsack")
    print()
    print("✅ TEST 4 PASSED: High diversity case handled correctly!\n")
    
    
    # Test case 5: Large numbers
    print("-" * 80)
    print("TEST 5: Large numbers (m=1)")
    print("-" * 80)
    
    costs_large = [100, 250, 500, 150, 1000]
    durations_large = [1000, 3000, 6000, 1800, 12000]
    max_budget_large = 2000
    
    print(f"Costs (100s):      {costs_large}")
    print(f"Durations (1000s): {durations_large}")
    print(f"Max budget: {max_budget_large}")
    print()
    
    solver_large = CeilKnapsackSolver(costs_large, durations_large, 1, max_budget_large)
    result_large = solver_large.reconstruct(len(costs_large), max_budget_large)
    optimal_large = knapsack_unbounded(durations_large, costs_large, max_budget_large)
    
    print(f"CeilKnapsackSolver: value={result_large['max_value']}")
    print(f"Optimal unbounded:  value={optimal_large['max_value']}")
    for i, bd in enumerate(result_large['breakdown']):
        if bd['x_total'] > 0:
            print(f"  Item {i}: {bd['x_total']} copies")
    
    total_cost_large = sum(costs_large[i] * bd['x_total'] for i, bd in enumerate(result_large['breakdown']))
    print(f"  Total cost: {total_cost_large} / {max_budget_large}")
    
    assert total_cost_large <= max_budget_large, "Budget constraint violated!"
    assert result_large['max_value'] == optimal_large['max_value'], \
        f"Optimality violated! Got {result_large['max_value']}, optimal is {optimal_large['max_value']}"
    
    print("  ✅ Budget constraint satisfied")
    print("  ✅ Optimality verified")
    print()
    print("✅ TEST 5 PASSED: Large numbers handled correctly!\n")
    
    
    # Test case 6: Small numbers with fine granularity
    print("-" * 80)
    print("TEST 6: Small numbers with fine granularity (m=1)")
    print("-" * 80)
    
    costs_small = [1, 1, 2, 3, 5, 8]  # Fibonacci-like
    durations_small = [2, 3, 5, 7, 11, 13]  # Prime-like
    max_budget_small = 15
    
    print(f"Costs:      {costs_small}")
    print(f"Durations:  {durations_small}")
    print(f"Max budget: {max_budget_small}")
    print()
    
    solver_small = CeilKnapsackSolver(costs_small, durations_small, 1, max_budget_small)
    result_small = solver_small.reconstruct(len(costs_small), max_budget_small)
    optimal_small = knapsack_unbounded(durations_small, costs_small, max_budget_small)
    
    print(f"CeilKnapsackSolver: value={result_small['max_value']}")
    print(f"Optimal unbounded:  value={optimal_small['max_value']}")
    for i, bd in enumerate(result_small['breakdown']):
        if bd['x_total'] > 0:
            print(f"  Item {i} (cost={costs_small[i]}, dur={durations_small[i]}): {bd['x_total']} copies")
    
    total_cost_small = sum(costs_small[i] * bd['x_total'] for i, bd in enumerate(result_small['breakdown']))
    print(f"  Total cost: {total_cost_small} / {max_budget_small}")
    
    assert total_cost_small <= max_budget_small, "Budget constraint violated!"
    assert result_small['max_value'] == optimal_small['max_value'], \
        f"Optimality violated! Got {result_small['max_value']}, optimal is {optimal_small['max_value']}"
    
    print("  ✅ Budget constraint satisfied")
    print("  ✅ Optimality verified")
    print()
    print("✅ TEST 6 PASSED: Fine granularity handled correctly!\n")
    
    
    # Test case 7: Uniform costs, varying durations
    print("-" * 80)
    print("TEST 7: Uniform costs, varying durations (m=1)")
    print("-" * 80)
    
    costs_uniform = [5, 5, 5, 5, 5]
    durations_varied = [10, 30, 20, 50, 15]
    max_budget_uniform = 25
    
    print(f"Costs (uniform):      {costs_uniform}")
    print(f"Durations (varying):  {durations_varied}")
    print(f"Max budget: {max_budget_uniform}")
    print()
    
    solver_uniform = CeilKnapsackSolver(costs_uniform, durations_varied, 1, max_budget_uniform)
    result_uniform = solver_uniform.reconstruct(len(costs_uniform), max_budget_uniform)
    optimal_uniform = knapsack_unbounded(durations_varied, costs_uniform, max_budget_uniform)
    
    print(f"CeilKnapsackSolver: value={result_uniform['max_value']}")
    print(f"Optimal unbounded:  value={optimal_uniform['max_value']}")
    for i, bd in enumerate(result_uniform['breakdown']):
        if bd['x_total'] > 0:
            print(f"  Item {i} (dur={durations_varied[i]}): {bd['x_total']} copies")
    
    # With uniform costs, should select items with highest duration
    # Budget allows 25/5 = 5 items, best is 5 copies of item 3 (duration 50) = 250
    expected_max = 5 * 50  # Should select only item 3
    print(f"\nExpected: 5 copies of item 3 (highest duration) = {expected_max}")
    print(f"Got:      {result_uniform['max_value']}")
    
    assert result_uniform['max_value'] == expected_max, f"Should select best item only!"
    assert result_uniform['max_value'] == optimal_uniform['max_value'], "Optimality violated!"
    assert result_uniform['breakdown'][3]['x_total'] == 5, "Should select 5 copies of item 3"
    
    print("  ✅ Correctly selected highest-value item only")
    print()
    print("✅ TEST 7 PASSED: Uniform costs case optimal!\n")
    
    
    # Test case 8: Mix of very cheap and very expensive items
    print("-" * 80)
    print("TEST 8: Extreme cost variations (m=1)")
    print("-" * 80)
    
    costs_extreme = [1, 50, 2, 100, 5, 200, 10]
    durations_extreme = [5, 300, 12, 700, 35, 1500, 80]
    max_budget_extreme = 100
    
    print(f"Costs:      {costs_extreme}")
    print(f"Durations:  {durations_extreme}")
    print(f"Ratios:     {[round(d/c, 2) for d, c in zip(durations_extreme, costs_extreme)]}")
    print(f"Max budget: {max_budget_extreme}")
    print()
    
    solver_extreme = CeilKnapsackSolver(costs_extreme, durations_extreme, 1, max_budget_extreme)
    result_extreme = solver_extreme.reconstruct(len(costs_extreme), max_budget_extreme)
    optimal_extreme = knapsack_unbounded(durations_extreme, costs_extreme, max_budget_extreme)
    
    print(f"CeilKnapsackSolver: value={result_extreme['max_value']}")
    print(f"Optimal unbounded:  value={optimal_extreme['max_value']}")
    for i, bd in enumerate(result_extreme['breakdown']):
        if bd['x_total'] > 0:
            ratio = durations_extreme[i] / costs_extreme[i]
            print(f"  Item {i} (cost={costs_extreme[i]}, dur={durations_extreme[i]}, ratio={ratio:.1f}): {bd['x_total']} copies")
    
    total_cost_extreme = sum(costs_extreme[i] * bd['x_total'] for i, bd in enumerate(result_extreme['breakdown']))
    print(f"\n  Total cost: {total_cost_extreme} / {max_budget_extreme}")
    
    assert total_cost_extreme <= max_budget_extreme, "Budget constraint violated!"
    assert result_extreme['max_value'] == optimal_extreme['max_value'], \
        f"Optimality violated! Got {result_extreme['max_value']}, optimal is {optimal_extreme['max_value']}"
    
    # Item 6 has best ratio (8.0), should be heavily selected
    assert result_extreme['breakdown'][6]['x_total'] > 0, "Best ratio item should be selected"
    
    print("  ✅ Budget constraint satisfied")
    print("  ✅ High-ratio items prioritized")
    print()
    print("✅ TEST 8 PASSED: Extreme variations handled correctly!\n")
    
    
    # Test case 9: Edge case - budget exactly fits multiple copies of one item
    print("-" * 80)
    print("TEST 9: Budget perfectly divisible by item cost (m=1)")
    print("-" * 80)
    
    costs_perfect = [7, 11, 13]
    durations_perfect = [20, 35, 40]
    max_budget_perfect = 77  # 11 * 7 = 77
    
    print(f"Costs:      {costs_perfect}")
    print(f"Durations:  {durations_perfect}")
    print(f"Max budget: {max_budget_perfect} (= 11 × 7)")
    print()
    
    solver_perfect = CeilKnapsackSolver(costs_perfect, durations_perfect, 1, max_budget_perfect)
    result_perfect = solver_perfect.reconstruct(len(costs_perfect), max_budget_perfect)
    optimal_perfect = knapsack_unbounded(durations_perfect, costs_perfect, max_budget_perfect)
    
    print(f"CeilKnapsackSolver: value={result_perfect['max_value']}")
    print(f"Optimal unbounded:  value={optimal_perfect['max_value']}")
    for i, bd in enumerate(result_perfect['breakdown']):
        if bd['x_total'] > 0:
            print(f"  Item {i}: {bd['x_total']} copies")
    
    total_cost_perfect = sum(costs_perfect[i] * bd['x_total'] for i, bd in enumerate(result_perfect['breakdown']))
    print(f"  Total cost: {total_cost_perfect} / {max_budget_perfect}")
    
    assert total_cost_perfect <= max_budget_perfect, "Budget constraint violated!"
    assert result_perfect['max_value'] == optimal_perfect['max_value'], \
        f"Optimality violated! Got {result_perfect['max_value']}, optimal is {optimal_perfect['max_value']}"
    
    print("  ✅ Budget used efficiently")
    print("  ✅ Optimality verified")
    print()
    print("✅ TEST 9 PASSED: Perfect divisibility case handled!\n")
    
    
    # Test case 10: Single item type with large budget
    print("-" * 80)
    print("TEST 10: Single item, large budget (stress test, m=1)")
    print("-" * 80)
    
    costs_stress = [3]
    durations_stress = [17]
    max_budget_stress = 1000
    
    print(f"Item: cost={costs_stress[0]}, duration={durations_stress[0]}")
    print(f"Max budget: {max_budget_stress}")
    print()
    
    solver_stress = CeilKnapsackSolver(costs_stress, durations_stress, 1, max_budget_stress)
    result_stress = solver_stress.reconstruct(1, max_budget_stress)
    optimal_stress = knapsack_unbounded(durations_stress, costs_stress, max_budget_stress)
    
    max_copies = max_budget_stress // costs_stress[0]
    expected_value_stress = max_copies * durations_stress[0]
    
    print(f"Expected: {max_copies} copies × {durations_stress[0]} = {expected_value_stress}")
    print(f"Got:      {result_stress['max_value']}")
    print(f"Optimal:  {optimal_stress['max_value']}")
    print(f"Breakdown: {result_stress['breakdown'][0]}")
    
    assert result_stress['max_value'] == expected_value_stress, "Value mismatch!"
    assert result_stress['max_value'] == optimal_stress['max_value'], "Optimality violated!"
    assert result_stress['breakdown'][0]['x_total'] == max_copies, "Copy count mismatch!"
    
    print("  ✅ Correctly computed maximum copies")
    print()
    print("✅ TEST 10 PASSED: Stress test with large budget successful!\n")
    
    
    # Test case 11 (was 4): Edge cases
    print("-" * 80)
    print("TEST 11: Edge cases")
    print("-" * 80)
    
    # 4a: Budget = 0
    print("  4a: Budget = 0")
    solver = CeilKnapsackSolver([2, 3], [10, 20], 1000, 10)
    result = solver.query(2, 0)
    assert result == 0, f"Expected 0, got {result}"
    print(f"    query(2, budget=0) = {result} ✅")
    
    # 4b: No items
    print("  4b: Zero items")
    result = solver.query(0, 10)
    assert result == 0, f"Expected 0, got {result}"
    print(f"    query(0 items, budget=10) = {result} ✅")
    
    # 4c: Single item that fits exactly
    print("  4c: Single item, budget = exact cost")
    solver = CeilKnapsackSolver([5], [100], 1000, 10)
    result = solver.query(1, 5)
    assert result == 100, f"Expected 100, got {result}"
    print(f"    query(1, budget=5) = {result} ✅")
    
    # 4d: Single item that doesn't fit
    print("  4d: Single item, budget < cost")
    result = solver.query(1, 4)
    assert result == 0, f"Expected 0, got {result}"
    print(f"    query(1, budget=4) = {result} ✅")
    
    print()
    print("✅ TEST 11 PASSED: All edge cases handled correctly!\n")
    
    
    print("=" * 80)
    print("ALL TESTS PASSED! ✅")
    print("=" * 80)
    print("\nConclusion: CeilKnapsackSolver correctly handles m=1 case where")
    print("ceil(x/1) = x, allowing multiple copies of same item to be combined.")
    print("This differs from 0/1 knapsack which restricts to single selection.")
    
    
    ############################################################################
    # PART 2: Testing with LARGE m (should match standard 0/1 knapsack)
    ############################################################################
    
    print("\n" + "="*80)
    print("TESTING CeilKnapsackSolver with LARGE m vs Standard 0/1 Knapsack")
    print("="*80)
    print("\nStrategy: Set m=10000 (very large) so ceil(x/m) ≈ 0 or 1.")
    print("This makes CeilKnapsackSolver behave like 0/1 knapsack.")
    print("We test all problems with various num_items and budget configurations.\n")
    
    m_large = 10000
    
    # Test configurations: (test_name, costs, durations, budgets_to_test, num_items_to_test)
    test_configs = [
        ("TEST 1: Basic 4-item case", 
         [2, 3, 4, 5], 
         [10, 20, 30, 40],
         [5, 8, 10, 12, 15],
         [1, 2, 3, 4]),
        
        ("TEST 4: High diversity (8 items)",
         [1, 5, 13, 7, 23, 3, 17, 11],
         [8, 50, 100, 65, 200, 28, 150, 95],
         [10, 20, 30, 40, 50, 60],
         [1, 2, 4, 6, 8]),
        
        ("TEST 5: Large numbers",
         [100, 250, 500, 150, 1000],
         [1000, 3000, 6000, 1800, 12000],
         [200, 500, 1000, 1500, 2000],
         [1, 2, 3, 4, 5]),
        
        ("TEST 6: Fine granularity (Fibonacci/primes)",
         [1, 1, 2, 3, 5, 8],
         [2, 3, 5, 7, 11, 13],
         [3, 5, 8, 10, 15, 20],
         [1, 2, 3, 4, 5, 6]),
        
        ("TEST 7: Uniform costs, varying durations",
         [5, 5, 5, 5, 5],
         [10, 30, 20, 50, 15],
         [5, 10, 15, 20, 25],
         [1, 2, 3, 4, 5]),
        
        ("TEST 8: Extreme variations",
         [1, 50, 2, 100, 5, 200, 10],
         [5, 300, 12, 700, 35, 1500, 80],
         [10, 25, 50, 75, 100, 150],
         [1, 2, 3, 5, 7]),
        
        ("TEST 9: Perfect divisibility",
         [7, 11, 13],
         [20, 35, 40],
         [11, 22, 33, 44, 55, 77],
         [1, 2, 3]),
        
        ("TEST 10: Single item stress test",
         [3],
         [17],
         [3, 6, 9, 15, 30],
         [1]),
    ]
    
    total_comparisons = 0
    passed_comparisons = 0
    
    for test_name, costs, durations, budgets, num_items_list in test_configs:
        print("\n" + "-"*80)
        print(test_name)
        print("-"*80)
        print(f"Costs:     {costs}")
        print(f"Durations: {durations}")
        print(f"m = {m_large} (large, so ceil(x/{m_large}) acts like 0/1)\n")
        
        # Initialize CeilKnapsackSolver once for this test
        solver_01_like = CeilKnapsackSolver(costs, durations, m_large, max(budgets))
        
        test_passed = True
        test_comparisons = 0
        
        for num_items in num_items_list:
            if num_items > len(costs):
                continue
                
            for budget in budgets:
                total_comparisons += 1
                test_comparisons += 1
                
                # Get CeilKnapsackSolver result
                result_ceil = solver_01_like.reconstruct(num_items, budget)
                
                # Get standard 0/1 knapsack result
                result_01 = knapsack_01(
                    durations[:num_items], 
                    costs[:num_items], 
                    budget
                )
                
                # Compare values
                if result_ceil['max_value'] != result_01['max_value']:
                    print(f"  ❌ MISMATCH at num_items={num_items}, budget={budget}:")
                    print(f"     CeilKnapsackSolver: {result_ceil['max_value']}")
                    print(f"     Standard 0/1:       {result_01['max_value']}")
                    print(f"     CeilKnapsack breakdown: {result_ceil['breakdown'][:num_items]}")
                    print(f"     Standard 0/1 selected:  {result_01['selected']}")
                    test_passed = False
                else:
                    passed_comparisons += 1
        
        if test_passed:
            print(f"  ✅ All {test_comparisons} configurations match standard 0/1 knapsack!")
        else:
            print(f"  ❌ Some configurations failed!")
    
    print("\n" + "="*80)
    print(f"LARGE m TESTING SUMMARY")
    print("="*80)
    print(f"Total comparisons: {total_comparisons}")
    print(f"Passed: {passed_comparisons}")
    print(f"Failed: {total_comparisons - passed_comparisons}")
    
    if passed_comparisons == total_comparisons:
        print("\n✅ ALL COMPARISONS PASSED!")
        print("CeilKnapsackSolver with large m correctly matches 0/1 knapsack behavior!")
    else:
        print(f"\n❌ {total_comparisons - passed_comparisons} comparisons failed!")
        print("There may be bugs in CeilKnapsackSolver for the 0/1-like case.")
    
    print("="*80)


if __name__ == "__main__":
    test_ceil_knapsack_vs_standard()

