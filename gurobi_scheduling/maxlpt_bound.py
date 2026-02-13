"""Max-LPT Upper Bound for Branch-and-Bound Algorithm.

This module implements the Max-LPT algorithm which provides a tighter
upper bound for the bilevel knapsack-scheduling problem. It maximizes
the makespan achieved by the Longest Processing Time (LPT) scheduling rule.

The Max-LPT approach provides a 3/4-approximation guarantee for the
bilevel problem, making it useful both as an initial incumbent and as
an upper bound during branch-and-bound search.
"""

import math
from typing import List, Tuple, Dict, Optional
import random


# ============================================================================
# Reference Implementation - Standard Unbounded Knapsack
# ============================================================================

def unbounded_knapsack_reference(capacity: int, durations: List[float], 
                                  prices: List[float]) -> float:
    """Reference implementation of unbounded knapsack.
    
    This is the standard algorithm where we iterate through items
    and allow each item to be selected multiple times. The key insight
    is that dp[i][j - wt[i]] refers to the SAME row, allowing the
    item i to be selected multiple times.
    
    Args:
        capacity: Maximum capacity (budget)
        durations: List of item values (durations)
        prices: List of item weights (prices)
        
    Returns:
        Maximum total duration achievable within budget
    """
    n = len(durations)
    # dp[i][j] = max value using items 0..i-1 with capacity j
    dp = [[0.0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Iterate backwards through items
    for i in range(n - 1, -1, -1):
        for j in range(1, capacity + 1):
            # Option 1: Take one copy of item i
            take = 0.0
            if j - prices[i] >= 0:
                # Use dp[i][j - prices[i]] (same row) for unbounded
                take = durations[i] + dp[i][j - prices[i]]
            
            # Option 2: Don't take item i
            noTake = dp[i + 1][j]
            
            # Take maximum
            dp[i][j] = max(take, noTake)
    
    return dp[0][capacity]


class UnboundedKnapsackDP:
    """Unbounded knapsack solver using dynamic programming.
    
    Solves the unbounded knapsack problem where each item can be selected
    multiple times. Precomputes DP tables for efficient querying.
    
    The DP table stores:
        dp[i][b] = maximum value using items 0..i with budget b
    
    Time complexity: O(n * B) for table construction
    Space complexity: O(n * B)
    """
    
    def __init__(self, durations: List[float], prices: List[float], max_budget: int):
        """Initialize the unbounded knapsack solver.
        
        Args:
            durations: List of item durations (values)
            prices: List of item prices (costs/weights)
            max_budget: Maximum budget to consider
        """
        self.durations = durations
        self.prices = prices
        self.n = len(durations)
        self.max_budget = max_budget
        
        # DP table: dp[i][b] = max value using items 0..i-1 with budget b
        self.dp = [[0.0 for _ in range(max_budget + 1)] for _ in range(self.n + 1)]
        
        # Build the DP table
        self._build_dp_table()
    
    def _build_dp_table(self):
        """Build the DP table using unbounded knapsack recurrence.
        
        For each item i and budget b:
            dp[i][b] = max over k of {dp[i-1][b - k*p_i] + k*d_i}
        where k is the number of copies of item i to take.
        """
        for i in range(1, self.n + 1):
            price = self.prices[i - 1]
            duration = self.durations[i - 1]
            
            for b in range(self.max_budget + 1):
                # Option 1: Don't use item i-1 at all
                self.dp[i][b] = self.dp[i - 1][b]
                
                # Option 2: Use k copies of item i-1
                if price > 0:
                    max_copies = int(b // price)
                    for k in range(1, max_copies + 1):
                        if k * price <= b:
                            value = self.dp[i - 1][b - k * price] + k * duration
                            self.dp[i][b] = max(self.dp[i][b], value)
    
    def query(self, num_items: int, budget: int) -> float:
        """Query the maximum value for first num_items with given budget.
        
        Args:
            num_items: Number of items to consider (0 to n)
            budget: Budget available
            
        Returns:
            Maximum value achievable
        """
        if num_items > self.n or budget > self.max_budget:
            raise ValueError(f"Invalid query: num_items={num_items}, budget={budget}")
        return self.dp[num_items][budget]
    
    def reconstruct(self, num_items: int, budget: int) -> Dict:
        """Reconstruct the optimal selection for given parameters.
        
        Args:
            num_items: Number of items to consider
            budget: Budget available
            
        Returns:
            dict with:
                - max_value: Maximum value achievable
                - counts: List of item counts in optimal solution
        """
        if num_items > self.n or budget > self.max_budget:
            raise ValueError(f"Invalid query: num_items={num_items}, budget={budget}")
        
        counts = [0] * num_items
        b = budget
        
        # Backtrack through DP table
        for i in range(num_items, 0, -1):
            price = self.prices[i - 1]
            duration = self.durations[i - 1]
            
            # Find how many copies of item i-1 were taken
            if price > 0:
                max_copies = int(b // price)
                for k in range(max_copies, -1, -1):
                    if k * price <= b:
                        if abs(self.dp[i][b] - (self.dp[i - 1][b - k * price] + k * duration)) < 1e-9:
                            counts[i - 1] = k
                            b -= k * price
                            break
        
        return {
            'max_value': self.dp[num_items][budget],
            'counts': counts
        }


def solve_max_lpt(durations: List[float], prices: List[float], budget: int, 
                  m: int, dp_solver: UnboundedKnapsackDP) -> float:
    """Solve the Max-LPT problem for given items and budget.
    
    The Max-LPT problem finds the job selection that maximizes the makespan
    when jobs are scheduled using the Longest Processing Time (LPT) rule.
    
    Algorithm:
    1. For each item type J as candidate for the last job:
       - Compute per-machine budget: B_i = floor(B/m)
       - Compute remaining budget: B_hat = B - m*B_i
       - If B_hat < c_J: adjust B_i downward until sufficient
       - Evaluate: z = DP(i, B_i) + d_J
    2. Return maximum z over all candidates
    
    Args:
        durations: List of job durations
        prices: List of job prices
        budget: Total budget available
        m: Number of machines
        dp_solver: Pre-computed UnboundedKnapsackDP solver
        
    Returns:
        Maximum makespan achievable with LPT scheduling
    """
    n = len(durations)
    z_star = 0.0
    
    for J in range(n):
        # Compute per-machine budget
        B_i = budget // m
        B_hat = budget - m * B_i
        
        # Check if remaining budget is sufficient for item J
        if B_hat < prices[J]:
            # Adjust B_i downward until we have enough remaining budget
            k = 0
            while B_hat < prices[J] and B_i > 0:
                k += 1
                B_i = (budget - k * m) // m
                B_hat = budget - m * B_i
                
                # Safety check to prevent infinite loop
                if B_i < 0 or B_hat < 0:
                    break
            
            # If still not enough budget, skip this item
            if B_hat < prices[J]:
                continue
        
        # Evaluate this configuration
        # DP solver gives us the best value for items up to J with budget B_i
        dp_value = dp_solver.query(J + 1, B_i)
        z = dp_value + durations[J]
        
        z_star = max(z_star, z)
    
    return z_star


def compute_maxlpt_bound(durations: List[float], prices: List[float], 
                         budget: int, m: int, depth: int,
                         dp_solvers: List[UnboundedKnapsackDP],
                         already_committed_makespan: float) -> float:
    """Compute Max-LPT upper bound at a given node in the BnB tree.
    
    This function uses the Max-LPT algorithm to compute an upper bound on
    the best possible makespan in the subtree rooted at the current node.
    
    Args:
        durations: Full list of job durations
        prices: Full list of job prices
        budget: Remaining budget at current node
        m: Number of machines
        depth: Current depth in search tree
        dp_solvers: List of UnboundedKnapsackDP solvers (one per depth)
        already_committed_makespan: Makespan from jobs already committed
        
    Returns:
        Upper bound on makespan in this subtree
    """
    # If at leaf or no remaining budget, return committed makespan
    if depth >= len(durations) or budget <= 0:
        return already_committed_makespan
    
    # Get the DP solver for this depth (remaining items from depth onwards)
    # dp_solvers[d] contains solver for items from depth d to n-1
    dp_solver = dp_solvers[depth]
    
    # Extract remaining items
    remaining_durations = durations[depth:]
    remaining_prices = prices[depth:]
    
    # Solve Max-LPT for remaining items
    maxlpt_contribution = solve_max_lpt(
        remaining_durations, 
        remaining_prices, 
        budget, 
        m, 
        dp_solver
    )
    
    # Total upper bound = committed makespan + Max-LPT contribution
    return already_committed_makespan + maxlpt_contribution


def precompute_maxlpt_dp_tables(durations: List[float], prices: List[float], 
                                max_budget: int) -> List[UnboundedKnapsackDP]:
    """Precompute DP tables for all possible depths in the BnB tree.
    
    For efficient bound computation during BnB search, we precompute
    a separate DP table for each depth level. The table at depth d
    contains the unbounded knapsack solution for items d, d+1, ..., n-1.
    
    Args:
        durations: List of job durations
        prices: List of job prices
        max_budget: Maximum budget to consider
        
    Returns:
        List of UnboundedKnapsackDP solvers, one for each depth
        
    Time complexity: O(n^2 * B)
    Space complexity: O(n^2 * B)
    """
    n = len(durations)
    dp_solvers = []
    
    for depth in range(n):
        # For depth d, we need DP table for items d, d+1, ..., n-1
        remaining_durations = durations[depth:]
        remaining_prices = prices[depth:]
        
        solver = UnboundedKnapsackDP(remaining_durations, remaining_prices, max_budget)
        dp_solvers.append(solver)
    
    return dp_solvers


# ============================================================================
# Testing and Validation
# ============================================================================


def generate_test_instances(num_instances: int = 20) -> List[Dict]:
    """Generate diverse unbounded knapsack test instances.
    
    Creates test cases with varying characteristics:
    - Different item counts (2-10 items)
    - Different budgets
    - Different value/weight ratios
    - Edge cases (all same ratio, highly skewed, etc.)
    
    Args:
        num_instances: Number of test instances to generate
        
    Returns:
        List of dictionaries with 'durations', 'prices', 'capacity' keys
    """
    instances = []
    random.seed(42)  # For reproducibility
    
    # Test 1-5: Small instances with different numbers of items
    for n_items in range(2, 7):
        capacity = 30 + n_items * 5
        durations = [random.randint(5, 20) for _ in range(n_items)]
        prices = [random.randint(2, 8) for _ in range(n_items)]
        instances.append({
            'durations': durations,
            'prices': prices,
            'capacity': capacity,
            'name': f'Random {n_items} items, capacity {capacity}'
        })
    
    # Test 6-10: Large budgets
    for trial in range(5):
        n_items = random.randint(3, 6)
        capacity = 100 + trial * 50
        durations = [random.randint(5, 15) for _ in range(n_items)]
        prices = [random.randint(1, 5) for _ in range(n_items)]
        instances.append({
            'durations': durations,
            'prices': prices,
            'capacity': capacity,
            'name': f'Large budget {capacity}, {n_items} items (trial {trial})'
        })
    
    # Test 11: Uniform ratios (all items equally attractive)
    durations = [4, 6, 8, 10]
    prices = [2, 3, 4, 5]
    capacity = 20
    instances.append({
        'durations': durations,
        'prices': prices,
        'capacity': capacity,
        'name': 'Uniform ratios (2.0 for all items)'
    })
    
    # Test 12: Highly skewed ratios
    durations = [100, 5, 3, 2]
    prices = [2, 10, 15, 20]
    capacity = 30
    instances.append({
        'durations': durations,
        'prices': prices,
        'capacity': capacity,
        'name': 'Highly skewed ratios (50.0, 0.5, 0.2, 0.1)'
    })
    
    # Test 13: Single item
    durations = [10]
    prices = [3]
    capacity = 20
    instances.append({
        'durations': durations,
        'prices': prices,
        'capacity': capacity,
        'name': 'Single item'
    })
    
    # Test 14: Expensive items (tight budget)
    durations = [10, 15, 20]
    prices = [8, 9, 10]
    capacity = 20
    instances.append({
        'durations': durations,
        'prices': prices,
        'capacity': capacity,
        'name': 'Expensive items with tight budget'
    })
    
    # Test 15: Very cheap items (loose budget)
    durations = [1, 2, 3, 4]
    prices = [1, 1, 1, 1]
    capacity = 100
    instances.append({
        'durations': durations,
        'prices': prices,
        'capacity': capacity,
        'name': 'Very cheap items with loose budget'
    })
    
    # Test 16: User's example
    durations = [1, 30]
    prices = [1, 50]
    capacity = 100
    instances.append({
        'durations': durations,
        'prices': prices,
        'capacity': capacity,
        'name': "User's example: [1,30] prices [1,50] capacity 100"
    })
    
    # Test 17-20: Additional random instances
    for trial in range(4):
        n_items = random.randint(3, 8)
        capacity = random.randint(20, 50)
        durations = [random.randint(1, 25) for _ in range(n_items)]
        prices = [random.randint(1, 10) for _ in range(n_items)]
        instances.append({
            'durations': durations,
            'prices': prices,
            'capacity': capacity,
            'name': f'Random instance {trial + 1} ({n_items} items, capacity {capacity})'
        })
    
    return instances[:num_instances]


def compare_implementations(durations: List[float], prices: List[float], 
                            capacity: int) -> Tuple[float, float, bool]:
    """Compare the two unbounded knapsack implementations.
    
    Args:
        durations: Item values
        prices: Item weights
        capacity: Knapsack capacity
        
    Returns:
        (reference_value, my_value, match) - Results and whether they match
    """
    # Reference implementation
    ref_value = unbounded_knapsack_reference(capacity, durations, prices)
    
    # My implementation
    solver = UnboundedKnapsackDP(durations, prices, capacity)
    my_value = solver.query(len(durations), capacity)
    
    # Check if they match (within floating point tolerance)
    match = abs(ref_value - my_value) < 1e-6
    
    return ref_value, my_value, match


if __name__ == "__main__":
    print("=" * 80)
    print("COMPARING UNBOUNDED KNAPSACK IMPLEMENTATIONS")
    print("=" * 80)
    
    # Generate test instances
    print("\nGenerating 20+ test instances...")
    instances = generate_test_instances(20)
    print(f"Generated {len(instances)} test instances")
    
    # Run comparison tests
    print("\n" + "=" * 80)
    print("RUNNING COMPARISON TESTS")
    print("=" * 80)
    
    results = []
    all_match = True
    
    for idx, instance in enumerate(instances, 1):
        durations = instance['durations']
        prices = instance['prices']
        capacity = instance['capacity']
        name = instance['name']
        
        try:
            ref_val, my_val, match = compare_implementations(durations, prices, capacity)
            results.append({
                'idx': idx,
                'name': name,
                'n_items': len(durations),
                'capacity': capacity,
                'ref_value': ref_val,
                'my_value': my_val,
                'match': match,
                'error': abs(ref_val - my_val)
            })
            
            if not match:
                all_match = False
            
            # Print results for this instance
            status = "✓ MATCH" if match else "✗ MISMATCH"
            print(f"\nTest {idx:2d}: {status}")
            print(f"  Name:     {name}")
            print(f"  Items:    {durations}")
            print(f"  Prices:   {prices}")
            print(f"  Capacity: {capacity}")
            print(f"  Reference: {ref_val}")
            print(f"  My impl:   {my_val}")
            if not match:
                print(f"  ERROR:     {abs(ref_val - my_val)}")
        
        except Exception as e:
            print(f"\nTest {idx:2d}: ✗ EXCEPTION")
            print(f"  Name: {name}")
            print(f"  Error: {e}")
            all_match = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'#':<4} {'Name':<40} {'Items':<8} {'Cap':<8} {'Ref':<12} {'My':<12} {'Match':<8}")
    print("-" * 92)
    
    for r in results:
        match_str = "✓" if r['match'] else "✗"
        print(f"{r['idx']:<4} {r['name']:<40} {r['n_items']:<8} {r['capacity']:<8} "
              f"{r['ref_value']:<12.2f} {r['my_value']:<12.2f} {match_str:<8}")
    
    # Final verdict
    print("\n" + "=" * 80)
    if all_match:
        print("✓ ALL TESTS PASSED - Implementations match on all instances!")
        print("=" * 80)
    else:
        print("✗ SOME TESTS FAILED - Implementations differ on some instances")
        print("=" * 80)
        print("\nFailed tests:")
        for r in results:
            if not r['match']:
                print(f"  Test {r['idx']}: {r['name']}")
                print(f"    Reference: {r['ref_value']}, Mine: {r['my_value']}, Error: {r['error']}")
    
    # Additional tests for maxlpt_bound functionality
    print("\n" + "=" * 80)
    print("ADDITIONAL TESTS: Max-LPT AND PREPROCESSING")
    print("=" * 80)
    
    # Test 1: Basic Unbounded Knapsack DP
    print("\n" + "=" * 70)
    print("TEST 1: Unbounded Knapsack DP")
    print("=" * 70)
    durations = [10, 40, 30, 50]
    prices = [5, 4, 6, 3]
    budget = 10
    print(f"Durations: {durations}")
    print(f"Prices:    {prices}")
    print(f"Budget:    {budget}")
    
    solver = UnboundedKnapsackDP(durations, prices, budget)
    result = solver.reconstruct(4, 10)
    print(f"\nResult:")
    print(f"  Max value: {result['max_value']}")
    print(f"  Counts:    {result['counts']}")
    print(f"  Total cost: {sum(p * c for p, c in zip(prices, result['counts']))}")
    
    # Test 2: Max-LPT with simple instance
    print("\n" + "=" * 70)
    print("TEST 2: Max-LPT Algorithm")
    print("=" * 70)
    durations = [4, 6, 8, 10]
    prices = [2, 3, 4, 5]
    budget = 12
    m = 2
    print(f"Durations: {durations}")
    print(f"Prices:    {prices}")
    print(f"Budget:    {budget}")
    print(f"Machines:  {m}")
    
    dp_solver = UnboundedKnapsackDP(durations, prices, budget)
    maxlpt_value = solve_max_lpt(durations, prices, budget, m, dp_solver)
    print(f"\nMax-LPT makespan: {maxlpt_value}")
    
    # Test 3: Precompute DP tables for all depths
    print("\n" + "=" * 70)
    print("TEST 3: Precompute DP Tables for All Depths")
    print("=" * 70)
    durations = [4, 6, 8, 10]
    prices = [2, 3, 4, 5]
    max_budget = 12
    print(f"Durations:   {durations}")
    print(f"Prices:      {prices}")
    print(f"Max Budget:  {max_budget}")
    print(f"Num Items:   {len(durations)}")
    
    dp_solvers = precompute_maxlpt_dp_tables(durations, prices, max_budget)
    print(f"\nPrecomputed {len(dp_solvers)} DP tables (one per depth)")
    
    # Test queries at different depths
    for depth in range(len(durations)):
        remaining_items = len(durations) - depth
        value = dp_solvers[depth].query(remaining_items, max_budget)
        print(f"  Depth {depth} ({remaining_items} remaining items): max value = {value}")
    
    # Test 4: Compute bound at specific node
    print("\n" + "=" * 70)
    print("TEST 4: Compute Max-LPT Bound at Node")
    print("=" * 70)
    depth = 1
    remaining_budget = 8
    already_committed = 5.0
    print(f"Current depth:          {depth}")
    print(f"Remaining budget:       {remaining_budget}")
    print(f"Already committed:      {already_committed}")
    
    bound = compute_maxlpt_bound(
        durations, prices, remaining_budget, m, depth,
        dp_solvers, already_committed
    )
    print(f"\nUpper bound: {bound}")
    
    # Test 5: Compare with simple ceiling bound
    print("\n" + "=" * 70)
    print("TEST 5: Uniform Ratios Instance")
    print("=" * 70)
    durations = [4, 6, 8, 10]
    prices = [2, 3, 4, 5]
    budget = 12
    m = 2
    print(f"Durations: {durations}")
    print(f"Prices:    {prices}")
    print(f"Budget:    {budget}")
    print(f"Machines:  {m}")
    print(f"Ratios:    {[d/p for d, p in zip(durations, prices)]}")
    
    dp_solver = UnboundedKnapsackDP(durations, prices, budget)
    maxlpt_value = solve_max_lpt(durations, prices, budget, m, dp_solver)
    print(f"\nMax-LPT makespan: {maxlpt_value}")
    
    # Simple upper bound: maximize sum of durations
    simple_result = dp_solver.reconstruct(4, budget)
    print(f"Simple bound (max duration): {simple_result['max_value']}")
    print(f"  (This ignores the ceiling effect)")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
