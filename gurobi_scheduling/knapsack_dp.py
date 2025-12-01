"""Dynamic Programming solution for the 0/1 Knapsack Problem.

This module implements a classic dynamic programming approach to solve
the knapsack problem where each item can be selected at most once.
"""


def knapsack_01(values, weights, capacity):
    """Solve 0/1 knapsack problem using dynamic programming.
    
    Each item can be selected at most once. Maximizes total value
    subject to weight constraint.
    
    Args:
        values: List of item values (profit/utility)
        weights: List of item weights (cost)
        capacity: Maximum weight capacity
        
    Returns:
        dict with:
            - max_value: Maximum achievable value
            - selected: Binary list indicating which items are selected
    """
    n = len(values)
    
    # Create DP table: dp[i][w] = max value using items 0..i-1 with capacity w
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Option 1: Don't take item i-1
            dp[i][w] = dp[i-1][w]
            
            # Option 2: Take item i-1 (if it fits)
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
                #we either take it and subtract its weight from capacity and add its value
                #  or leave it
    
    max_value = dp[n][capacity]
    
    # Backtrack to find which items were selected
    selected = [0] * n
    w = capacity
    for i in range(n, 0, -1):
        # If value came from including item i-1
        if dp[i][w] != dp[i-1][w]:
            selected[i-1] = 1
            w -= weights[i-1]
    
    return {
        'max_value': max_value,
        'selected': selected
    }


def knapsack_unbounded(values, weights, capacity):
    """Solve unbounded knapsack problem using dynamic programming.
    
    Each item can be selected multiple times. Maximizes total value
    subject to weight constraint.
    
    Args:
        values: List of item values (profit/utility)
        weights: List of item weights (cost)
        capacity: Maximum weight capacity
        
    Returns:
        dict with:
            - max_value: Maximum achievable value
            - counts: List of counts for each item type
    """
    n = len(values)
    
    # Create DP table: dp[w] = max value with capacity w
    dp = [0] * (capacity + 1)
    # Track which item was last added to reach each capacity
    last_item = [-1] * (capacity + 1)
    
    # Fill DP table
    for w in range(1, capacity + 1):
        for i in range(n):
            if weights[i] <= w:
                new_value = dp[w - weights[i]] + values[i]
                if new_value > dp[w]:
                    dp[w] = new_value
                    last_item[w] = i
    
    max_value = dp[capacity]
    
    # Backtrack to find item counts
    counts = [0] * n
    w = capacity
    while w > 0 and last_item[w] != -1:
        i = last_item[w]
        counts[i] += 1
        w -= weights[i]
    
    return {
        'max_value': max_value,
        'counts': counts
    }


def knapsack_bounded(values, weights, max_counts, capacity):
    """Solve bounded knapsack problem using dynamic programming.
    
    Each item can be selected up to a specified maximum count.
    Maximizes total value subject to weight constraint.
    
    Args:
        values: List of item values (profit/utility)
        weights: List of item weights (cost)
        max_counts: List of maximum counts for each item type
        capacity: Maximum weight capacity
        
    Returns:
        dict with:
            - max_value: Maximum achievable value
            - counts: List of counts for each item type
    """
    n = len(values)
    
    # Create DP table: dp[i][w] = max value using items 0..i-1 with capacity w
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Try taking 0, 1, 2, ..., max_counts[i-1] copies of item i-1
            max_k = min(max_counts[i-1], w // weights[i-1])
            for k in range(max_k + 1):
                if k * weights[i-1] <= w:
                    dp[i][w] = max(dp[i][w], 
                                  dp[i-1][w - k * weights[i-1]] + k * values[i-1])
    
    max_value = dp[n][capacity]
    
    # Backtrack to find item counts
    counts = [0] * n
    w = capacity
    for i in range(n, 0, -1):
        # Find how many of item i-1 were taken
        for k in range(max_counts[i-1] + 1):
            if k * weights[i-1] <= w:
                if dp[i][w] == dp[i-1][w - k * weights[i-1]] + k * values[i-1]:
                    counts[i-1] = k
                    w -= k * weights[i-1]
                    break
    
    return {
        'max_value': max_value,
        'counts': counts
    }






###########################################################################################################################
'''
The following is a knapsack solver that maximizes sum(duration[i] * ceil(x[i]/m)) subject to budget constraint using gurobi.
'''
import math
import time
from collections import deque
from typing import Optional

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    gp = None
    GRB = None

from solvers import solve_scheduling, solve_leader_knapsack, solve_scheduling_readable
from models import MainProblem, ProblemNode
from logger import BnBLogger



def compute_ip_bound_exact(problem: MainProblem, node: ProblemNode, time_limit=2.0):
    """Compute exact integer-program upper bound using grouping.
    
    Maximizes sum(duration[i] * ceil(x[i]/m)) subject to budget constraint
    on remaining items. Uses binary variables for copies and group variables
    to model ceil(x[i]/m).
    
    Args:
        problem: MainProblem instance
        node: Current ProblemNode in the search tree
        time_limit: Time limit for Gurobi solver
        
    Returns:
        tuple: (bound, selection) - Objective value and selection for remaining items
        
    Raises:
        ImportError: If Gurobi is not available
    """
    if gp is None:
        raise ImportError("Gurobi is not available. Please install gurobipy to use this function.")

    # Extract remaining items and budget
    remaining_items = problem.items[node.depth:]
    rem_budget = node.remaining_budget
    m = problem.machines
    
    bound, selection = compute_ip_bound_direct(problem.items, node.depth, rem_budget, m, time_limit)
    return bound, selection


# TODO: Review and understand this code line by line
def compute_ip_bound_direct(items, depth, rem_budget, m, time_limit=2.0):
    """Compute exact integer-program upper bound using grouping (direct version).
    
    This is the same as compute_ip_bound_exact but takes direct parameters
    instead of problem and node objects, making it easier to test.
    
    Maximizes sum(duration[i] * ceil(x[i]/m)) subject to budget constraint
    on remaining items (from depth onwards). Uses binary variables for copies
    and group variables to model ceil(x[i]/m).
    
    Args:
        items: Full list of items (with .duration and .price attributes)
        depth: Current depth in the search tree
        rem_budget: Remaining budget available
        m: Number of machines
        time_limit: Time limit for Gurobi solver
        
    Returns:
        float: Objective value (sum duration * ceil(x_i/m))
        
    Raises:
        ImportError: If Gurobi is not available
    """
    if gp is None:
        raise ImportError("Gurobi is not available. Please install gurobipy to use this function.")

    # Extract remaining items from depth+1 onwards (truly undecided items)
    # The item at depth is currently being decided and should be in already_committed
    remaining_items = items[depth+1:]

    model = gp.Model("exact_ip_bound")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)

    # For each item type i, compute max copies allowed
    K = []
    for it in remaining_items:
        if it.price <= 0:
            # Infinite copies possible
            K.append(0)
        else:
            K.append(rem_budget // it.price)

    # Decision variables:
    # u_{i,k} = 1 if copy k of item i is selected
    # w_{i,g} = 1 if at least one copy in group g of item i is selected
    u = {}
    w = {}
    for i, Ki in enumerate(K):
        # G is the number of groups (ceil(Ki / m))
        # Each group represents a batch of up to m copies
        G = (Ki + m - 1) // m if Ki > 0 else 0
        for k in range(1, Ki + 1):
            u[i, k] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{k}")
        for g in range(1, G + 1):
            w[i, g] = model.addVar(vtype=GRB.BINARY, name=f"w_{i}_{g}")

    model.update()

    # Link group binaries w and copy binaries u
    for i, Ki in enumerate(K):
        G = (Ki + m - 1) // m if Ki > 0 else 0
        for g in range(1, G + 1):
            # Copies in this group
            first = (g - 1) * m + 1
            last = min(g * m, Ki)
            if first > last:
                # Empty group
                model.addConstr(w[i, g] == 0)
                continue
            # If any u in group is 1, then w must be 1
            model.addConstr(gp.quicksum(u[i, k] for k in range(first, last + 1)) >= w[i, g], name=f"w_lower_{i}_{g}")
            # Each u can only be 1 if w is 1
            for k in range(first, last + 1):
                model.addConstr(u[i, k] <= w[i, g], name=f"u_le_w_{i}_{g}_{k}")
        
        # Force sequential packing: group g can only be used if group g-1 is full
        # This ensures ceil(x[i]/m) = sum_g w[i,g]
        for g in range(2, G + 1):
            # If group g is used (w[i,g]=1), then all copies in group g-1 must be selected
            first_prev = (g - 2) * m + 1
            last_prev = min((g - 1) * m, Ki)
            if first_prev <= last_prev:
                # w[i,g] <= (1/m) * sum of all u in previous group
                # Equivalently: m * w[i,g] <= sum of u in previous group
                model.addConstr(
                    m * w[i, g] <= gp.quicksum(u[i, k] for k in range(first_prev, last_prev + 1)),
                    name=f"seq_pack_{i}_{g}"
                )

    # Budget constraint
    budget_expr = gp.quicksum(remaining_items[i].price * u[i, k] for (i, k) in u.keys())
    model.addConstr(budget_expr <= rem_budget, name="budget")

    # Objective: maximize sum_i duration[i] * sum_g w[i,g]
    # (sum_g w[i,g] represents ceil(x[i]/m))
    obj = gp.quicksum(remaining_items[i].duration * w[i, g] for (i, g) in w.keys())
    model.setObjective(obj, GRB.MAXIMIZE)

    model.optimize()







if __name__ == "__main__":
    print("=" * 70)
    print("TESTING KNAPSACK DYNAMIC PROGRAMMING SOLUTIONS")
    print("=" * 70)
    
    # Test 1: Classic 0/1 Knapsack
    print("\n" + "=" * 70)
    print("TEST 1: 0/1 Knapsack Problem")
    print("=" * 70)
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50
    print(f"Values:   {values}")
    print(f"Weights:  {weights}")
    print(f"Capacity: {capacity}")
    
    result = knapsack_01(values, weights, capacity)
    print(f"\nResult:")
    print(f"  Max value: {result['max_value']}")
    print(f"  Selected:  {result['selected']}")
    print(f"  Total weight: {sum(w * s for w, s in zip(weights, result['selected']))}")
    
    # Test 2: Another 0/1 instance
    print("\n" + "=" * 70)
    print("TEST 2: 0/1 Knapsack - More items")
    print("=" * 70)
    values = [10, 40, 30, 50]
    weights = [5, 4, 6, 3]
    capacity = 10
    print(f"Values:   {values}")
    print(f"Weights:  {weights}")
    print(f"Capacity: {capacity}")
    
    result = knapsack_01(values, weights, capacity)
    print(f"\nResult:")
    print(f"  Max value: {result['max_value']}")
    print(f"  Selected:  {result['selected']}")
    print(f"  Total weight: {sum(w * s for w, s in zip(weights, result['selected']))}")
    
    # Test 3: Unbounded Knapsack
    print("\n" + "=" * 70)
    print("TEST 3: Unbounded Knapsack Problem")
    print("=" * 70)
    values = [10, 40, 30, 50]
    weights = [5, 4, 6, 3]
    capacity = 10
    print(f"Values:   {values}")
    print(f"Weights:  {weights}")
    print(f"Capacity: {capacity}")
    print("(Each item can be selected multiple times)")
    
    result = knapsack_unbounded(values, weights, capacity)
    print(f"\nResult:")
    print(f"  Max value: {result['max_value']}")
    print(f"  Counts:    {result['counts']}")
    print(f"  Total weight: {sum(w * c for w, c in zip(weights, result['counts']))}")
    
    # Test 4: Bounded Knapsack
    print("\n" + "=" * 70)
    print("TEST 4: Bounded Knapsack Problem")
    print("=" * 70)
    values = [10, 40, 30, 50]
    weights = [5, 4, 6, 3]
    max_counts = [2, 1, 3, 2]
    capacity = 15
    print(f"Values:     {values}")
    print(f"Weights:    {weights}")
    print(f"Max counts: {max_counts}")
    print(f"Capacity:   {capacity}")
    
    result = knapsack_bounded(values, weights, max_counts, capacity)
    print(f"\nResult:")
    print(f"  Max value: {result['max_value']}")
    print(f"  Counts:    {result['counts']}")
    print(f"  Total weight: {sum(w * c for w, c in zip(weights, result['counts']))}")
    
    # Test 5: Uniform Ratios instance (from your examples)
    print("\n" + "=" * 70)
    print("TEST 5: Uniform Ratios (like bilevel problem)")
    print("=" * 70)
    durations = [4, 6, 8, 10]
    prices = [2, 3, 4, 5]
    budget = 12
    print(f"Durations (values): {durations}")
    print(f"Prices (weights):   {prices}")
    print(f"Budget (capacity):  {budget}")
    print("(Unbounded - maximize total duration)")
    
    result = knapsack_unbounded(durations, prices, budget)
    print(f"\nResult:")
    print(f"  Max total duration: {result['max_value']}")
    print(f"  Counts:             {result['counts']}")
    print(f"  Total cost: {sum(p * c for p, c in zip(prices, result['counts']))}")
    
    # Test 6: Small instance with fractional ratios
    print("\n" + "=" * 70)
    print("TEST 6: Items with different value/weight ratios")
    print("=" * 70)
    values = [24, 18, 18, 10]
    weights = [24, 10, 10, 7]
    capacity = 25
    print(f"Values:   {values}")
    print(f"Weights:  {weights}")
    print(f"Ratios:   {[v/w for v, w in zip(values, weights)]}")
    print(f"Capacity: {capacity}")
    
    result = knapsack_01(values, weights, capacity)
    print(f"\nResult (0/1):")
    print(f"  Max value: {result['max_value']}")
    print(f"  Selected:  {result['selected']}")
    print(f"  Total weight: {sum(w * s for w, s in zip(weights, result['selected']))}")
    
    result_unbounded = knapsack_unbounded(values, weights, capacity)
    print(f"\nResult (Unbounded):")
    print(f"  Max value: {result_unbounded['max_value']}")
    print(f"  Counts:    {result_unbounded['counts']}")
    print(f"  Total weight: {sum(w * c for w, c in zip(weights, result_unbounded['counts']))}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)




'''
I would now like to do the following: when solving Maximizes sum(duration[i] * ceil(x[i]/m)) subject to budget constraint, the objective only changes when ceil(x[i]/m) changes for some i.
Therefore, I would like to implement a knapsack solver that takes as input the values, weights, capacity, and also a parameter m (number of machines), and returns the optimal selection of items that maximizes sum(duration[i] * ceil(x[i]/m)) subject to the budget constraint.

In each row of the dp table, a new item is considered. For each capacity w, we can either not take the item, or take k copies of it (0 <= k <= max_counts[i-1]) if it fits
(here max_counts[i-1] is the maximum number of copies of item i-1 that can fit in capacity w so that a budget constraint is not violated). The dp table is updated accordingly.
If we start the table with the last item that will be explored in the bnb tree, we can read off the optimal selection directly for the bound that is the modified knapsack objective by only reading items up to that depth.
We also have to make sure to account for the fact that the objective uses ceil(x[i]/m) instead of x[i], so we need to adjust the value contribution accordingly when filling the dp table.
That is, when we consider taking k copies of item i-1, the value contribution should be duration[i-1] * ceil(k/m) instead of just duration[i-1] * k.
We can achieve this by adding, for each item type i-1, an additional value contribution of duration[i-1] whenever k is a multiple of m.
'''

class CeilKnapsackSolver:
    """
    Solves the modified knapsack problem for maximizing:
        sum_i duration[i] * ceil(x[i]/m)
    under a budget constraint:
        sum_i cost[i] * x[i] <= budget

    Approach:
    - For each item type i:
        * One "single-item" option: value = duration[i], cost = cost[i]
        * Multiple "package" options: each of size m, value = duration[i], cost = m*cost[i]
          (number of packages limited by budget)
    - Each option can be taken at most once (0-1 knapsack)
    - DP table is precomputed for all items and max_budget
    - Reconstruction gives per-item breakdown and total x[i]
    """

    def __init__(self, costs, durations, m, max_budget):
        self.costs = costs
        self.durations = durations
        self.m = m
        self.n = len(costs)
        self.max_budget = max_budget

        # Build the full list of 0-1 options (groups)
        # Each group = (value, weight, item_index, type)
        # type: "single" or "package"
        self.groups = []
        for i in range(self.n):
            # Single-item option
            if costs[i] <= max_budget:
                self.groups.append((durations[i], costs[i], i, "single"))

            # Package options
            max_packages = (max_budget // (m * costs[i]))
            for k in range(max_packages):
                self.groups.append((durations[i], m * costs[i], i, "package"))

        self.total_groups = len(self.groups)

        # DP table: dp[g][b] = max value using first g groups with budget b
        self.dp = [[0] * (max_budget + 1) for _ in range(self.total_groups + 1)]
        # Decision table: take[g][b] = True if group g-1 was taken
        self.take = [[False] * (max_budget + 1) for _ in range(self.total_groups + 1)]

        self._build_dp()

    def _build_dp(self):
        """Fill the DP table"""
        for g in range(1, self.total_groups + 1):
            val, wt, item_i, typ = self.groups[g - 1]
            for b in range(self.max_budget + 1):
                # Option 1: skip group
                self.dp[g][b] = self.dp[g - 1][b]
                # Option 2: take group if it fits
                if wt <= b:
                    new_val = self.dp[g - 1][b - wt] + val
                    if new_val > self.dp[g][b]:
                        self.dp[g][b] = new_val
                        self.take[g][b] = True

    def reconstruct(self, num_item_types, budget):
        """
        Reconstruct solution for first `num_item_types` items and given budget.
        Returns both breakdown and total x[i].
        """
        if num_item_types > self.n:
            raise ValueError("num_item_types exceeds total items")
        if budget > self.max_budget:
            raise ValueError("budget exceeds max_budget")

        # Allowed groups = groups belonging to the first num_item_types items
        allowed_groups = [g for g, (_, _, i, _) in enumerate(self.groups) if i < num_item_types]

        # Initialize breakdown
        breakdown = [{"single": 0, "packages": 0, "x_total": 0} for _ in range(num_item_types)]

        # Backtrack to determine which groups were taken
        b = budget
        g = self.total_groups
        while g > 0 and b >= 0:
            if g - 1 in allowed_groups and self.take[g][b]:
                val, wt, item_i, typ = self.groups[g - 1]
                if typ == "single":
                    breakdown[item_i]["single"] = 1
                else:  # "package"
                    breakdown[item_i]["packages"] += 1
                b -= wt
            g -= 1

        # Compute total x[i] for each item
        for i in range(num_item_types):
            breakdown[i]["x_total"] = breakdown[i]["single"] + breakdown[i]["packages"] * self.m

        max_value = self.dp[self.total_groups][budget]

        return {"max_value": max_value, "breakdown": breakdown}

    def query(self, num_item_types, budget):
        """Return only the max value (shortcut)"""
        if num_item_types > self.n or budget > self.max_budget:
            raise ValueError("Invalid query")
        return self.dp[self.total_groups][budget]

'''
Ex. Usage
costs = [3, 5, 10]
durations = [4, 7, 20]
m = 4
max_budget = 50

solver = CeilKnapsackSolver(costs, durations, m, max_budget)

# Reconstruct solution for all items and budget 30
result = solver.reconstruct(num_item_types=3, budget=30)
print("Max value:", result["max_value"])
for i, info in enumerate(result["breakdown"]):
    print(f"Item {i}: single={info['single']}, packages={info['packages']}, x_total={info['x_total']}")

# Query only max value
print("Max value for first 2 items, budget 20:", solver.reconstruct(2, 20)["max_value"])

'''