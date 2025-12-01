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

IN each row of the dp table, a new item is considered. For each capacity w, we can either not take the item, or take k copies of it (0 <= k <= max_counts[i-1]) if it fits. The dp table is updated accordingly.
If we start the table with the last item that will be explored in the bnb tree, we can read off the optimal selection directly for the bound that is the modified knapsack objective by only reading items uop to that depth.
We also have to make sure to account for the fact that the objective uses ceil(x[i]/m) instead of x[i], so we need to adjust the value contribution accordingly when filling the dp table.
That is, when we consider taking k copies of item i-1, the value contribution should be duration[i-1] * ceil(k/m) instead of just duration[i-1] * k.
We can achieve this by adding, for each item type i-1, an additional value contribution of duration[i-1] whenever k is a multiple of m.

class KnapsackCeilSolver:
    def __init__(self):
        self.dp_table = None
        self.last_item = None
    
    def solve(self, values, weights, capacity, m):
        """Solve modified knapsack problem with ceil objective.
        
        Args:
            values: List of item values (durations)
            weights: List of item weights (prices)
            capacity: Maximum weight capacity (budget)
            m: Parameter for ceiling function in objective, resembling number of machines
            
        Returns:
            dict with:
                - max_value: Maximum achievable value
                - counts: List of counts for each item type
        """
        n = len(values)
        
        # Create DP table: dp[i][w] = max value using items 0..i-1 with capacity w
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        last_item = [[-1 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Fill DP table
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                # Option 1: Don't take item i-1
                dp[i][w] = dp[i-1][w]
                
                # Option 2: Take k copies of item i-1 (if they fit)
                max_k = w // weights[i-1]
                for k in range(1, max_k + 1):
                    if k * weights[i-1] <= w:
                        # Calculate value contribution with ceiling
                        value_contribution = values[i-1] * ((k + m - 1) // m)
                        new_value = dp[i-1][w - k * weights[i-1]] + value_contribution
                        if new_value > dp[i][w]:
                            dp[i][w] = new_value
                            last_item[i][w] = k
        
        max_value = dp[n][capacity]
        
        # Backtrack to find item counts
        counts = [0] * n
        w = capacity
        for i in range(n, 0, -1):
            k = last_item[i][w]
            if k != -1:
                counts[i-1] = k
                w -= k * weights[i-1]
        
        return {
            'max_value': max_value,


'''