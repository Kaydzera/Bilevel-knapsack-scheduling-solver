"""Debug script to trace through CeilKnapsackSolver reconstruction"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from knapsack_dp import CeilKnapsackSolver

# Simple test case: Item 0 only, budget=12, m=1
costs = [2]
durations = [10]
max_budget = 12
m = 1

print("=" * 80)
print("DEBUG: CeilKnapsackSolver Reconstruction")
print("=" * 80)
print(f"Item 0: cost={costs[0]}, duration={durations[0]}")
print(f"Budget: {max_budget}")
print(f"m: {m}")
print()

solver = CeilKnapsackSolver(costs, durations, m, max_budget)

print("Groups created:")
for idx, group in enumerate(solver.groups):
    val, wt, item_idx, typ = group
    print(f"  Group {idx}: value={val}, cost={wt}, item={item_idx}, type={typ}")
print()

print("item_to_groups mapping:")
for i, groups in enumerate(solver.item_to_groups):
    print(f"  Item {i}: group indices {groups}")
print()

# Check DP table
print("DP table for item 0 (first 13 budgets):")
for b in range(13):
    print(f"  dp[1][{b:2d}] = {solver.dp_prefix[1][b]}")
print()

# Reconstruct
result = solver.reconstruct(1, 12)
print(f"Reconstructed solution:")
print(f"  Max value: {result['max_value']}")
print(f"  Breakdown: {result['breakdown'][0]}")
print()

# Manually check what should be selected
print("Manual verification:")
print("  Budget=12, cost per item=2 → can buy 6 items")
print("  Expected: 6 items × 10 = 60 value")
print(f"  Got: {result['max_value']} value")
print()

# Check the take array for budget=12
print("Take array for item 0, budget=12:")
groups = solver.item_to_groups[0]
for g_idx, group in enumerate(groups):
    val, wt, _, typ = solver.groups[group]
    take_val = solver.take[1][g_idx][12]
    print(f"  Group {g_idx} (type={typ}, cost={wt}): take={take_val}")
print()

# Trace through reconstruction manually
print("Manual reconstruction trace:")
b = 12
selected_groups = []
for g_idx, group in enumerate(groups):
    if solver.take[1][g_idx][b] == 1:
        val, wt, _, typ = solver.groups[group]
        selected_groups.append((g_idx, typ, wt, b))
        print(f"  Selected group {g_idx} (type={typ}, cost={wt}) at budget={b}")
        b -= wt

print(f"  Final remaining budget: {b}")
print(f"  Total groups selected: {len(selected_groups)}")
