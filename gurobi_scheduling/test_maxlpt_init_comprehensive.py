"""Test Max-LPT with an instance that requires BnB search."""

from models import MainProblem
from bnb import run_bnb_classic
import random

# Create an instance with specific characteristics to avoid immediate optimality
# Use items with varying ratios that might create a gap between Max-LPT and optimal
random.seed(42)

prices = [3, 5, 7, 11]
durations = [10, 15, 20, 28]
m = 3
budget = 30

problem = MainProblem(prices, durations, m, budget)

print("=" * 70)
print("Testing with instance designed for BnB search:")
print("=" * 70)
print(f"Prices:    {prices}")
print(f"Durations: {durations}")
print(f"Machines:  {m}")
print(f"Budget:    {budget}")
print(f"Ratios:    {[d/p for d, p in zip(durations, prices)]}")
print()

# Test with ceiling bound
print("=" * 70)
print("CEILING BOUND:")
print("=" * 70)
result_ceil = run_bnb_classic(problem, max_nodes=50000, verbose=True, 
                               enable_logging=False, bound_type='ceiling')
print(f"\nResult:")
print(f"  Makespan: {result_ceil['best_obj']}")
print(f"  Selection: {result_ceil['best_selection']}")
print(f"  Nodes explored: {result_ceil['nodes_explored']}")
print(f"  Proven optimal: {result_ceil.get('proven_optimal', False)}")

# Test with maxlpt bound
print("\n" + "=" * 70)
print("MAXLPT BOUND:")
print("=" * 70)
result_maxlpt = run_bnb_classic(problem, max_nodes=50000, verbose=True, 
                                enable_logging=False, bound_type='maxlpt')
print(f"\nResult:")
print(f"  Makespan: {result_maxlpt['best_obj']}")
print(f"  Selection: {result_maxlpt['best_selection']}")
print(f"  Nodes explored: {result_maxlpt['nodes_explored']}")
print(f"  Proven optimal: {result_maxlpt.get('proven_optimal', False)}")

# Compare
print("\n" + "=" * 70)
print("COMPARISON:")
print("=" * 70)
print(f"Same makespan: {result_ceil['best_obj'] == result_maxlpt['best_obj']}")
print(f"Ceiling nodes: {result_ceil['nodes_explored']}")
print(f"MaxLPT nodes:  {result_maxlpt['nodes_explored']}")
if result_ceil['nodes_explored'] > 0 and result_maxlpt['nodes_explored'] > 0:
    reduction = (1 - result_maxlpt['nodes_explored'] / result_ceil['nodes_explored']) * 100
    print(f"Node reduction: {reduction:.1f}%")
