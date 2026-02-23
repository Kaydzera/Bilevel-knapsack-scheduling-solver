"""Test Max-LPT initialization with a harder instance."""

from models import MainProblem
from bnb import run_bnb_classic

# Harder test instance where Max-LPT won't immediately prove optimality
prices = [2, 2, 8, 13, 15, 2, 2, 6]
durations = [18, 14, 7, 12, 14, 10, 13, 9]
m = 4
budget = 100

problem = MainProblem(prices, durations, m, budget)

print("=" * 70)
print("Testing harder instance with maxlpt bound:")
print("=" * 70)
print(f"Prices:    {prices}")
print(f"Durations: {durations}")
print(f"Machines:  {m}")
print(f"Budget:    {budget}")
print()

result = run_bnb_classic(problem, max_nodes=50000, verbose=True, 
                        enable_logging=False, bound_type='maxlpt')

print(f"\nResult:")
print(f"  Makespan: {result['best_obj']}")
print(f"  Selection: {result['best_selection']}")
print(f"  Nodes explored: {result['nodes_explored']}")
print(f"  Proven optimal: {result.get('proven_optimal', False)}")
print(f"  Runtime: {result.get('runtime', 'N/A')}")
