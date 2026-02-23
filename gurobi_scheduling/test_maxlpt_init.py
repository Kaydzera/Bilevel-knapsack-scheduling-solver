"""Test the new Max-LPT initialization function."""

from models import MainProblem
from bnb import run_bnb_classic

# Simple test instance
prices = [2, 3, 4, 5]
durations = [4, 6, 8, 10]
m = 2
budget = 12

problem = MainProblem(prices, durations, m, budget)

print("=" * 70)
print("Testing with ceiling bound:")
print("=" * 70)
result_ceil = run_bnb_classic(problem, max_nodes=10000, verbose=True, 
                              enable_logging=False, bound_type='ceiling')
print(f"\nResult:")
print(f"  Makespan: {result_ceil['best_obj']}")
print(f"  Selection: {result_ceil['best_selection']}")
print(f"  Nodes explored: {result_ceil['nodes_explored']}")
print(f"  Proven optimal: {result_ceil.get('proven_optimal', False)}")

print("\n" + "=" * 70)
print("Testing with maxlpt bound:")
print("=" * 70)
result_maxlpt = run_bnb_classic(problem, max_nodes=10000, verbose=True, 
                                enable_logging=False, bound_type='maxlpt')
print(f"\nResult:")
print(f"  Makespan: {result_maxlpt['best_obj']}")
print(f"  Selection: {result_maxlpt['best_selection']}")
print(f"  Nodes explored: {result_maxlpt['nodes_explored']}")  
print(f"  Proven optimal: {result_maxlpt.get('proven_optimal', False)}")

# Verify both give same result
print("\n" + "=" * 70)
print("Verification:")
print("=" * 70)
if result_ceil['best_obj'] == result_maxlpt['best_obj']:
    print("✓ Both methods found the same makespan")
else:
    print(f"✗ Different makespans: ceiling={result_ceil['best_obj']}, maxlpt={result_maxlpt['best_obj']}")

if result_ceil['best_selection'] == result_maxlpt['best_selection']:
    print("✓ Both methods found the same selection")
else:
    print(f"✗ Different selections")
    print(f"  Ceiling: {result_ceil['best_selection']}")
    print(f"  MaxLPT:  {result_maxlpt['best_selection']}")

print(f"\nNodes explored: ceiling={result_ceil['nodes_explored']}, maxlpt={result_maxlpt['nodes_explored']}")
if result_maxlpt['nodes_explored'] < result_ceil['nodes_explored']:
    reduction = (1 - result_maxlpt['nodes_explored'] / result_ceil['nodes_explored']) * 100
    print(f"✓ MaxLPT explored {reduction:.1f}% fewer nodes")
