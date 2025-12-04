"""Quick test to verify CeilKnapsackSolver integration into BnB."""

from models import MainProblem
from bnb import run_bnb_classic

# Test with a small instance
print("="*70)
print("Testing BnB with CeilKnapsackSolver Integration")
print("="*70)

# Small test case: Uniform ratios
prices = [2, 3, 4, 5]
durations = [10, 20, 30, 40]
m = 2
budget = 12

print(f"\nTest Instance:")
print(f"  Items: {list(zip(durations, prices))}")
print(f"  Machines: {m}")
print(f"  Budget: {budget}")
print()

problem = MainProblem(prices, durations, m, budget)

try:
    result = run_bnb_classic(
        problem, 
        max_nodes=1000, 
        verbose=True,
        enable_logging=False  # Disable file logging for quick test
    )
    
    print("\n" + "="*70)
    print("TEST PASSED!")
    print("="*70)
    print(f"Best makespan: {result['best_obj']}")
    print(f"Best selection: {result['best_selection']}")
    print(f"Nodes explored: {result['nodes_explored']}")
    print("\nCeilKnapsackSolver successfully integrated and verified!")
    print("All bounds matched between DP and IP methods.")
    
except ValueError as e:
    print("\n" + "="*70)
    print("TEST FAILED!")
    print("="*70)
    print(f"Error: {e}")
    print("\nThere is a mismatch between CeilKnapsackSolver and IP bound.")
except Exception as e:
    print("\n" + "="*70)
    print("TEST ERROR!")
    print("="*70)
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
