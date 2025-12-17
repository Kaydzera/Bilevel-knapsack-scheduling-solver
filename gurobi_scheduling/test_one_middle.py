"""Test a single medium instance with full logging and enumeration verification."""
import sys
sys.path.insert(0, 'c:\\Users\\oleda\\.vscode\\Solving stuff with Gurobi\\gurobi_scheduling')

from test_middle import test_instances, run_instance

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING ONE MEDIUM INSTANCE WITH FULL LOGGING")
    print("=" * 70)
    
    # Run the first instance (Medium Baseline) with logging and enumeration
    result = run_instance(test_instances[0], use_enumeration=True, enable_logging=True)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)
    print(f"Result: makespan={result['best_obj']:.1f}, nodes={result['nodes_explored']}")
    print("Check logs/test_middle/ for detailed log files")
