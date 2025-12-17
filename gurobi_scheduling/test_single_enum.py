"""
Test individual instances with enumeration
"""
import time
from models import Item
from bilevel_gurobi import solve_bilevel_simpler

# Instance 11: Large Scale Baseline
def test_instance_11():
    print("=" * 70)
    print("TESTING INSTANCE #11: Large Scale Baseline")
    print("=" * 70)
    
    items = [
        Item(name="A", duration=15, price=8),   # ratio: 1.88
        Item(name="B", duration=18, price=10),  # ratio: 1.80
        Item(name="C", duration=12, price=6),   # ratio: 2.00
        Item(name="D", duration=20, price=12),  # ratio: 1.67
        Item(name="E", duration=14, price=7),   # ratio: 2.00
        Item(name="F", duration=16, price=9),   # ratio: 1.78
        Item(name="G", duration=10, price=5),   # ratio: 2.00
    ]
    m = 6
    budget = 70
    
    print(f"Jobs: {len(items)}, Machines: {m}, Budget: {budget}")
    print("\nRunning enumeration with 600s time limit...")
    
    start_time = time.time()
    try:
        makespan, occurrences, assignments, nodes_evaluated, runtime = solve_bilevel_simpler(items, m, budget, time_limit=600.0, verbose=True)
        
        print("\n" + "=" * 70)
        print("RESULT:")
        print(f"  Makespan: {makespan}")
        print(f"  Selection: {occurrences}")
        print(f"  Nodes evaluated: {nodes_evaluated}")
        print(f"  Runtime: {runtime:.2f}s")
        print("=" * 70)
    except TimeoutError as e:
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"TIMEOUT: {e}")
        print(f"  Runtime: {elapsed:.2f}s")
        print("=" * 70)

# Instance 12: High Capacity
def test_instance_12():
    print("=" * 70)
    print("TESTING INSTANCE #12: High Capacity")
    print("=" * 70)
    
    items = [
        Item(name="A", duration=22, price=10),  # ratio: 2.20
        Item(name="B", duration=18, price=8),   # ratio: 2.25
        Item(name="C", duration=25, price=12),  # ratio: 2.08
        Item(name="D", duration=20, price=9),   # ratio: 2.22
        Item(name="E", duration=15, price=7),   # ratio: 2.14
        Item(name="F", duration=12, price=6),   # ratio: 2.00
        Item(name="G", duration=28, price=14),  # ratio: 2.00
        Item(name="H", duration=16, price=8),   # ratio: 2.00
        Item(name="I", duration=14, price=7),   # ratio: 2.00
    ]
    m = 8
    budget = 85
    
    print(f"Jobs: {len(items)}, Machines: {m}, Budget: {budget}")
    print("\nRunning enumeration with 600s time limit...")
    
    start_time = time.time()
    try:
        makespan, occurrences, assignments, nodes_evaluated, runtime = solve_bilevel_simpler(items, m, budget, time_limit=600.0, verbose=True)
        
        print("\n" + "=" * 70)
        print("RESULT:")
        print(f"  Makespan: {makespan}")
        print(f"  Selection: {occurrences}")
        print(f"  Nodes evaluated: {nodes_evaluated}")
        print(f"  Runtime: {runtime:.2f}s")
        print("=" * 70)
    except TimeoutError as e:
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"TIMEOUT: {e}")
        print(f"  Runtime: {elapsed:.2f}s")
        print("=" * 70)

# Instance 14: Wide Variety
def test_instance_14():
    print("=" * 70)
    print("TESTING INSTANCE #14: Wide Variety")
    print("=" * 70)
    
    items = [
        Item(name="A", duration=40, price=8),   # ratio: 5.00 - very cheap
        Item(name="B", duration=15, price=15),  # ratio: 1.00 - expensive
        Item(name="C", duration=25, price=10),  # ratio: 2.50
        Item(name="D", duration=18, price=9),   # ratio: 2.00
        Item(name="E", duration=22, price=11),  # ratio: 2.00
        Item(name="F", duration=30, price=12),  # ratio: 2.50
        Item(name="G", duration=20, price=8),   # ratio: 2.50
        Item(name="H", duration=12, price=6),   # ratio: 2.00
        Item(name="I", duration=16, price=8),   # ratio: 2.00
        Item(name="J", duration=28, price=14),  # ratio: 2.00
    ]
    m = 9
    budget = 95
    
    print(f"Jobs: {len(items)}, Machines: {m}, Budget: {budget}")
    print("\nRunning enumeration with 600s time limit...")
    
    start_time = time.time()
    try:
        makespan, occurrences, assignments, nodes_evaluated, runtime = solve_bilevel_simpler(items, m, budget, time_limit=600.0, verbose=True)
        
        print("\n" + "=" * 70)
        print("RESULT:")
        print(f"  Makespan: {makespan}")
        print(f"  Selection: {occurrences}")
        print(f"  Nodes evaluated: {nodes_evaluated}")
        print(f"  Runtime: {runtime:.2f}s")
        print("=" * 70)
    except TimeoutError as e:
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"TIMEOUT: {e}")
        print(f"  Runtime: {elapsed:.2f}s")
        print("=" * 70)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_single_enum.py <instance_number> [time_limit]")
        print("  instance_number: 11, 12, or 14")
        print("  time_limit: optional, in seconds (default: 600)")
        sys.exit(1)
    
    instance = sys.argv[1]
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    
    # Override the time limit in the functions
    if instance == "11":
        test_instance_11()
    elif instance == "12":
        # Modify instance 12 test to use custom time limit
        print("=" * 70)
        print("TESTING INSTANCE #12: High Capacity")
        print("=" * 70)
        
        from models import Item
        items = [
            Item(name="A", duration=22, price=10),
            Item(name="B", duration=18, price=8),
            Item(name="C", duration=25, price=12),
            Item(name="D", duration=20, price=9),
            Item(name="E", duration=15, price=7),
            Item(name="F", duration=12, price=6),
            Item(name="G", duration=28, price=14),
            Item(name="H", duration=16, price=8),
            Item(name="I", duration=14, price=7),
        ]
        m = 8
        budget = 85
        
        print(f"Jobs: {len(items)}, Machines: {m}, Budget: {budget}")
        print(f"\nRunning enumeration with {time_limit:.0f}s time limit...")
        
        import time
        start_time = time.time()
        try:
            makespan, occurrences, assignments, nodes_evaluated, runtime = solve_bilevel_simpler(items, m, budget, time_limit=time_limit, verbose=True)
            
            print("\n" + "=" * 70)
            print("RESULT:")
            print(f"  Makespan: {makespan}")
            print(f"  Selection: {occurrences}")
            print(f"  Nodes evaluated: {nodes_evaluated}")
            print(f"  Runtime: {runtime:.2f}s")
            print("=" * 70)
        except TimeoutError as e:
            elapsed = time.time() - start_time
            print("\n" + "=" * 70)
            print(f"TIMEOUT: {e}")
            print(f"  Runtime: {elapsed:.2f}s")
            print("=" * 70)
    elif instance == "14":
        # Modify instance 14 test to use custom time limit
        print("=" * 70)
        print("TESTING INSTANCE #14: Wide Variety")
        print("=" * 70)
        
        from models import Item
        items = [
            Item(name="A", duration=40, price=8),
            Item(name="B", duration=15, price=15),
            Item(name="C", duration=25, price=10),
            Item(name="D", duration=18, price=9),
            Item(name="E", duration=22, price=11),
            Item(name="F", duration=30, price=12),
            Item(name="G", duration=20, price=8),
            Item(name="H", duration=12, price=6),
            Item(name="I", duration=16, price=8),
            Item(name="J", duration=28, price=14),
        ]
        m = 9
        budget = 95
        
        print(f"Jobs: {len(items)}, Machines: {m}, Budget: {budget}")
        print(f"\nRunning enumeration with {time_limit:.0f}s time limit...")
        
        import time
        start_time = time.time()
        try:
            makespan, occurrences, assignments, nodes_evaluated, runtime = solve_bilevel_simpler(items, m, budget, time_limit=time_limit, verbose=True)
            
            print("\n" + "=" * 70)
            print("RESULT:")
            print(f"  Makespan: {makespan}")
            print(f"  Selection: {occurrences}")
            print(f"  Nodes evaluated: {nodes_evaluated}")
            print(f"  Runtime: {runtime:.2f}s")
            print("=" * 70)
        except TimeoutError as e:
            elapsed = time.time() - start_time
            print("\n" + "=" * 70)
            print(f"TIMEOUT: {e}")
            print(f"  Runtime: {elapsed:.2f}s")
            print("=" * 70)
    else:
        print(f"Invalid instance number: {instance}")
        print("Valid options: 11, 12, 14")
        sys.exit(1)
