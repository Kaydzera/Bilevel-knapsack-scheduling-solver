"""Test multiple instances on both enumeration and BnB."""
import json
import os
from models import MainProblem, Item
from bnb import run_bnb_classic
from bilevel_gurobi import solve_bilevel_simpler

# Path to cache enumeration results
ENUMERATION_CACHE_FILE = "enumeration_results_cache.json"

# ============================================================
# TEST INSTANCES
# ============================================================

instances = []

# Instance 1: Tiny - baseline (4 jobs, 2 machines, budget 10)
instances.append({
    "name": "Tiny Baseline",
    "items": [
        Item(name="A", duration=5, price=3),   # ratio: 1.67
        Item(name="B", duration=8, price=6),   # ratio: 1.33
        Item(name="C", duration=3, price=2),   # ratio: 1.50
        Item(name="D", duration=7, price=4),   # ratio: 1.75
    ],
    "machines": 2,
    "budget": 10
})


# Instance 2: Uniform - all items have same duration/price ratio
instances.append({
    "name": "Uniform Ratios",
    "items": [
        Item(name="A", duration=4, price=2),   # ratio: 2.00
        Item(name="B", duration=6, price=3),   # ratio: 2.00
        Item(name="C", duration=8, price=4),   # ratio: 2.00
        Item(name="D", duration=10, price=5),  # ratio: 2.00
    ],
    "machines": 2,
    "budget": 12
})


# Instance 3: High variance - one very attractive item
instances.append({
    "name": "Dominant Item",
    "items": [
        Item(name="A", duration=20, price=2),  # ratio: 10.00 - very cheap, long duration
        Item(name="B", duration=3, price=5),   # ratio: 0.60 - expensive, short
        Item(name="C", duration=5, price=4),   # ratio: 1.25
        Item(name="D", duration=4, price=3),   # ratio: 1.33
    ],
    "machines": 2,
    "budget": 10
})

# Instance 4: Larger budget - more exploration
instances.append({
    "name": "Large Budget",
    "items": [
        Item(name="A", duration=3, price=2),   # ratio: 1.50
        Item(name="B", duration=7, price=3),   # ratio: 2.33
        Item(name="C", duration=5, price=4),   # ratio: 1.25
        Item(name="D", duration=9, price=5),   # ratio: 1.80
    ],
    "machines": 2,
    "budget": 20
})

# Instance 5: More machines - different scheduling dynamics
instances.append({
    "name": "More Machines",
    "items": [
        Item(name="A", duration=6, price=3),   # ratio: 2.00
        Item(name="B", duration=8, price=4),   # ratio: 2.00
        Item(name="C", duration=4, price=2),   # ratio: 2.00
        Item(name="D", duration=10, price=5),  # ratio: 2.00
    ],
    "machines": 4,
    "budget": 15
})

# Instance 6: Extreme ratios - cheap vs expensive
instances.append({
    "name": "Extreme Ratios",
    "items": [
        Item(name="A", duration=15, price=1),  # ratio: 15.00 - very cheap
        Item(name="B", duration=2, price=10),  # ratio: 0.20 - very expensive
        Item(name="C", duration=8, price=4),   # ratio: 2.00
        Item(name="D", duration=6, price=3),   # ratio: 2.00
    ],
    "machines": 2,
    "budget": 12
})

# Instance 7: Five job types - larger search space
instances.append({
    "name": "Five Jobs",
    "items": [
        Item(name="A", duration=4, price=2),   # ratio: 2.00
        Item(name="B", duration=6, price=3),   # ratio: 2.00
        Item(name="C", duration=8, price=5),   # ratio: 1.60
        Item(name="D", duration=5, price=4),   # ratio: 1.25
        Item(name="E", duration=7, price=3),   # ratio: 2.33
    ],
    "machines": 3,
    "budget": 15
})

# Instance 8: Tight budget - few feasible solutions
instances.append({
    "name": "Tight Budget",
    "items": [
        Item(name="A", duration=10, price=5),  # ratio: 2.00
        Item(name="B", duration=8, price=4),   # ratio: 2.00
        Item(name="C", duration=6, price=3),   # ratio: 2.00
        Item(name="D", duration=4, price=2),   # ratio: 2.00
    ],
    "machines": 2,
    "budget": 6
})

# Instance 9: All expensive - budget is the main constraint
instances.append({
    "name": "Expensive Items",
    "items": [
        Item(name="A", duration=3, price=8),   # ratio: 0.38
        Item(name="B", duration=5, price=10),  # ratio: 0.50
        Item(name="C", duration=4, price=9),   # ratio: 0.44
        Item(name="D", duration=2, price=7),   # ratio: 0.29
    ],
    "machines": 2,
    "budget": 15
})

# Instance 10: Mixed - combination of interesting properties
instances.append({
    "name": "Mixed Characteristics",
    "items": [
        Item(name="A", duration=12, price=2),  # ratio: 6.00 - cheap, long
        Item(name="B", duration=3, price=8),   # ratio: 0.38 - expensive, short
        Item(name="C", duration=7, price=4),   # ratio: 1.75
        Item(name="D", duration=5, price=5),   # ratio: 1.00
        Item(name="E", duration=9, price=3),   # ratio: 3.00
    ],
    "machines": 3,
    "budget": 18
})

# ============================================================
# ENUMERATION CACHE MANAGEMENT
# ============================================================

def load_enumeration_cache():
    """Load cached enumeration results from file."""
    if os.path.exists(ENUMERATION_CACHE_FILE):
        try:
            with open(ENUMERATION_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache file: {e}")
            return {}
    return {}

def save_enumeration_cache(cache):
    """Save enumeration results to cache file."""
    try:
        with open(ENUMERATION_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")

def get_instance_key(name, items, m, budget):
    """Generate a unique key for an instance."""
    # Use name, m, budget, and item specs as key
    item_specs = [(i.duration, i.price) for i in items]
    return f"{name}_{m}_{budget}_{item_specs}"

# ============================================================
# RUN TESTS
# ============================================================

def run_instance(instance_data, use_enumeration=False, enable_logging=True):
    """Run BnB (and optionally enumeration) on an instance."""
    items = instance_data["items"]
    m = instance_data["machines"]
    budget = instance_data["budget"]
    name = instance_data["name"]
    
    prices = [i.price for i in items]
    durations = [i.duration for i in items]
    
    print("\n" + "=" * 70)
    print(f"INSTANCE: {name}")
    print("=" * 70)
    print(f"Jobs: {len(items)}, Machines: {m}, Budget: {budget}")
    print("Items (duration/price/ratio):")
    for item in items:
        ratio = item.duration / item.price
        print(f"  {item.name}: duration={item.duration}, price={item.price}, ratio={ratio:.2f}")
    print("=" * 70)
    
    # Create problem
    problem = MainProblem(prices, durations, m, budget)
    
    # Run BnB
    print("\n### Branch-and-Bound ###")
    result_bnb = run_bnb_classic(problem, max_nodes=100000, verbose=False, instance_name=name, enable_logging=enable_logging)
    print(f"BnB Result: makespan={result_bnb['best_obj']:.1f}, "
          f"selection={result_bnb['best_selection']}, "
          f"nodes={result_bnb['nodes_explored']}")
    
    # Optionally run enumeration (only for small instances)
    if use_enumeration:
        print("\n### Complete Enumeration ###")
        
        # Load cache
        cache = load_enumeration_cache()
        instance_key = get_instance_key(name, items, m, budget)
        
        # Check if result is cached
        if instance_key in cache:
            print("Using cached enumeration result...")
            cached = cache[instance_key]
            makespan_enum = cached['makespan']
            occ_enum = cached['selection']
            nodes_enum = cached['nodes_evaluated']
        else:
            print("Running enumeration (not cached)...")
            makespan_enum, occ_enum, _, nodes_enum = solve_bilevel_simpler(items, m, budget, time_limit=30.0, verbose=False)
            
            # Save to cache
            cache[instance_key] = {
                'makespan': makespan_enum,
                'selection': occ_enum,
                'nodes_evaluated': nodes_enum
            }
            save_enumeration_cache(cache)
        
        print(f"Enumeration Result: makespan={makespan_enum:.1f}, selection={occ_enum}")
        print(f"Nodes: BnB explored {result_bnb['nodes_explored']}, Enumeration evaluated {nodes_enum}")
        
        # Compare results
        if abs(result_bnb['best_obj'] - makespan_enum) < 0.01:
            print("OK - Results match!")
        else:
            print("FAIL - Results differ!")
        
        # Log comparison to BnB log if logging enabled
        if enable_logging:
            # Use absolute path to workspace root logs directory
            workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(workspace_root, "logs")
            # Find the most recent log file for this instance
            import glob
            log_pattern = os.path.join(log_dir, f"{name}_*.log")
            log_files = glob.glob(log_pattern)
            if log_files:
                latest_log = max(log_files, key=os.path.getmtime)
                try:
                    with open(latest_log, 'a') as f:
                        f.write(f"\n{'='*70}\n")
                        f.write(f"ENUMERATION COMPARISON\n")
                        f.write(f"{'='*70}\n")
                        f.write(f"BnB nodes explored: {result_bnb['nodes_explored']}\n")
                        f.write(f"Enumeration nodes evaluated: {nodes_enum}\n")
                        f.write(f"Ratio (BnB/Enum): {result_bnb['nodes_explored']/nodes_enum:.2f}x\n")
                        f.write(f"Enumeration makespan: {makespan_enum:.1f}\n")
                        f.write(f"BnB makespan: {result_bnb['best_obj']:.1f}\n")
                        f.write(f"Match: {'YES' if abs(result_bnb['best_obj'] - makespan_enum) < 0.01 else 'NO'}\n")
                        f.write(f"{'='*70}\n")
                except Exception as e:
                    print(f"Warning: Could not write to log file: {e}")
    
    return result_bnb


# Run all instances
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING MULTIPLE INSTANCES")
    print("=" * 70)
    
    # Run instances with enumeration for verification (logging disabled)
    for i in range(len(instances)):
        run_instance(instances[i], use_enumeration=True, enable_logging=False)
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)

    #cd 'c:\Users\oleda\.vscode\Solving stuff with Gurobi\gurobi_scheduling'; & "C:/Users/oleda/.vscode/Solving stuff with Gurobi/.venv311/Scripts/python.exe" test_small.py



