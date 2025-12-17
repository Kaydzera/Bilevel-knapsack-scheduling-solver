"""Test multiple medium-sized instances on both enumeration and BnB."""
import json
import os
from models import MainProblem, Item
from bnb import run_bnb_classic
from bilevel_gurobi import solve_bilevel_simpler

# Path to cache enumeration results
ENUMERATION_CACHE_FILE = "enumeration_results_cache_middle.json"

# ============================================================
# TEST INSTANCES
# ============================================================

test_instances = []

# Instance 1: Medium baseline (6 jobs, 3 machines, budget 50)
test_instances.append({
    "name": "Medium Baseline",
    "items": [
        Item(name="A", duration=8, price=4),   # ratio: 2.00
        Item(name="B", duration=10, price=5),  # ratio: 2.00
        Item(name="C", duration=6, price=3),   # ratio: 2.00
        Item(name="D", duration=12, price=6),  # ratio: 2.00
        Item(name="E", duration=7, price=4),   # ratio: 1.75
        Item(name="F", duration=9, price=5),   # ratio: 1.80
    ],
    "machines": 3,
    "budget": 50
})

# Instance 2: Varied ratios (6 jobs, 4 machines, budget 30)
test_instances.append({
    "name": "Varied Ratios",
    "items": [
        Item(name="A", duration=15, price=3),  # ratio: 5.00
        Item(name="B", duration=8, price=7),   # ratio: 1.14
        Item(name="C", duration=12, price=4),  # ratio: 3.00
        Item(name="D", duration=6, price=5),   # ratio: 1.20
        Item(name="E", duration=10, price=6),  # ratio: 1.67
        Item(name="F", duration=7, price=4),   # ratio: 1.75
    ],
    "machines": 4,
    "budget": 30
})

# Instance 3: High budget exploration (7 jobs, 4 machines, budget 40)
test_instances.append({
    "name": "High Budget",
    "items": [
        Item(name="A", duration=9, price=5),   # ratio: 1.80
        Item(name="B", duration=11, price=6),  # ratio: 1.83
        Item(name="C", duration=8, price=4),   # ratio: 2.00
        Item(name="D", duration=13, price=7),  # ratio: 1.86
        Item(name="E", duration=7, price=3),   # ratio: 2.33
        Item(name="F", duration=10, price=5),  # ratio: 2.00
        Item(name="G", duration=6, price=4),   # ratio: 1.50
    ],
    "machines": 4,
    "budget": 40
})

# Instance 4: Many machines (6 jobs, 6 machines, budget 28)
test_instances.append({
    "name": "Many Machines",
    "items": [
        Item(name="A", duration=12, price=5),  # ratio: 2.40
        Item(name="B", duration=10, price=4),  # ratio: 2.50
        Item(name="C", duration=8, price=3),   # ratio: 2.67
        Item(name="D", duration=14, price=6),  # ratio: 2.33
        Item(name="E", duration=9, price=4),   # ratio: 2.25
        Item(name="F", duration=11, price=5),  # ratio: 2.20
    ],
    "machines": 6,
    "budget": 28
})

# Instance 5: Dominant cheap item (7 jobs, 3 machines, budget 22)
test_instances.append({
    "name": "Cheap Dominant",
    "items": [
        Item(name="A", duration=25, price=3),  # ratio: 8.33 - very cheap
        Item(name="B", duration=8, price=6),   # ratio: 1.33
        Item(name="C", duration=10, price=7),  # ratio: 1.43
        Item(name="D", duration=7, price=5),   # ratio: 1.40
        Item(name="E", duration=9, price=6),   # ratio: 1.50
        Item(name="F", duration=6, price=4),   # ratio: 1.50
        Item(name="G", duration=11, price=8),  # ratio: 1.38
    ],
    "machines": 3,
    "budget": 22
})

# Instance 6: Expensive focus (6 jobs, 4 machines, budget 35)
test_instances.append({
    "name": "Expensive Focus",
    "items": [
        Item(name="A", duration=7, price=10),  # ratio: 0.70
        Item(name="B", duration=9, price=12),  # ratio: 0.75
        Item(name="C", duration=6, price=9),   # ratio: 0.67
        Item(name="D", duration=8, price=11),  # ratio: 0.73
        Item(name="E", duration=5, price=8),   # ratio: 0.63
        Item(name="F", duration=10, price=13), # ratio: 0.77
    ],
    "machines": 4,
    "budget": 35
})

# Instance 7: Eight jobs (8 jobs, 4 machines, budget 32)
test_instances.append({
    "name": "Eight Jobs",
    "items": [
        Item(name="A", duration=10, price=5),  # ratio: 2.00
        Item(name="B", duration=8, price=4),   # ratio: 2.00
        Item(name="C", duration=12, price=6),  # ratio: 2.00
        Item(name="D", duration=9, price=5),   # ratio: 1.80
        Item(name="E", duration=11, price=6),  # ratio: 1.83
        Item(name="F", duration=7, price=4),   # ratio: 1.75
        Item(name="G", duration=13, price=7),  # ratio: 1.86
        Item(name="H", duration=6, price=3),   # ratio: 2.00
    ],
    "machines": 4,
    "budget": 32
})

# Instance 8: Tight constraints (6 jobs, 3 machines, budget 18)
test_instances.append({
    "name": "Tight Constraints",
    "items": [
        Item(name="A", duration=14, price=6),  # ratio: 2.33
        Item(name="B", duration=12, price=5),  # ratio: 2.40
        Item(name="C", duration=10, price=4),  # ratio: 2.50
        Item(name="D", duration=8, price=3),   # ratio: 2.67
        Item(name="E", duration=11, price=5),  # ratio: 2.20
        Item(name="F", duration=9, price=4),   # ratio: 2.25
    ],
    "machines": 3,
    "budget": 18
})

# Instance 9: Mixed complexity (7 jobs, 5 machines, budget 35)
test_instances.append({
    "name": "Mixed Complexity",
    "items": [
        Item(name="A", duration=18, price=4),  # ratio: 4.50 - cheap, long
        Item(name="B", duration=6, price=9),   # ratio: 0.67 - expensive, short
        Item(name="C", duration=11, price=5),  # ratio: 2.20
        Item(name="D", duration=9, price=6),   # ratio: 1.50
        Item(name="E", duration=13, price=5),  # ratio: 2.60
        Item(name="F", duration=7, price=4),   # ratio: 1.75
        Item(name="G", duration=10, price=7),  # ratio: 1.43
    ],
    "machines": 5,
    "budget": 35
})

# Instance 10: Balanced medium (6 jobs, 4 machines, budget 27)
test_instances.append({
    "name": "Balanced Medium",
    "items": [
        Item(name="A", duration=11, price=5),  # ratio: 2.20
        Item(name="B", duration=9, price=4),   # ratio: 2.25
        Item(name="C", duration=13, price=6),  # ratio: 2.17
        Item(name="D", duration=8, price=4),   # ratio: 2.00
        Item(name="E", duration=10, price=5),  # ratio: 2.00
        Item(name="F", duration=12, price=6),  # ratio: 2.00
    ],
    "machines": 4,
    "budget": 27
})

# Instance 11: Large scale baseline (7 jobs, 6 machines, budget 120)
test_instances.append({
    "name": "Large Scale Baseline",
    "items": [
        Item(name="A", duration=15, price=8),   # ratio: 1.88
        Item(name="B", duration=18, price=10),  # ratio: 1.80
        Item(name="C", duration=12, price=6),   # ratio: 2.00
        Item(name="D", duration=20, price=12),  # ratio: 1.67
        Item(name="E", duration=14, price=7),   # ratio: 2.00
        Item(name="F", duration=16, price=9),   # ratio: 1.78
        Item(name="G", duration=10, price=5),   # ratio: 2.00
    ],
    "machines": 6,
    "budget": 70
})

# Instance 12: High capacity (9 jobs, 8 machines, budget 150)
test_instances.append({
    "name": "High Capacity",
    "items": [
        Item(name="A", duration=22, price=10),  # ratio: 2.20
        Item(name="B", duration=18, price=8),   # ratio: 2.25
        Item(name="C", duration=25, price=12),  # ratio: 2.08
        Item(name="D", duration=20, price=9),   # ratio: 2.22
        Item(name="E", duration=15, price=7),   # ratio: 2.14
        Item(name="F", duration=12, price=6),   # ratio: 2.00
        Item(name="G", duration=28, price=14),  # ratio: 2.00
        Item(name="H", duration=16, price=8),   # ratio: 2.00
        Item(name="I", duration=14, price=7),   # ratio: 2.00
    ],
    "machines": 8,
    "budget": 85
})

# Instance 13: Cost sensitive (6 jobs, 7 machines, budget 100)
test_instances.append({
    "name": "Cost Sensitive",
    "items": [
        Item(name="A", duration=35, price=15),  # ratio: 2.33
        Item(name="B", duration=30, price=12),  # ratio: 2.50
        Item(name="C", duration=25, price=10),  # ratio: 2.50
        Item(name="D", duration=28, price=14),  # ratio: 2.00
        Item(name="E", duration=32, price=16),  # ratio: 2.00
        Item(name="F", duration=20, price=8),   # ratio: 2.50
    ],
    "machines": 7,
    "budget": 65
})

# Instance 14: Wide variety (10 jobs, 9 machines, budget 95)
test_instances.append({
    "name": "Wide Variety",
    "items": [
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
    ],
    "machines": 9,
    "budget": 95
})

# Instance 15: Heavy workload (8 jobs, 5 machines, budget 140)
test_instances.append({
    "name": "Heavy Workload",
    "items": [
        Item(name="A", duration=35, price=14),  # ratio: 2.50
        Item(name="B", duration=30, price=12),  # ratio: 2.50
        Item(name="C", duration=40, price=16),  # ratio: 2.50
        Item(name="D", duration=25, price=10),  # ratio: 2.50
        Item(name="E", duration=28, price=14),  # ratio: 2.00
        Item(name="F", duration=32, price=16),  # ratio: 2.00
        Item(name="G", duration=22, price=11),  # ratio: 2.00
        Item(name="H", duration=20, price=8),   # ratio: 2.50
    ],
    "machines": 5,
    "budget": 75
})

# Instance 16: Distributed resources (7 jobs, 10 machines, budget 130)
test_instances.append({
    "name": "Distributed Resources",
    "items": [
        Item(name="A", duration=24, price=12),  # ratio: 2.00
        Item(name="B", duration=20, price=10),  # ratio: 2.00
        Item(name="C", duration=18, price=9),   # ratio: 2.00
        Item(name="D", duration=16, price=8),   # ratio: 2.00
        Item(name="E", duration=22, price=11),  # ratio: 2.00
        Item(name="F", duration=14, price=7),   # ratio: 2.00
        Item(name="G", duration=12, price=6),   # ratio: 2.00
    ],
    "machines": 10,
    "budget": 80
})

# Instance 17: Premium options (5 jobs, 6 machines, budget 110)
test_instances.append({
    "name": "Premium Options",
    "items": [
        Item(name="A", duration=20, price=25),  # ratio: 0.80 - very expensive
        Item(name="B", duration=25, price=30),  # ratio: 0.83
        Item(name="C", duration=18, price=22),  # ratio: 0.82
        Item(name="D", duration=30, price=20),  # ratio: 1.50 - cheaper
        Item(name="E", duration=35, price=15),  # ratio: 2.33 - cheap
    ],
    "machines": 6,
    "budget": 60
})

# Instance 18: Flexible capacity (9 jobs, 7 machines, budget 170)
test_instances.append({
    "name": "Flexible Capacity",
    "items": [
        Item(name="A", duration=28, price=14),  # ratio: 2.00
        Item(name="B", duration=24, price=12),  # ratio: 2.00
        Item(name="C", duration=32, price=16),  # ratio: 2.00
        Item(name="D", duration=20, price=10),  # ratio: 2.00
        Item(name="E", duration=26, price=13),  # ratio: 2.00
        Item(name="F", duration=22, price=11),  # ratio: 2.00
        Item(name="G", duration=18, price=9),   # ratio: 2.00
        Item(name="H", duration=30, price=15),  # ratio: 2.00
        Item(name="I", duration=16, price=8),   # ratio: 2.00
    ],
    "machines": 7,
    "budget": 90
})

# Instance 19: High budget extreme (8 jobs, 8 machines, budget 200)
test_instances.append({
    "name": "High Budget Extreme",
    "items": [
        Item(name="A", duration=45, price=18),  # ratio: 2.50
        Item(name="B", duration=38, price=16),  # ratio: 2.38
        Item(name="C", duration=42, price=20),  # ratio: 2.10
        Item(name="D", duration=35, price=14),  # ratio: 2.50
        Item(name="E", duration=40, price=16),  # ratio: 2.50
        Item(name="F", duration=30, price=12),  # ratio: 2.50
        Item(name="G", duration=32, price=16),  # ratio: 2.00
        Item(name="H", duration=28, price=14),  # ratio: 2.00
    ],
    "machines": 8,
    "budget": 100
})

# Instance 20: Balanced large (6 jobs, 8 machines, budget 115)
test_instances.append({
    "name": "Balanced Large",
    "items": [
        Item(name="A", duration=26, price=13),  # ratio: 2.00
        Item(name="B", duration=22, price=11),  # ratio: 2.00
        Item(name="C", duration=24, price=12),  # ratio: 2.00
        Item(name="D", duration=20, price=10),  # ratio: 2.00
        Item(name="E", duration=28, price=14),  # ratio: 2.00
        Item(name="F", duration=18, price=9),   # ratio: 2.00
    ],
    "machines": 8,
    "budget": 55
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

def run_instance(instance_data, use_enumeration=False, enable_logging=True, instance_number=None):
    """Run BnB (and optionally enumeration) on an instance."""
    items = instance_data["items"]
    m = instance_data["machines"]
    budget = instance_data["budget"]
    name = instance_data["name"]
    
    prices = [i.price for i in items]
    durations = [i.duration for i in items]
    
    # Add instance number to display name if provided
    display_name = f"#{instance_number}: {name}" if instance_number is not None else name
    
    print("\n" + "=" * 70)
    print(f"INSTANCE: {display_name}")
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
    from logger import create_logger
    import time
    logger = create_logger(instance_name=name, log_dir="logs/test_middle") if enable_logging else None
    start_time = time.time()
    result_bnb = run_bnb_classic(problem, max_nodes=500000, verbose=False, logger=logger, instance_name=name, enable_logging=enable_logging)
    elapsed = time.time() - start_time
    if elapsed > 1800:  # 30 minutes
        print(f"WARNING: Instance exceeded 30 minute time limit ({elapsed:.1f}s)")
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
            runtime_enum = cached.get('runtime', None)  # Get runtime if available
        else:
            print("Running enumeration (not cached)...")
            makespan_enum, occ_enum, _, nodes_enum, runtime_enum = solve_bilevel_simpler(items, m, budget, time_limit=600.0, verbose=False)
            
            # Save to cache
            cache[instance_key] = {
                'makespan': makespan_enum,
                'selection': occ_enum,
                'nodes_evaluated': nodes_enum,
                'runtime': runtime_enum
            }
            save_enumeration_cache(cache)
        
        print(f"Enumeration Result: makespan={makespan_enum:.1f}, selection={occ_enum}")
        print(f"Nodes: BnB explored {result_bnb['nodes_explored']}, Enumeration evaluated {nodes_enum}")
        
        # Compare runtimes
        bnb_runtime = result_bnb.get('runtime', None)
        if bnb_runtime is not None and runtime_enum is not None:
            speedup = runtime_enum / bnb_runtime if bnb_runtime > 0 else float('inf')
            print(f"Runtime: BnB {bnb_runtime:.4f}s, Enumeration {runtime_enum:.4f}s (Speedup: {speedup:.2f}x)")
        
        # Compare results
        if abs(result_bnb['best_obj'] - makespan_enum) < 0.01:
            print("OK - Results match!")
        else:
            print("FAIL - Results differ!")
        
        # Log comparison to BnB log if logging enabled
        if enable_logging:
            from logger import BnBLogger
            from pathlib import Path
            import glob
            
            # Find the most recent log file for this instance
            log_dir = Path("logs/test_middle")
            log_pattern = f"{name}_*.log"
            log_files = list(log_dir.glob(log_pattern))
            
            if log_files:
                # Sort by modification time and get the most recent
                most_recent_log = max(log_files, key=lambda p: p.stat().st_mtime)
                
                # Append comparison to log file
                with open(most_recent_log, 'a') as f:
                    f.write("\n" + "=" * 70 + "\n")
                    f.write("ENUMERATION COMPARISON\n")
                    f.write("=" * 70 + "\n")
                    f.write(f"BnB nodes explored: {result_bnb['nodes_explored']}\n")
                    f.write(f"Enumeration nodes evaluated: {nodes_enum}\n")
                    f.write(f"Ratio (BnB/Enum): {result_bnb['nodes_explored']/nodes_enum:.2f}x\n")
                    if bnb_runtime is not None and runtime_enum is not None:
                        f.write(f"BnB runtime: {bnb_runtime:.4f}s\n")
                        f.write(f"Enumeration runtime: {runtime_enum:.4f}s\n")
                        speedup = runtime_enum / bnb_runtime if bnb_runtime > 0 else float('inf')
                        f.write(f"Speedup (Enum/BnB): {speedup:.2f}x\n")
                    f.write(f"Enumeration makespan: {makespan_enum}\n")
                    f.write(f"BnB makespan: {result_bnb['best_obj']}\n")
                    match = "YES" if abs(result_bnb['best_obj'] - makespan_enum) < 0.01 else "NO"
                    f.write(f"Match: {match}\n")
                    f.write("=" * 70 + "\n")
    
    return result_bnb


# Run all instances
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING MEDIUM-SIZED INSTANCES")
    print("=" * 70)
    
    # Run instances with enumeration for verification (logging enabled)
    for i in range(len(test_instances)):
        run_instance(test_instances[i], use_enumeration=True, enable_logging=True, instance_number=i+1)
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
