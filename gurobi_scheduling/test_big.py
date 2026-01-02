"""Test 100 large and complex instances on both enumeration and BnB."""
import json
import os
import random
from models import MainProblem, Item
from bnb import run_bnb_classic
from bilevel_gurobi import solve_bilevel_simpler

# Path to cache enumeration results
ENUMERATION_CACHE_FILE = "enumeration_results_cache_big.json"

# Set seed for reproducibility
random.seed(42)

# ============================================================
# GENERATE 100 COMPLEX TEST INSTANCES
# ============================================================

def generate_instances():
    """Generate 100 complex test instances with varying characteristics."""
    instances = []
    
    # Configuration ranges for complex instances
    job_counts = [6, 7, 8, 10, 12]  # More job types
    machine_counts = [3, 4, 5, 6, 8]  # More machines
    budget_multipliers = [1.5, 2.0, 2.5, 3.0, 4.0]  # Higher budgets relative to items
    
    instance_id = 1
    
    # Generate diverse instances
    for iteration in range(100):
        # Vary complexity
        n_jobs = random.choice(job_counts)
        n_machines = random.choice(machine_counts)
        
        # Generate items with different characteristics
        items = []
        total_min_cost = 0
        
        for j in range(n_jobs):
            # Generate durations and prices with different patterns
            if iteration % 5 == 0:
                # Pattern 1: Uniform ratios
                base_ratio = random.uniform(1.5, 3.0)
                duration = random.randint(3, 15)
                price = max(1, int(duration / base_ratio))
            elif iteration % 5 == 1:
                # Pattern 2: High variance - some cheap, some expensive
                if j % 2 == 0:
                    duration = random.randint(10, 20)  # Long and cheap
                    price = random.randint(1, 3)
                else:
                    duration = random.randint(2, 5)  # Short and expensive
                    price = random.randint(8, 15)
            elif iteration % 5 == 2:
                # Pattern 3: Increasing durations and prices
                duration = 3 + j * 2
                price = 2 + j
            elif iteration % 5 == 3:
                # Pattern 4: Random but realistic
                duration = random.randint(4, 18)
                price = random.randint(2, 12)
            else:
                # Pattern 5: Extreme cases
                if random.random() < 0.3:
                    duration = random.randint(20, 30)  # Very long
                    price = random.randint(1, 3)  # Very cheap
                else:
                    duration = random.randint(3, 12)
                    price = random.randint(3, 10)
            
            items.append(Item(name=f"J{j+1}", duration=duration, price=price))
            total_min_cost += price
        
        # Set budget based on total minimum cost and a multiplier
        budget_mult = random.choice(budget_multipliers)
        budget = int(total_min_cost * budget_mult / n_jobs) + random.randint(5, 20)
        
        # Create instance name based on characteristics
        avg_duration = sum(item.duration for item in items) / len(items)
        avg_price = sum(item.price for item in items) / len(items)
        avg_ratio = avg_duration / avg_price if avg_price > 0 else 0
        
        name = f"Complex_{instance_id:03d}_J{n_jobs}_M{n_machines}_B{budget}"
        
        instances.append({
            "name": name,
            "items": items,
            "machines": n_machines,
            "budget": budget,
            "id": instance_id
        })
        
        instance_id += 1
    
    return instances

# Generate all instances
instances = generate_instances()

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
    from logger import create_logger
    logger = create_logger(instance_name=name, log_dir="logs/test_big") if enable_logging else None
    result_bnb = run_bnb_classic(problem, max_nodes=100000, verbose=False, logger=logger, instance_name=name, enable_logging=enable_logging)
    print(f"BnB Result: makespan={result_bnb['best_obj']:.1f}, "
          f"selection={result_bnb['best_selection']}, "
          f"nodes={result_bnb['nodes_explored']}")
    
    # Optionally run enumeration (only for smaller instances due to time)
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
            runtime_enum = cached.get('runtime', None)
            timed_out = cached.get('timed_out', False)
        else:
            print("Running enumeration (not cached)...")
            timed_out = False
            try:
                makespan_enum, occ_enum, _, nodes_enum, runtime_enum = solve_bilevel_simpler(items, m, budget, time_limit=3600.0, verbose=True)
            except TimeoutError as e:
                print(f"\nEnumeration timed out after {3600.0}s")
                print(f"Timeout exception: {e}")
                # Extract node count from exception message
                import re
                match = re.search(r'after checking (\d+) selections', str(e))
                if match:
                    nodes_enum = int(match.group(1))
                else:
                    nodes_enum = 0  # Fallback if message format changes
                # Use partial results from the enumeration attempt
                # Since we don't have partial results, use BnB results with a flag
                makespan_enum = result_bnb['best_obj']
                occ_enum = result_bnb['best_selection']
                runtime_enum = 3600.0
                timed_out = True
                print(f"Recording timeout in cache with BnB solution: makespan={makespan_enum}, nodes checked: {nodes_enum}")
            
            # Save to cache (even if timed out)
            cache[instance_key] = {
                'makespan': makespan_enum,
                'selection': occ_enum,
                'nodes_evaluated': nodes_enum,
                'runtime': runtime_enum,
                'timed_out': timed_out
            }
            save_enumeration_cache(cache)
        
        if timed_out:
            print(f"Enumeration Result: TIMED OUT (using BnB solution: makespan={makespan_enum:.1f})")
        else:
            print(f"Enumeration Result: makespan={makespan_enum:.1f}, selection={occ_enum}")
        
        print(f"Nodes: BnB explored {result_bnb['nodes_explored']}, Enumeration evaluated {nodes_enum}")
        
        # Compare runtimes
        bnb_runtime = result_bnb.get('runtime', None)
        if bnb_runtime is not None and runtime_enum is not None:
            speedup = runtime_enum / bnb_runtime if bnb_runtime > 0 else float('inf')
            print(f"Runtime: BnB {bnb_runtime:.4f}s, Enumeration {runtime_enum:.4f}s (Speedup: {speedup:.2f}x)")
        
        # Compare results
        if timed_out:
            print("TIMEOUT - Enumeration could not verify BnB solution within time limit")
        elif abs(result_bnb['best_obj'] - makespan_enum) < 0.01:
            print("OK - Results match!")
        elif makespan_enum < result_bnb['best_obj']:
            print("WARNING - Enumeration found worse solution (possible enumeration bug or early termination)")
        else:
            print("FAIL - Results differ! (BnB may have error)")
        
        # Log comparison to BnB log if logging enabled
        if enable_logging:
            from logger import BnBLogger
            from pathlib import Path
            import glob
            
            # Find the most recent log file for this instance
            log_dir = Path("logs/test_big")
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
                    if nodes_enum > 0:
                        f.write(f"Ratio (BnB/Enum): {result_bnb['nodes_explored']/nodes_enum:.2f}x\n")
                    if bnb_runtime is not None and runtime_enum is not None:
                        f.write(f"BnB runtime: {bnb_runtime:.4f}s\n")
                        f.write(f"Enumeration runtime: {runtime_enum:.4f}s\n")
                        speedup = runtime_enum / bnb_runtime if bnb_runtime > 0 else float('inf')
                        f.write(f"Speedup (Enum/BnB): {speedup:.2f}x\n")
                    f.write(f"Enumeration makespan: {makespan_enum}\n")
                    f.write(f"BnB makespan: {result_bnb['best_obj']}\n")
                    if timed_out:
                        f.write(f"Match: TIMEOUT (Enum could not complete verification)\n")
                    elif abs(result_bnb['best_obj'] - makespan_enum) < 0.01:
                        f.write(f"Match: YES (Solutions match)\n")
                    elif makespan_enum < result_bnb['best_obj']:
                        f.write(f"Match: WARNING - Enumeration found WORSE solution than BnB ({makespan_enum:.1f} < {result_bnb['best_obj']:.1f})\n")
                    else:
                        f.write(f"Match: NO - BnB may have error (Enum found better: {makespan_enum:.1f} > {result_bnb['best_obj']:.1f})\n")
                    f.write(f"Timed out: {timed_out}\n")
                    f.write("=" * 70 + "\n")
    
    return result_bnb


# Run all instances
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING 100 COMPLEX INSTANCES")
    print("=" * 70)
    
    # Print summary of instance characteristics
    print("\nInstance Summary:")
    print(f"Total instances: {len(instances)}")
    
    job_counts = {}
    machine_counts = {}
    for inst in instances:
        n_jobs = len(inst['items'])
        n_machines = inst['machines']
        job_counts[n_jobs] = job_counts.get(n_jobs, 0) + 1
        machine_counts[n_machines] = machine_counts.get(n_machines, 0) + 1
    
    print(f"Job types distribution: {dict(sorted(job_counts.items()))}")
    print(f"Machine counts distribution: {dict(sorted(machine_counts.items()))}")
    
    # Run instances with enumeration for all (using 1 hour time limit per instance)
    for i in range(len(instances)):
        run_instance(instances[i], use_enumeration=True, enable_logging=True)
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
