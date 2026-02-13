"""
Bilevel knapsack scheduling problem solver using complete enumeration.

This module solves the bilevel problem by enumerating all feasible job selections
and solving the follower's scheduling problem for each one.

Leader (upper level): Selects job occurrences to maximize makespan
Follower (lower level): Assigns jobs to machines to minimize makespan

Note: Direct Gurobi formulation is not possible for discrete bilevel problems
as KKT conditions only apply to continuous optimization problems.
"""

from models import Item
from typing import List, Tuple, Dict
import time


def solve_bilevel_simpler(
    items: List[Item],
    m: int,
    budget: int,
    time_limit: float = 300.0,
    verbose: bool = True
) -> Tuple[float, List[int], Dict[int, List[int]]]:
    """
    Simplified bilevel formulation: enumerate leader's decisions and solve follower's problem.
    
    This is a straightforward approach that iterates through feasible job selections
    and solves the scheduling problem for each, keeping track of the worst case.
    
    This is more tractable but only practical for small instances.
    
    Args:
        items: List of job types with durations and prices
        m: Number of machines
        budget: Leader's budget constraint
        time_limit: Maximum solve time in seconds
        verbose: Whether to print progress
        
    Returns:
        Tuple of (makespan, occurrences, machine_assignments)
    """
    from solvers import solve_scheduling_readable
    import itertools
    
    n = len(items)
    
    # Generate all feasible job selections (within budget)
    max_per_job = [budget // items[i].price for i in range(n)]
    
    best_makespan = 0
    best_occurrences = None
    best_assignments = None
    nodes_evaluated = 0  # Counter for solve_scheduling_readable calls
    
    count = 0
    start_time = time.time()
    
    if verbose:
        print("=" * 70)
        print("Enumerating feasible job selections...")
        print(f"Jobs: {n}, Machines: {m}, Budget: {budget}")
    
    # Generate candidates (bounded enumeration)
    ranges = [range(max_per_job[i] + 1) for i in range(n)]

    #find item with the lowest price:
    min_price = min(items[i].price for i in range(n))
    min_price_index = next(i for i in range(n) if items[i].price == min_price)
    
    for occurrences in itertools.product(*ranges):
        # Check time limit
        if time.time() - start_time > time_limit:
            if verbose:
                print(f"\nTime limit reached after checking {count} selections")
            #raise an exception with count information
            raise TimeoutError(f"Time limit exceeded in bilevel enumeration after checking {count} selections.")
        
        # Check budget
        cost = sum(items[i].price * occurrences[i] for i in range(n))
        if cost > budget:
            continue
        
        count += 1

        #check if occurrences can be expaned by one more job of type 0 without violating budget
        if occurrences[min_price_index] < max_per_job[min_price_index]:
            new_cost = cost + items[min_price_index].price
            if new_cost <= budget:
                # If we can add one more job of the cheapest type, skip this combination and let the next iteration handle it
                continue


        # Create job list from occurrences
        jobs = []
        for i in range(n):
            for _ in range(occurrences[i]):
                jobs.append(items[i].duration)
        
        # Skip if no jobs selected
        if len(jobs) == 0:
            continue
        
        # Solve follower's problem
        result = solve_scheduling_readable(len(jobs), m, jobs, verbose=False)
        nodes_evaluated += 1  # Count each scheduling evaluation
        
        if result is not None and result['makespan'] > best_makespan:
            best_makespan = result['makespan']
            best_occurrences = list(occurrences)
            best_assignments = result['machine_assignments']
            
        if verbose and count % 1000 == 0:
            print(f"  Checked {count} selections, best makespan so far: {best_makespan}")
    
    solve_time = time.time() - start_time
    
    if verbose:
        print(f"\nChecked {count} feasible selections in {solve_time:.2f} seconds")
        print(f"Evaluated {nodes_evaluated} schedules (nodes)")
        if best_occurrences:
            print(f"Best makespan: {best_makespan}")
            print(f"Best occurrences: {best_occurrences}")
            print(f"Budget used: {sum(items[i].price * best_occurrences[i] for i in range(n))}/{budget}")
        print("=" * 70)
    
    return best_makespan, best_occurrences, best_assignments, nodes_evaluated, solve_time


if __name__ == "__main__":
    # Small test instance to verify enumeration works
    # 4 job types, 2 machines, small budget
    # This should enumerate quickly: max combinations ≈ 4×2×3×2 = 48
    items = [
        Item(name="A", duration=5, price=3),   # Job 0: duration 5, costs 3
        Item(name="B", duration=8, price=6),   # Job 1: duration 8, costs 6
        Item(name="C", duration=3, price=2),   # Job 2: duration 3, costs 2
        Item(name="D", duration=7, price=4),   # Job 3: duration 7, costs 4
    ]
    
    m = 2
    budget = 10
    
    # Expected max combinations: [10//3+1] × [10//6+1] × [10//2+1] × [10//4+1]
    # = 4 × 2 × 6 × 3 = 144 combinations (but many will violate budget)
    
    print("\n" + "=" * 70)
    print("TESTING BILEVEL ENUMERATION")
    print("=" * 70)
    
    # Test enumeration approach
    print("\n### Complete enumeration method (exact solution) ###")
    makespan, occ, assign, nodes, runtime = solve_bilevel_simpler(items, m, budget, time_limit=60.0)



'''
pls explain the follwing points or tell me a littlebit about these:
Real-world applications (how is this related to cloud computing?)
Node representation and branching rules
Bound dominance pruning
Optimality dominance pruning
List scheduling algorithm
Gurobi MIP formulation (optional benchmark)
Node priority queue

Pruning mechanisms - Bound dominance vs optimality dominance
'''