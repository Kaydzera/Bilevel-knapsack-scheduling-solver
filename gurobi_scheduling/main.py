"""Bilevel optimization solver CLI.

This script provides a command-line interface for the bilevel optimization
solver. The leader maximizes the follower's makespan by selecting items
under a budget constraint, while the follower minimizes makespan by
scheduling jobs on identical machines.

Usage:
    python main.py                      # Run default scheduling demo
    python main.py bilevel-step1        # Print sample bilevel problem
    python main.py bilevel-step2        # Run leader heuristic + follower scheduling
    python main.py bilevel-step4-class  # Run branch-and-bound solver
    python main.py test-exact-bound     # Test exact IP bound computation
"""

import json
import os
import sys

from models import Item, MainProblem
from solvers import solve_scheduling, solve_leader_knapsack
from bnb import run_bnb_classic


def create_sample_bilevel():
    """Return a small sample bilevel instance (items, budget, machines).

    Returns:
        tuple: (items, budget, m) where:
            - items: List[Item] with job types
            - budget: int total budget
            - m: int number of machines for follower
    """
    durations = [2, 17, 4, 24, 7, 5, 11, 9]
    prices =    [1,  8, 2, 10, 3, 3, 5, 4]
    items = [Item(name=f"item{i}", duration=durations[i], price=prices[i]) 
             for i in range(len(durations))]
    budget = 30
    m = 3
    return items, budget, m


def load_bilevel_from_json(path):
    """Load bilevel problem from a JSON file.

    Expects format:
    {
        "items": [{"name": "item0", "duration": 5, "price": 3}, ...],
        "budget": 100,
        "machines": 3
    }

    Args:
        path: Path to JSON file

    Returns:
        tuple: (items, budget, m)

    Raises:
        FileNotFoundError: If path does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        data = json.load(f)
    items = [Item(**it) for it in data.get("items", [])]
    budget = int(data.get("budget", 0))
    m = int(data.get("machines", 1))
    return items, budget, m


def print_bilevel(items, budget, m):
    """Print bilevel problem details to console.

    Args:
        items: List of Item objects
        budget: Total budget
        m: Number of machines
    """
    print(f"Bilevel problem: {len(items)} items, budget={budget}, machines={m}")
    for it in items:
        print(f"  {it.name}: duration={it.duration}, price={it.price}")


if __name__ == "__main__":
    args = sys.argv[1:]
    
    # Step 1: Print sample bilevel problem
    if len(args) >= 1 and args[0] in ("bilevel-step1", "step1"):
        items, budget, m = create_sample_bilevel()
        print_bilevel(items, budget, m)
        print("\nStep 1 complete: data structures and sample instance created.")
        sys.exit(0)
    
    # Step 2: Run leader heuristic + follower scheduling
    if len(args) >= 1 and args[0] in ("bilevel-step2", "step2"):
        try:
            items, budget, m = create_sample_bilevel()
            print_bilevel(items, budget, m)
            print("\nRunning leader knapsack heuristic (maximize total processing time)...")
            result = solve_leader_knapsack(items, budget, verbose=True)
        except Exception as exc:
            print("Error running leader knapsack:", exc)
            sys.exit(1)

        if 'x' in result:
            sol = result['x']
            print(f"Leader selection (counts): {sol}")
            print(f"Leader objective (total processing time): {result.get('obj')}")
            
            # Build follower jobs from selection
            processing_times = []
            for i, cnt in enumerate(sol):
                processing_times += [items[i].duration] * cnt

            if len(processing_times) == 0:
                print("No jobs selected by leader. Makespan is 0.")
                sys.exit(0)

            print(f"Follower will have {len(processing_times)} jobs. Running scheduling solver...")
            sched = solve_scheduling(len(processing_times), m, processing_times, verbose=True)
            if 'makespan' in sched:
                print(f"Makespan (Cmax) from leader heuristic selection: {sched['makespan']}")
            else:
                print("Scheduling solver failed:", sched)
        else:
            print("Leader knapsack did not return a solution:", result)
        sys.exit(0)
    
    # Step 4: Run branch-and-bound solver
    if len(args) >= 1 and args[0] in ("bilevel-step4-class", "step4-class"):
        from logger import create_logger
        
        items, budget, m = create_sample_bilevel()
        prices = [it.price for it in items]
        durations = [it.duration for it in items]
        problem = MainProblem(prices, durations, m, budget)
        
        # Create logger for this run
        logger = create_logger(instance_name="sample_bilevel")
        
        print("Running branch-and-bound on:", problem)
        # results
        res = run_bnb_classic(problem, max_nodes=40000, verbose=False, 
                             logger=logger, instance_name="sample_bilevel")
        print("\nBranch-and-bound result:")
        print(f"  Best makespan: {res['best_obj']}")
        print(f"  Best selection: {res['best_selection']}")
        print(f"  Best Schedule: {res['best_schedule']}")
        print(f"  Nodes explored: {res['nodes_explored']}")
        print(f"\nLog files saved to: logs/")
        sys.exit(0)
    






    # Test exact IP bound
    if len(args) >= 1 and args[0] == "test-exact-bound":
        from bnb import compute_ip_bound_exact
        from models import ProblemNode
        
        items, budget, m = create_sample_bilevel()
        prices = [it.price for it in items]
        durations = [it.duration for it in items]
        problem = MainProblem(prices, durations, m, budget)
        root = ProblemNode(None, depth=0, remaining_budget=budget, n_job_types=problem.n_job_types)
        
        print("Testing exact IP bound on root node:")
        print(f"Problem: {problem}")
        print(f"Root node: {root}")
        
        exact_val = compute_ip_bound_exact(problem, root, time_limit=5.0)
        print(f"\nExact IP bound (sum duration*ceil(x/m)): {exact_val}")
        print(f"This represents the optimistic makespan bound for remaining items")
        sys.exit(0)
    
    # Default: Run simple scheduling demo
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, "sample_data.json")

    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
        n_jobs = int(data.get("n_jobs"))
        n_machines = int(data.get("n_machines"))
        processing_times = list(data.get("processing_times"))
    else:
        # Fallback example
        print("Data file not found, using default example.")
        n_jobs = 8
        n_machines = 3
        processing_times = [2, 14, 4, 16, 6, 5, 3, 7]

    print(f"Jobs: {n_jobs}, Machines: {n_machines}")
    print(f"Processing times: {processing_times}")

    try:
        result = solve_scheduling(n_jobs, n_machines, processing_times, verbose=True)
    except Exception as exc:
        print("Error running model:", exc)
        sys.exit(1)

    if 'makespan' in result:
        print(f"Makespan (Cmax): {result['makespan']}")
        print("Assignments (job -> machine):")
        for j, m in enumerate(result['assignment']):
            print(f"  job {j} (p={processing_times[j]}) -> machine {m}")
    else:
        print("Solver did not produce a makespan; status:", 
              result.get('status'), result.get('message'))




'''
To do
1. Example Instances

Identify small illustrative instances for the algorithm that show different behaviors (e.g., easy, hard, degenerate, pathological).

Select instances that are small enough to follow by hand.

Prepare or visualize these instances (tables, diagrams, timelines, etc.).

2. Theoretical Preparation

Finalize the notation to be used in the thesis.

Prepare the description of model extensions (e.g., constraints, variants, assumptions).

Draft the structure of the main proof(s).

3. Algorithmic Components

Implement and compare different branching rules.

Investigate whether the upper bound can be computed via dynamic programming.

Develop a new or improved upper bound if it offers advantages.

4. Experimental Analysis

Generate different types of instances
(e.g., varying distributions for job lengths and prices).

Implement detailed logging to capture relevant metrics
(nodes explored, pruning rate, bound quality, runtime breakdowns, etc.).

Plan and execute the experiment series.

5. Model Extensions

Explore potential extensions of the problem formulation.

Investigate whether job counts can be bounded or controlled artificially
(e.g., upper limits, scaling, regularization).
'''