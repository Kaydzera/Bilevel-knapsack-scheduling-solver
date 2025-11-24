"""Branch-and-bound solver for the bilevel optimization problem.

This module implements the branch-and-bound algorithm that solves the
leader's problem by enumerating selections and using bounds to prune.
Includes exact IP bounds and heuristic initialization.
"""

import math
import time
from collections import deque
from typing import Optional

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    gp = None
    GRB = None

from solvers import solve_scheduling, solve_leader_knapsack, solve_scheduling_readable
from models import MainProblem, ProblemNode
from logger import BnBLogger


def greedy_fractional_total(items, budget):
    """Fast fractional knapsack for upper bound computation.
    
    Sorts items by duration/price ratio and greedily fills the budget,
    allowing fractional quantities. Used for quick bounds.
    
    Args:
        items: Iterable of objects with .duration and .price attributes
        budget: Available budget
        
    Returns:
        float: Fractional total duration achievable within budget
    """
    # Sort by duration/price ratio descending
    good = sorted(items, key=lambda it: (it.duration / it.price) if it.price > 0 else float('inf'), reverse=True)
    rem = budget
    total = 0.0
    for it in good:
        if rem <= 0:
            break
        if it.price <= rem:
            # Take integer count greedily
            count = rem // it.price
            total += count * it.duration
            rem -= count * it.price
        else:
            # Fractional take
            frac = rem / it.price
            total += frac * it.duration
            rem = 0
    return total


def initialize_bnb_state_from_heuristic(items, budget, m, verbose=False):
    """Initialize branch-and-bound with heuristic incumbent.
    
    Runs the leader knapsack heuristic (maximize total processing time)
    and evaluates the resulting follower makespan to provide a good
    starting incumbent for branch-and-bound.
    
    Args:
        items: List of objects with .duration and .price
        budget: Budget constraint
        m: Number of machines for follower
        verbose: Whether to show solver output
        
    Returns:
        dict with:
            - best_obj: Makespan from heuristic selection
            - best_selection: Counts list
            - proc_len: Number of jobs in heuristic selection
    """
    res = solve_leader_knapsack(items, budget, verbose=verbose)
    if 'x' not in res:
        return {'best_obj': 0.0, 'best_selection': [0]*len(items)}
    sel = res['x']

    # Build processing times for follower scheduling
    proc = []
    for i, cnt in enumerate(sel):
        proc += [items[i].duration] * cnt
    if len(proc) == 0:
        makespan = 0.0
    
    # Solve scheduling for follower
    else:
        sched = solve_scheduling(len(proc), m, proc, verbose=False)
        makespan = sched.get('makespan', 0.0)
    return {'best_obj': makespan, 'best_selection': sel, 'proc_len': len(proc)}


def compute_ip_bound_exact(problem: MainProblem, node: ProblemNode, time_limit=2.0):
    """Compute exact integer-program upper bound using grouping.
    
    Maximizes sum(duration[i] * ceil(x[i]/m)) subject to budget constraint
    on remaining items. Uses binary variables for copies and group variables
    to model ceil(x[i]/m).
    
    Args:
        problem: MainProblem instance
        node: Current ProblemNode in the search tree
        time_limit: Time limit for Gurobi solver
        
    Returns:
        float: Objective value (sum duration * ceil(x_i/m)) for remaining items
        
    Raises:
        ImportError: If Gurobi is not available
    """
    if gp is None:
        raise ImportError("Gurobi is not available. Please install gurobipy to use this function.")

    model = gp.Model("exact_ip_bound")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)

    # Extract remaining items and budget
    remaining_items = problem.items[node.depth:]
    rem_budget = node.remaining_budget
    m = problem.machines

    # For each item type i, compute max copies allowed
    K = []
    for it in remaining_items:
        if it.price <= 0:
            K.append(0)
        else:
            K.append(rem_budget // it.price)

    # Decision variables:
    # u_{i,k} = 1 if copy k of item i is selected
    # w_{i,g} = 1 if at least one copy in group g of item i is selected
    u = {}
    w = {}
    for i, Ki in enumerate(K):
        # G is the number of groups (ceil(Ki / m))
        # Each group represents a batch of up to m copies
        G = (Ki + m - 1) // m if Ki > 0 else 0
        for k in range(1, Ki + 1):
            u[i, k] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{k}")
        for g in range(1, G + 1):
            w[i, g] = model.addVar(vtype=GRB.BINARY, name=f"w_{i}_{g}")

    model.update()

    # Link group binaries w and copy binaries u
    for i, Ki in enumerate(K):
        G = (Ki + m - 1) // m if Ki > 0 else 0
        for g in range(1, G + 1):
            # Copies in this group
            first = (g - 1) * m + 1
            last = min(g * m, Ki)
            if first > last:
                # Empty group
                model.addConstr(w[i, g] == 0)
                continue
            # If any u in group is 1, then w must be 1
            model.addConstr(gp.quicksum(u[i, k] for k in range(first, last + 1)) >= w[i, g], name=f"w_lower_{i}_{g}")
            # Each u can only be 1 if w is 1
            for k in range(first, last + 1):
                model.addConstr(u[i, k] <= w[i, g], name=f"u_le_w_{i}_{g}_{k}")

    # Budget constraint
    budget_expr = gp.quicksum(remaining_items[i].price * u[i, k] for (i, k) in u.keys())
    model.addConstr(budget_expr <= rem_budget, name="budget")

    # Objective: maximize sum_i duration[i] * sum_g w[i,g]
    # (sum_g w[i,g] represents ceil(x[i]/m))
    obj = gp.quicksum(remaining_items[i].duration * w[i, g] for (i, g) in w.keys())
    model.setObjective(obj, GRB.MAXIMIZE)

    model.optimize()

    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.SUBOPTIMAL:
        return model.ObjVal
    else:
        return 0.0


def run_bnb_classic(problem: MainProblem, max_nodes=100000, verbose=False, 
                    logger: Optional[BnBLogger] = None, instance_name: str = "default"):
    """Branch-and-bound solver for the bilevel optimization problem.
    
    Uses depth-first search with exact IP bounds to prune the search tree.
    Evaluates leaf nodes by solving the follower's scheduling problem.
    
    Args:
        problem: MainProblem instance
        max_nodes: Maximum number of nodes to explore
        verbose: Whether to print progress to console
        logger: Optional BnBLogger instance for detailed logging
        instance_name: Name for the instance (used if logger is None)
        
    Returns:
        dict with:
            - best_obj: Best makespan found
            - best_selection: Counts for each job type in best solution
            - best_schedule: Readable schedule for best solution
            - nodes_explored: Number of nodes visited
    """
    # Create logger if not provided
    if logger is None:
        from logger import create_logger
        logger = create_logger(instance_name=instance_name)
    
    # Log problem characteristics
    problem_data = {
        "n_job_types": problem.n_job_types,
        "machines": problem.machines,
        "budget": problem.budget_total,
        "prices": problem.prices,
        "durations": problem.durations,
        "max_nodes": max_nodes
    }
    logger.start_run(problem_data)
    
    n = problem.n_job_types
    frontier = deque()
    seen = set()

    # Initialize with root node
    init_node = ProblemNode(None, depth=0, remaining_budget=problem.budget_total, n_job_types=n)
    frontier.append(init_node)
    seen.add((tuple(init_node.job_occurrences), init_node.depth, init_node.remaining_budget))

    # Get initial incumbent from heuristic
    logger.info("Computing initial heuristic solution...")
    init_state = initialize_bnb_state_from_heuristic(
        [type("_", (), {'duration':d,'price':p})() for d,p in zip(problem.durations, problem.prices)],
        problem.budget_total, problem.machines, verbose=False)
    incumbent = init_state['best_obj']
    incumbent_sel = init_state['best_selection']
    
    logger.log_incumbent_update(incumbent, incumbent_sel, node_count=0)
    if verbose:
        print(f"Initial incumbent from heuristic: makespan={incumbent}, selection={incumbent_sel}")

    logger.info(f"Starting branch-and-bound with max_nodes={max_nodes}")
    print(f"Starting branch-and-bound with max_nodes={max_nodes}")
    
    nodes = 0
    while frontier:
        node = frontier.pop()  # DFS
        nodes += 1
        
        # Log node visit
        node_info = {
            "depth": node.depth,
            "occurrences": node.job_occurrences,
            "remaining_budget": node.remaining_budget
        }
        logger.log_node_visit(node_info)
        
        if verbose:
            print(f"Visiting node: {node}")

        # If node is a leaf, evaluate exactly by solving scheduling
        if not node.extendable:
            proc = []
            for i, cnt in enumerate(node.job_occurrences):
                proc += [problem.durations[i]] * cnt
            if len(proc) == 0:
                makespan = 0.0
            else:
                sched = solve_scheduling(len(proc), problem.machines, proc, verbose=False)
                makespan = sched.get('makespan', None)
            
            if makespan is not None:
                logger.log_node_evaluated(makespan, node_info)
                
                if makespan > incumbent:
                    incumbent = makespan
                    incumbent_sel = list(node.job_occurrences)
                    logger.log_incumbent_update(incumbent, incumbent_sel, node_count=nodes)
                    if verbose:
                        print(f"New incumbent makespan={incumbent} selection={incumbent_sel}")
            continue

        # Compute upper bound for pruning
        try:
            # Solve exact IP bound for remaining items
            bound_start = time.time()
            solution_node_ip = compute_ip_bound_exact(problem, node, time_limit=2.0)
            bound_time = time.time() - bound_start

            # Add already committed jobs' contribution
            already_committed_length = 0.0
            for i in range(0, node.depth):
                occ = node.job_occurrences[i]
                dur = problem.durations[i]
                already_committed_length += dur * math.ceil(occ / problem.machines)

            bound = solution_node_ip + already_committed_length
            
            # Log bound computation
            logger.log_bound_computation(bound, "exact_ip", node.depth, bound_time)
            
            if verbose:
                print(f"Computed bound (optimistic makespan) = {bound}")
            
            # Prune if bound <= incumbent
            if bound <= incumbent:
                logger.log_node_pruned("bound_dominated", node_info)
                if verbose:
                    print(f"Pruning node: bound {bound} <= incumbent {incumbent}")
                continue
        except Exception as e:
            logger.warning(f"Bound computation failed: {e}")
            if verbose:
                print(f"Bound computation failed: {e}")

        # Expand node: right_child (increment), lower_child (commit)
        price = problem.prices[node.depth]
        right_child = node.increment_current(price)
        lower_child = node.commit_current()

        def try_add(child):
            key = (tuple(child.job_occurrences), child.depth, child.remaining_budget)
            if key in seen:
                return False
            seen.add(key)
            frontier.append(child)
            return True

        if right_child is not None:
            added = try_add(right_child)
        else:
            logger.log_node_pruned("budget_infeasible", 
                                  {"depth": node.depth, "child_type": "right"})
            
        if lower_child is not None:
            added = try_add(lower_child)

        if nodes >= max_nodes:
            logger.warning(f"Node limit {max_nodes} reached, stopping")
            if verbose:
                print(f"Node limit {max_nodes} reached, stopping")
            break
    
    # Solve the scheduling for the best selection found
    logger.info("Computing final schedule for best solution...")
    proc = []
    for i, cnt in enumerate(incumbent_sel):
        proc += [problem.durations[i]] * cnt
    sched = solve_scheduling_readable(len(proc), problem.machines, proc, verbose=False)
    
    # Prepare final result
    result = {
        'best_obj': incumbent, 
        'best_selection': incumbent_sel, 
        'best_schedule': sched, 
        'nodes_explored': nodes
    }
    
    # End logging
    logger.end_run(result)
    
    return result
