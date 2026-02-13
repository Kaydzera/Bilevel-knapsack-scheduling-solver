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
from knapsack_dp import CeilKnapsackSolver
from maxlpt_bound import precompute_maxlpt_dp_tables, compute_maxlpt_bound


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
        tuple: (bound, selection) - Objective value and selection for remaining items
        
    Raises:
        ImportError: If Gurobi is not available
    """
    if gp is None:
        raise ImportError("Gurobi is not available. Please install gurobipy to use this function.")

    # Extract remaining items and budget
    remaining_items = problem.items[node.depth:]
    rem_budget = node.remaining_budget
    m = problem.machines
    
    bound, selection = compute_ip_bound_direct(problem.items, node.depth, rem_budget, m, time_limit)
    return bound, selection


# TODO: Review and understand this code line by line
def compute_ip_bound_direct(items, depth, rem_budget, m, time_limit=2.0):
    """Compute exact integer-program upper bound using grouping (direct version).
    
    This is the same as compute_ip_bound_exact but takes direct parameters
    instead of problem and node objects, making it easier to test.
    
    Maximizes sum(duration[i] * ceil(x[i]/m)) subject to budget constraint
    on remaining items (from depth onwards). Uses binary variables for copies
    and group variables to model ceil(x[i]/m).
    
    Args:
        items: Full list of items (with .duration and .price attributes)
        depth: Current depth in the search tree
        rem_budget: Remaining budget available
        m: Number of machines
        time_limit: Time limit for Gurobi solver
        
    Returns:
        float: Objective value (sum duration * ceil(x_i/m))
        
    Raises:
        ImportError: If Gurobi is not available
    """
    if gp is None:
        raise ImportError("Gurobi is not available. Please install gurobipy to use this function.")

    # Extract remaining items from depth+1 onwards (truly undecided items)
    # The item at depth is currently being decided and should be in already_committed
    remaining_items = items[depth+1:]

    model = gp.Model("exact_ip_bound")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)

    # For each item type i, compute max copies allowed
    K = []
    for it in remaining_items:
        if it.price <= 0:
            # Infinite copies possible
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
        
        # Force sequential packing: group g can only be used if group g-1 is full
        # This ensures ceil(x[i]/m) = sum_g w[i,g]
        for g in range(2, G + 1):
            # If group g is used (w[i,g]=1), then all copies in group g-1 must be selected
            first_prev = (g - 2) * m + 1
            last_prev = min((g - 1) * m, Ki)
            if first_prev <= last_prev:
                # w[i,g] <= (1/m) * sum of all u in previous group
                # Equivalently: m * w[i,g] <= sum of u in previous group
                model.addConstr(
                    m * w[i, g] <= gp.quicksum(u[i, k] for k in range(first_prev, last_prev + 1)),
                    name=f"seq_pack_{i}_{g}"
                )

    # Budget constraint
    budget_expr = gp.quicksum(remaining_items[i].price * u[i, k] for (i, k) in u.keys())
    model.addConstr(budget_expr <= rem_budget, name="budget")

    # Objective: maximize sum_i duration[i] * sum_g w[i,g]
    # (sum_g w[i,g] represents ceil(x[i]/m))
    obj = gp.quicksum(remaining_items[i].duration * w[i, g] for (i, g) in w.keys())
    model.setObjective(obj, GRB.MAXIMIZE)

    model.optimize()

    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.SUBOPTIMAL:
        #return the optimal objective value and the optimal selection

        # Extract the solution
        selection = [0] * len(remaining_items)
        for (i, k) in u.keys():
            if u[i, k].X > 0.5:
                selection[i] += 1
        
        return (model.ObjVal, selection)


        #return model.ObjVal
    else:
        return 0.0


def run_bnb_classic(problem: MainProblem, max_nodes=100000, verbose=False, 
                    logger: Optional[BnBLogger] = None, instance_name: str = "default",
                    enable_logging: bool = True, bound_type: str = 'ceiling'):
    """Branch-and-bound solver for the bilevel optimization problem.
    
    Uses depth-first search with exact IP bounds to prune the search tree.
    Evaluates leaf nodes by solving the follower's scheduling problem.
    
    Args:
        problem: MainProblem instance
        max_nodes: Maximum number of nodes to explore
        verbose: Whether to print progress to console
        logger: Optional BnBLogger instance for detailed logging
        instance_name: Name for the instance (used if logger is None)
        enable_logging: Whether to create logs (default: True)
        bound_type: Type of upper bound to use ('ceiling' or 'maxlpt')
        
    Returns:
        dict with:
            - best_obj: Best makespan found
            - best_selection: Counts for each job type in best solution
            - best_schedule: Readable schedule for best solution
            - nodes_explored: Number of nodes visited
    """
    # Create logger if not provided and logging is enabled
    if logger is None and enable_logging:
        from logger import create_logger
        logger = create_logger(instance_name=instance_name)
    elif logger is None:
        # Create a no-op logger when logging is disabled
        from logger import NoOpLogger
        logger = NoOpLogger()
    
    # Log problem characteristics
    problem_data = {
        "n_job_types": problem.n_job_types,
        "machines": problem.machines,
        "budget": problem.budget_total,
        "prices": problem.prices,
        "durations": problem.durations,
        "max_nodes": max_nodes,
        "bound_type": bound_type
    }
    logger.start_run(problem_data)
    
         # Validate bound_type
    if bound_type not in ['ceiling', 'maxlpt']:
        raise ValueError(f"Invalid bound_type: {bound_type}. Must be 'ceiling' or 'maxlpt'.")
    
    # Track timing statistics for bound computations
    
    #total_ceil_time = 0.0
    #total_ip_time = 0.0
    #bound_computation_count = 0
    
    n = problem.n_job_types
    frontier = deque()
    seen = set()

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

    # Initialize root node at depth=0 with heuristic solution
    # At depth 0, only job at index 0 is "paid for"
    budget_spent_at_depth_0 = incumbent_sel[0] * problem.prices[0]
    remaining_budget_at_depth_0 = problem.budget_total - budget_spent_at_depth_0
    
    init_node = ProblemNode(incumbent_sel, depth=0, remaining_budget=remaining_budget_at_depth_0, 
                           n_job_types=n, m=problem.machines, branch_type='root')
    frontier.append(init_node)

    # Track seen nodes to avoid duplicates, this part is obsolete
    seen.add((tuple(init_node.job_occurrences), init_node.depth, init_node.remaining_budget))

   
    # Initialize CeilKnapsackSolver with REVERSED item order
    # This allows us to query the last N items by querying the first N items in reversed solver
    # Example: if depth=1 (deciding item 1), remaining items are [2,3,4,...]
    # In reversed order, these become [last, ..., 4, 3, 2] which are the first items
    
    ceil_solver = None
    maxlpt_dp_solvers = None
    
    if bound_type == 'ceiling':
        logger.info("Initializing CeilKnapsackSolver with reversed item order...")
        ceil_solver = CeilKnapsackSolver(
            costs=list(reversed(problem.prices)),
            durations=list(reversed(problem.durations)),
            m=problem.machines,
            max_budget=problem.budget_total
        )
        logger.info("CeilKnapsackSolver initialized successfully")
    elif bound_type == 'maxlpt':
        logger.info("Precomputing Max-LPT DP tables for all depths...")
        preprocessing_start = time.time()
        maxlpt_dp_solvers = precompute_maxlpt_dp_tables(
            problem.durations,
            problem.prices,
            problem.budget_total
        )
        preprocessing_time = time.time() - preprocessing_start
        logger.info(f"Max-LPT preprocessing completed in {preprocessing_time:.3f}s "
                   f"({n} DP tables computed)")
    

    logger.info(f"Starting branch-and-bound with max_nodes={max_nodes}")
    print(f"Starting branch-and-bound with max_nodes={max_nodes}")

    #Node counter
    nodes = 0

    while frontier:
        node = frontier.pop()  # DFS
        nodes += 1
        
        # Log node visit
        node_info = {
            "depth": node.depth,
            "occurrences": node.job_occurrences,
            "remaining_budget": node.remaining_budget,
            "node_type": node._branch_type
        }
        logger.log_node_visit(node_info)
    

        # Helper function to add children to frontier
        def try_add(child):
            key = (tuple(child.job_occurrences), child.depth, child.remaining_budget)
            if key in seen:
                #This is not suppoed to happen. Throw an Exception
                raise Exception("Duplicate node encountered in branch-and-bound search")
            seen.add(key)
            frontier.append(child)
            return True

        # Get which children can be created based on branch type
        can_create = node.can_create_children()
        price = problem.prices[node.depth]
        duration = problem.durations[node.depth]
        
        # Try to add right child (increment current item) if allowed
        if can_create['right']:
            right_child = node.increment_current(price, duration=duration)
            if right_child is not None:
                added = try_add(right_child)
            else:
                logger.log_node_pruned("budget_infeasible", nodes)
        
        # Try to add left child (decrement current item) if allowed
        if can_create['left']:
            left_child = node.decrement_current(price)
            if left_child is not None:
                added = try_add(left_child)
            # If decrement returns None, it means occurrences[depth] is already 0
        # Depth semantics: depth=N means we are DECIDING item N
        # The current item at depth has a committed amount in occurrences[depth]

        # At depth n-1, all items are decided, treat as leaf
        if node.depth == n - 1:
            # All items are decided at this depth, evaluate as leaf directly
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
        


        # Compute bound for potentially adding lower child (commit to next depth)
        if can_create['lower']:
            try:
                # Compute upper bound based on selected method
                if bound_type == 'ceiling':
                    # Method 1: CeilKnapsackSolver (fast DP method with ceiling)
                    # Remaining items are from depth+1 onwards
                    num_remaining_items = n - (node.depth + 1)
                    if num_remaining_items > 0:
                        # Query the reversed solver for the first num_remaining_items
                        # which correspond to the last num_remaining_items in original order
                        result_ceil = ceil_solver.reconstruct(num_remaining_items, node.remaining_budget)
                        solution_node = result_ceil['max_value']
                    else:
                        solution_node = 0.0
                    bound_method_name = "ceil_knapsack_dp"
                    
                elif bound_type == 'maxlpt':
                    # Method 2: Max-LPT bound (tighter, 3/4-approximation)
                    # Compute makespan contribution from remaining items using Max-LPT
                    if node.depth + 1 < n:
                        solution_node = compute_maxlpt_bound(
                            problem.durations,
                            problem.prices,
                            node.remaining_budget,
                            problem.machines,
                            node.depth + 1,  # Remaining items start from depth+1
                            maxlpt_dp_solvers,
                            0.0  # We add already_committed separately below
                        )
                    else:
                        solution_node = 0.0
                    bound_method_name = "maxlpt"

                # Add already committed jobs' contribution using optimized node tracking
                # For 'right' branch: get_already_committed_length() returns max directly
                # For 'root'/'left' branch: need to pass duration to distribute jobs at depth
                if node._branch_type == 'right':
                    already_committed_length = node.get_already_committed_length()
                else:
                    already_committed_length = node.get_already_committed_length(duration_at_depth=duration)

                bound = solution_node + already_committed_length
                
                # Log bound computation (one line)
                logger.log_bound_computation(bound, bound_method_name, node.depth, 
                                            solution_node, already_committed_length)
                
                # Prune if bound <= incumbent
                if bound <= incumbent:
                    logger.log_node_pruned("bound_dominated", nodes)
                    continue
            except Exception as e:
                logger.warning(f"Bound computation failed: {e}")

            # Expand node: lower_child (commit)
            # Pass the price at the NEW depth for budget calculation
            price_at_new_depth = problem.prices[node.depth + 1] if node.depth + 1 < n else None
            lower_child = node.commit_current(duration=duration, price_at_new_depth=price_at_new_depth)            
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
    
    # Calculate runtime before creating result
    runtime = None
    if hasattr(logger, 'metrics') and 'start_time' in logger.metrics:
        runtime = time.time() - logger.metrics['start_time']
    
    # Prepare final result
    result = {
        'best_obj': incumbent, 
        'best_selection': incumbent_sel, 
        'best_schedule': sched, 
        'nodes_explored': nodes,
        'runtime': runtime
    }
    
    # End logging
    logger.end_run(result)
    
    return result





if __name__ == "__main__":    # Simple test run
    print("=" * 70)
    print("TESTING compute_ip_bound_exact")
    print("=" * 70)
    

    from models import MainProblem

    # Instance 2: Uniform ratios - all items equally attractive
    print("\nInstance: Uniform Ratios")
    items = [
        type("_", (), {'duration': 4, 'price': 2})(),
        type("_", (), {'duration': 6, 'price': 3})(),
        type("_", (), {'duration': 8, 'price': 4})(),
        type("_", (), {'duration': 10, 'price': 5})()
    ]
    prices = [it.price for it in items]
    durations = [it.duration for it in items]
    m = 2
    budget = 12
    problem = MainProblem(prices, durations, m, budget)
    print(f"Items: {list(zip(durations, prices))}")
    print(f"Machines: {m}, Budget: {budget}")
    
    depth = 3
    remaining_budget = 2
    print(f"\n--- Depth {depth}, Remaining Budget {remaining_budget} ---")
    remaining_items = items[depth:]
    print(f"Remaining items from depth {depth}: {[(it.duration, it.price) for it in remaining_items]}")
    
    bound, selection = compute_ip_bound_direct(items, depth=depth, rem_budget=remaining_budget, m=m, time_limit=5.0)
    print(f"Computed exact IP bound: {bound}")
    print(f"Optimal selection for remaining items: {selection}")

    print("\n" + "=" * 70)

    already_committed_length = 0.0
    job_occurrences = [1, 0, 2, 0]
    for i in range(0, depth):
        occ = job_occurrences[i]
        dur = items[i].duration
        print(f"Committed item {i}: occurrences={occ}, duration={dur}")
        already_committed_length += dur * math.ceil(occ / problem.machines)

    final_bound = bound + already_committed_length
    print(f"Final bound including committed jobs: {final_bound}")

    print("\n" + "=" * 70)

