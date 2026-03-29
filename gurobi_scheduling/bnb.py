"""Branch-and-bound solver for the bilevel optimization problem
This module implements the branch-and-   bound algorithm that solves the
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
from maxlpt_bound import precompute_maxlpt_dp_tables, compute_maxlpt_bound, UnboundedKnapsackDP


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


def solve_max_lpt_with_selection(durations, prices, budget, m, dp_solver):
    """Solve Max-LPT problem and return both the value and selection.
    
    The Max-LPT problem finds the job selection that maximizes the makespan
    when jobs are scheduled using the Longest Processing Time (LPT) rule.
    
    This version reconstructs the actual selection, not just the makespan value.
    
    Args:
        durations: List of job durations
        prices: List of job prices  
        budget: Total budget available
        m: Number of machines
        dp_solver: Pre-computed UnboundedKnapsackDP solver
        
    Returns:
        dict with:
            - makespan: Maximum makespan achievable with LPT scheduling
            - selection: List of item counts in optimal solution
            - last_job_type: Index of the job type used as "last job" on each machine
    """
    n = len(durations)
    z_star = 0.0
    best_J = None
    best_B_i = None
    
    for J in range(n):
        # Compute per-machine budget
        B_i = budget // m
        B_hat = budget - m * B_i
        
        # Check if remaining budget is sufficient for item J
        if B_hat < prices[J]:
            # Adjust B_i downward until we have enough remaining budget
            k = 0
            while B_hat < prices[J] and B_i > 0:
                k += 1
                B_i = (budget - k * m) // m
                B_hat = budget - m * B_i
                
                # Safety check to prevent infinite loop
                if B_i < 0 or B_hat < 0:
                    break
            
            # If still not enough budget, skip this item
            if B_hat < prices[J]:
                continue
        
        # Evaluate this configuration
        dp_value = dp_solver.query(J + 1, B_i)
        z = dp_value + durations[J]
        
        if z > z_star:
            z_star = z
            best_J = J
            best_B_i = B_i
    
    # Reconstruct the selection for the best configuration
    if best_J is None:
        # No feasible solution found
        return {
            'makespan': 0.0,
            'selection': [0] * n,
            'last_job_type': None
        }
    
    # Get the counts from DP for filling m machines with budget B_i each               
    dp_result = dp_solver.reconstruct(best_J + 1, best_B_i)
    base_counts = dp_result['counts']
    
    # Create full selection array initialized with zeros
    selection = [0] * n
    
    # Fill in the counts for items 0 through best_J (multiply by m)
    for i in range(len(base_counts)):
        selection[i] = base_counts[i] * m
    
    # Add one more copy of the last job type
    selection[best_J] += 1
    
    return {
        'makespan': z_star,
        'selection': selection,
        'last_job_type': best_J
    }


def initialize_bnb_state_from_maxlpt(problem: MainProblem, bound_type: str = 'ceiling',
                                     maxlpt_dp_solvers=None, verbose=False):
    """Initialize branch-and-bound with Max-LPT incumbent.
    
    Solves the Max-LPT problem to find the best selection according to the
    LPT scheduling rule, then evaluates the actual makespan using solve_scheduling.
    
    If the Max-LPT upper bound equals the actual makespan, the problem is proven
    optimal and we can skip the branch-and-bound search entirely.
    
    Args:
        problem: MainProblem instance
        bound_type: Type of bound being used ('ceiling' or 'maxlpt')
        maxlpt_dp_solvers: Optional pre-computed DP solvers (if bound_type='maxlpt')
        verbose: Whether to show solver output
        
    Returns:
        dict with:
            - best_obj: Makespan from Max-LPT selection
            - best_selection: Counts list
            - maxlpt_bound: Upper bound from Max-LPT
            - starting_incumbent_optimal: True if makespan equals Max-LPT bound
            - proc_len: Number of jobs in selection
    """
    n = problem.n_job_types
    budget = problem.budget_total
    m = problem.machines
    
    # Get or create the DP solver for depth 0 (all items)
    if bound_type == 'maxlpt' and maxlpt_dp_solvers is not None:
        # Reuse pre-computed solver for depth 0
        dp_solver = maxlpt_dp_solvers[0]
    else:
        # Compute just one DP table for all items
        dp_solver = UnboundedKnapsackDP(problem.durations, problem.prices, budget)
    
    # Solve Max-LPT problem to get the selection
    maxlpt_result = solve_max_lpt_with_selection(
        problem.durations,
        problem.prices,
        budget,
        m,
        dp_solver
    )
    
    maxlpt_bound = maxlpt_result['makespan']
    selection = maxlpt_result['selection']
    
    # Build processing times for actual scheduling
    proc = []
    for i, cnt in enumerate(selection):
        proc += [problem.durations[i]] * cnt
    
    if len(proc) == 0:
        actual_makespan = 0.0
    else:
        # Solve actual scheduling problem
        sched = solve_scheduling(len(proc), m, proc, verbose=verbose)
        actual_makespan = sched.get('makespan', 0.0)
    
    # Check if the solution is proven optimal
    # Max-LPT provides an upper bound, so if actual makespan equals it, we're done
    starting_incumbent_optimal = abs(actual_makespan - maxlpt_bound) < 1e-6
    
    return {
        'best_obj': actual_makespan,
        'best_selection': selection,
        'maxlpt_bound': maxlpt_bound,
        'starting_incumbent_optimal': starting_incumbent_optimal,
        'proc_len': len(proc)
    }


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


class LeafTimeoutError(Exception):
    """Raised when a single leaf node evaluation exceeds the timeout limit."""
    pass


def run_bnb_classic(problem: MainProblem, max_nodes=1_000_000_000, verbose=False, 
                    logger: Optional[BnBLogger] = None, instance_name: str = "default",
                    enable_logging: bool = False, bound_type: str = 'ceiling',
                    time_limit: float = None, leaf_timeout: float = 900.0):
    """Branch-and-bound solver for the bilevel optimization problem.
    
    Uses depth-first search with exact IP bounds to prune the search tree.
    Evaluates leaf nodes by solving the follower's scheduling problem.
    
    Args:
        problem: MainProblem instance
        max_nodes: Maximum number of nodes to explore
        verbose: Whether to print progress to console
        logger: Optional BnBLogger instance for detailed logging
        instance_name: Name for the instance (used if logger is None)
        enable_logging: Whether to create logs (default: False)
        bound_type: Type of upper bound to use ('ceiling' or 'maxlpt')
        time_limit: Maximum runtime in seconds (None = no limit)
        leaf_timeout: Maximum time for single leaf evaluation (default: 900s = 15 min)
        
    Returns:
        dict with:
            - best_obj: Best makespan found
            - best_selection: Counts for each job type in best solution
            - best_schedule: Readable schedule for best solution
            - nodes_explored: Number of nodes visited
            - runtime: Total runtime in seconds
            - proven_optimal: Whether solution is proven optimal
            - initial_makespan: Initial makespan before BnB search
    """
    # Track start time independently of logger
    start_time = time.time()
    print(f"[BnB] Starting run_bnb_classic: bound_type={bound_type}, time_limit={time_limit}")
    
    # Always use NoOpLogger when logging is disabled
    if not enable_logging:
        from logger import NoOpLogger
        logger = NoOpLogger()
    elif logger is None:
        from logger import create_logger
        logger = create_logger(instance_name=instance_name)
    
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

    # Initialize CeilKnapsackSolver or Max-LPT DP tables based on bound_type
    # This must be done BEFORE initialization so we can reuse tables if needed
    
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
        print(f"[BnB] CeilKnapsackSolver initialized in {time.time()-start_time:.2f}s")
        logger.info("CeilKnapsackSolver initialized successfully")
        # Check if we've already exceeded time limit during initialization
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            logger.warning(f"Time limit exceeded during CeilKnapsackSolver initialization")
            print(f"[BnB] ERROR: Time limit exceeded during CeilKnapsackSolver initialization")
            return {
                'best_obj': None,
                'best_selection': None,
                'best_schedule': None,
                'nodes_explored': 0,
                'runtime': time.time() - start_time,
                'proven_optimal': False,
                'initial_makespan': None
            }
    elif bound_type == 'maxlpt':
        logger.info("Precomputing Max-LPT DP tables for all depths...")
        preprocessing_start = time.time()
        maxlpt_dp_solvers = precompute_maxlpt_dp_tables(
            problem.durations,
            problem.prices,
            problem.budget_total
        )
        preprocessing_time = time.time() - preprocessing_start
        print(f"[BnB] Max-LPT preprocessing completed in {preprocessing_time:.3f}s")
        logger.info(f"Max-LPT preprocessing completed in {preprocessing_time:.3f}s "
                   f"({n} DP tables computed)")
        # Check if we've already exceeded time limit during preprocessing
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            logger.warning(f"Time limit exceeded during Max-LPT preprocessing")
            return {
                'best_obj': None,
                'best_selection': None,
                'best_schedule': None,
                'nodes_explored': 0,
                'runtime': time.time() - start_time,
                'proven_optimal': False,
                'initial_makespan': None
            }

    # Get initial incumbent using Max-LPT approach
    logger.info("Computing initial Max-LPT solution...")

    init_state = initialize_bnb_state_from_maxlpt(
        problem, bound_type=bound_type, maxlpt_dp_solvers=maxlpt_dp_solvers, verbose=False)
    print(f"[BnB] Initial solution computed in {time.time()-start_time:.2f}s: makespan={init_state.get('best_obj')}")
    
    # Check if we've already exceeded time limit after initial solution
    if time_limit is not None and (time.time() - start_time) >= time_limit:
        logger.warning(f"Time limit exceeded during initial solution computation")
        print(f"[BnB] ERROR: Time limit exceeded during initial solution computation")
        return {
            'best_obj': init_state.get('best_obj'),
            'best_selection': init_state.get('best_selection'),
            'best_schedule': None,
            'nodes_explored': 0,
            'runtime': time.time() - start_time,
            'proven_optimal': False,
            'initial_makespan': init_state.get('best_obj')
        }
    
    incumbent = init_state['best_obj']
    incumbent_sel = init_state['best_selection']
    maxlpt_bound = init_state['maxlpt_bound']
    starting_incumbent_optimal = init_state['starting_incumbent_optimal']
    initial_incumbent = incumbent  # Save initial solution for comparison
    
    logger.log_incumbent_update(incumbent, incumbent_sel, node_count=0)
    if verbose:
        print(f"Initial incumbent from Max-LPT: makespan={incumbent}, selection={incumbent_sel}")
        print(f"Max-LPT upper bound: {maxlpt_bound}")
        if starting_incumbent_optimal:
            print(f"✓ PROVEN OPTIMAL: Max-LPT bound equals actual makespan!")
    
    # If the solution is proven optimal, we can skip BnB entirely
    if starting_incumbent_optimal:
        logger.info("Solution proven optimal by Max-LPT, skipping branch-and-bound")
        if verbose:
            print("Solution proven optimal by Max-LPT, skipping branch-and-bound")
        
        # Solve the scheduling for the best selection found
        logger.info("Computing final schedule for optimal solution...")
        proc = []
        for i, cnt in enumerate(incumbent_sel):
            proc += [problem.durations[i]] * cnt
        sched = solve_scheduling_readable(len(proc), problem.machines, proc, verbose=False)
        
        # Use solver-local start_time so runtime is consistent for all logger types
        runtime = time.time() - start_time
        
        # Prepare final result
        result = {
            'best_obj': incumbent, 
            'best_selection': incumbent_sel, 
            'best_schedule': sched, 
            'nodes_explored': 0,  # No BnB nodes explored
            'runtime': runtime,
            'proven_optimal': True,
            'initial_makespan': initial_incumbent
        }
        
        # End logging
        logger.end_run(result)
        
        return result
    
    # Continue with branch-and-bound if not proven optimal

    # Initialize root node at depth=0 with Max-LPT solution
    # At depth 0, only job at index 0 is "paid for"
    budget_spent_at_depth_0 = incumbent_sel[0] * problem.prices[0]
    remaining_budget_at_depth_0 = problem.budget_total - budget_spent_at_depth_0
    
    init_node = ProblemNode(incumbent_sel, depth=0, remaining_budget=remaining_budget_at_depth_0, 
                           n_job_types=n, m=problem.machines, branch_type='root')
    frontier.append(init_node)

    logger.info(f"Starting branch-and-bound with max_nodes={max_nodes}")
    if verbose:
        print(f"Starting branch-and-bound with max_nodes={max_nodes}")
    print(f"[BnB] Entering main loop with frontier size={len(frontier)}")

    #Node counter
    nodes = 0
    last_progress_time = time.time()  # Track last progress report
    leaf_nodes_evaluated = 0  # Track number of leaf nodes evaluated
    while frontier:
        # Periodic progress report every 1 minute
        if time.time() - last_progress_time >= 60:
            elapsed = time.time() - start_time
            print(f"[BnB Progress] nodes={nodes:,},  frontier={len(frontier):,}, leaf_nodes={leaf_nodes_evaluated:,}, incumbent={incumbent}, elapsed={elapsed:.1f}s")
            last_progress_time = time.time()
            
        # Check node limit before processing next node
        if nodes >= max_nodes:
            logger.warning(f"Node limit {max_nodes} reached, stopping")
            if True:  # Always print node limit reached regardless of verbose flag
                print(f"Node limit {max_nodes} reached, stopping")
            break
        
        # Check time limit if specified
        if time_limit is not None:
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                logger.warning(f"Time limit {time_limit}s reached after {elapsed:.1f}s, stopping")
                # Always print timeout regardless of verbose flag
                print(f"[BnB] Time limit {time_limit}s reached after {elapsed:.1f}s at node {nodes}, stopping")
                break
            
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
        
        # OPTIMIZATION: If we're at depth n-1 and can create right child, 
        # jump directly to maximum budget-feasible count instead of incrementing by 1
        # This avoids evaluating all intermediate leaf nodes that would have lower makespan
        if can_create['right'] and node.depth == n - 1:
            # Calculate maximum additional copies we can afford for current item
            max_additional_copies = node.remaining_budget // price
            
            if max_additional_copies > 0:
                # Create the final optimized state directly
                new_occ = list(node.job_occurrences)
                new_occ[node.depth] += max_additional_copies
                new_budget = node.remaining_budget - (max_additional_copies * price)
                
                # Update machine loads if they exist
                new_loads = None
                if node._machine_loads is not None and len(node._machine_loads) > 0:
                    new_loads = list(node._machine_loads)
                    # Add all the additional jobs to machines
                    for _ in range(max_additional_copies):
                        # Add to least loaded machine (first in sorted list)
                        new_loads[0] += duration
                        val = new_loads.pop(0)
                        # Insert back in sorted order - find correct position
                        inserted = False
                        for i in range(len(new_loads)):
                            if val <= new_loads[i]:
                                new_loads.insert(i, val)
                                inserted = True
                                break
                        if not inserted:
                            new_loads.append(val)
                    
                    # Compute LPT upper bound and check against incumbent
                    lpt_makespan = max(new_loads)
                    if lpt_makespan <= incumbent:
                        logger.log_node_pruned("lpt_bound_dominated", nodes)
                        if max_additional_copies > 1:
                            logger.info(f"LPT pruning: Skipped +{max_additional_copies} for item {node.depth} " +
                                       f"(LPT makespan {lpt_makespan} <= incumbent {incumbent})")
                        continue
                
                # Create the optimized right child directly using ProblemNode constructor
                optimized_right_child = ProblemNode(
                    new_occ, node.depth, new_budget, node.n_job_types,
                    m=node._m, branch_type='right', machine_loads=new_loads
                )
                try_add(optimized_right_child)
                
                if max_additional_copies > 1:
                    logger.info(f"Optimization: Jumped +{max_additional_copies} for item {node.depth} " +
                               f"(skipped {max_additional_copies-1} intermediate evaluations)")
            else:
                logger.log_node_pruned("budget_infeasible", nodes)
        
        # Regular right child increment for all other depths  
        elif can_create['right']:
            right_child = node.increment_current(price, duration=duration)
            if right_child is not None:
                added = try_add(right_child)
            else:
                logger.log_node_pruned("budget_infeasible", nodes)
        
        # Try to add left child (decrement current item) if allowed.
        # At leaf depth (n-1), left descendants only remove jobs from a fully decided
        # solution and cannot improve the maximization objective, so skip them.
        if can_create['left'] and node.depth != n - 1:
            left_child = node.decrement_current(price)
            if left_child is not None:
                added = try_add(left_child)
            # If decrement returns None, it means occurrences[depth] is already 0
        # Depth semantics: depth=N means we are DECIDING item N
        # The current item at depth has a committed amount in occurrences[depth]

        # At depth n-1, all items are decided, treat as leaf
        if node.depth == n - 1:
            # All items are decided at this depth, evaluate as leaf directly
            leaf_nodes_evaluated += 1  # Increment leaf node counter
            proc = []
            for i, cnt in enumerate(node.job_occurrences):
                proc += [problem.durations[i]] * cnt
            if len(proc) == 0:
                makespan = 0.0
            else:
                leaf_start = time.time()
                # Pass leaf_timeout directly to Gurobi solver to prevent hanging
                sched = solve_scheduling(len(proc), problem.machines, proc, 
                                       time_limit=leaf_timeout, verbose=False)
                makespan = sched.get('makespan', None)
                leaf_time = time.time() - leaf_start
                
                # Check for leaf timeout (should rarely trigger now that Gurobi has timeout)
                if leaf_timeout is not None and leaf_time > leaf_timeout:
                    print(f"\n{'='*80}")
                    print(f"LEAF TIMEOUT DETAILED ANALYSIS")
                    print(f"{'='*80}")
                    print(f"Node {nodes} took {leaf_time:.1f}s (limit: {leaf_timeout:.1f}s)")
                    print(f"Node details:")
                    print(f"  - Depth: {node.depth}")
                    print(f"  - Job occurrences: {node.job_occurrences}")
                    print(f"  - Remaining budget: {node.remaining_budget}")
                    print(f"  - Branch type: {node._branch_type}")
                    print(f"  - Machine loads: {node._machine_loads}")
                    print(f"  - Number of job types: {node.n_job_types}")
                    print(f"  - Number of machines: {node._m}")
                    print(f"  - Total jobs to schedule: {len(proc)} (sum: {sum(node.job_occurrences)})")
                    print(f"  - Job durations in scheduling: {proc[:10]}{'...' if len(proc) > 10 else ''}")
                    print(f"  - Problem prices: {problem.prices}")
                    print(f"  - Problem durations: {problem.durations}")
                    if hasattr(node, 'extendable'):
                        print(f"  - Node extendable: {node.extendable}")
                    print(f"{'='*80}")
                    print(f"TERMINATING TEST due to leaf timeout")
                    print(f"{'='*80}\n")
                    raise LeafTimeoutError(f"Leaf node evaluation exceeded {leaf_timeout}s limit")
                
                # Log leaf solving time
                logger.info(f"Leaf node {nodes} solved in {leaf_time:.3f}s (jobs: {len(proc)})")
                
                # Print to console if solving took more than a minute
                if leaf_time > 60.0:
                    print(f"WARNING: Leaf node {nodes} took {leaf_time:.3f}s to solve ({len(proc)} jobs)")
            
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
                        dp_start = time.time()
                        result_ceil = ceil_solver.reconstruct(num_remaining_items, node.remaining_budget)
                        solution_node = result_ceil['max_value']
                        dp_time = time.time() - dp_start
                        if dp_time > 0.1:  # Log slow DP queries
                            print(f"[BnB] Slow DP query: {dp_time:.2f}s, items={num_remaining_items}, budget={node.remaining_budget}")
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
            except Exception as e:
                msg = (
                    "Critical error: bound computation failed "
                    f"at node {nodes} (depth={node.depth}, "
                    f"bound_type={bound_type}, remaining_budget={node.remaining_budget})."
                )
                logger.warning(f"{msg} Original error: {e}")
                raise RuntimeError(msg) from e

            if bound is not None:
                # Logging should never affect solver correctness.
                try:
                    logger.log_bound_computation(
                        bound, bound_method_name, node.depth,
                        solution_node, already_committed_length
                    )
                except Exception as e:
                    logger.warning(f"Bound logging failed: {e}")

                # Prune if bound <= incumbent
                if bound <= incumbent:
                    logger.log_node_pruned("bound_dominated", nodes)
                    continue

            # Expand node: lower_child (commit)
            # Pass the price at the NEW depth for budget calculation
            price_at_new_depth = problem.prices[node.depth + 1] if node.depth + 1 < n else None
            lower_child = node.commit_current(duration=duration, price_at_new_depth=price_at_new_depth)            
            if lower_child is not None:
                added = try_add(lower_child)
    
    # Solve the scheduling for the best selection found
    logger.info("Computing final schedule for best solution...")
    proc = []
    for i, cnt in enumerate(incumbent_sel):
        proc += [problem.durations[i]] * cnt
    sched = solve_scheduling_readable(len(proc), problem.machines, proc, verbose=False)
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Prepare final result
    result = {
        'best_obj': incumbent, 
        'best_selection': incumbent_sel, 
        'best_schedule': sched, 
        'nodes_explored': nodes,
        'runtime': runtime,
        'proven_optimal': False,  # Not proven optimal through Max-LPT
        'initial_makespan': initial_incumbent
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

