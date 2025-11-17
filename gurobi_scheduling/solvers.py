"""Gurobi optimization solvers for scheduling and knapsack problems.

This module contains Gurobi-based solvers used in the bilevel optimization:
- solve_scheduling: Minimize makespan for job assignment to identical machines
- solve_leader_knapsack: Maximize total processing time within budget (heuristic)
"""
import sys
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    gp = None
    GRB = None


def solve_scheduling(n_jobs, n_machines, processing_times, time_limit=None, verbose=True):
    """Solve the parallel machine scheduling problem to minimize makespan.
    
    Assigns n jobs to m identical machines to minimize the maximum load (makespan).
    This is the follower's problem in the bilevel optimization.
    
    Args:
        n_jobs: Number of jobs to schedule
        n_machines: Number of identical machines
        processing_times: List of processing times for each job
        time_limit: Optional time limit in seconds for Gurobi
        verbose: Whether to show Gurobi output
        
    Returns:
        dict with keys:
            - status: Gurobi solution status
            - makespan: Objective value (maximum machine load)
            - assignment: List of machine indices (one per job)
            - model: Gurobi model object
            
    Raises:
        RuntimeError: If gurobipy is not available
    """
    if gp is None:
        raise RuntimeError("gurobipy is not available. Make sure Gurobi is installed and the Python environment is correct.")

    model = gp.Model("parallel_machines_makespan")
    model.setParam('OutputFlag', 1 if verbose else 0)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)

    # Decision variables: x[j,m] = 1 if job j assigned to machine m
    x = {}
    for j in range(n_jobs):
        for m in range(n_machines):
            x[j, m] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{m}")

    # Load on each machine and makespan
    load = {m: model.addVar(lb=0.0, name=f"load_{m}") for m in range(n_machines)}
    Cmax = model.addVar(lb=0.0, name="Cmax")

    model.update()

    # Each job assigned exactly once
    for j in range(n_jobs):
        model.addConstr(gp.quicksum(x[j, m] for m in range(n_machines)) == 1, name=f"assign_{j}")

    # Define machine loads
    for m in range(n_machines):
        model.addConstr(load[m] == gp.quicksum(processing_times[j] * x[j, m] for j in range(n_jobs)), name=f"load_def_{m}")
        model.addConstr(Cmax >= load[m], name=f"cmax_ge_load_{m}")

    # Objective: minimize makespan
    model.setObjective(Cmax, GRB.MINIMIZE)

    model.optimize()

    status = model.Status
    if status == GRB.OPTIMAL or status == GRB.TIME_LIMIT or status == GRB.SUBOPTIMAL:
        # Extract assignment
        assignment = [-1] * n_jobs
        for j in range(n_jobs):
            for m in range(n_machines):
                val = x[j, m].X
                if val > 0.5:
                    assignment[j] = m
                    break
        makespan = Cmax.X
        return {"status": status, "makespan": makespan, "assignment": assignment, "model": model}
    else:
        return {"status": status, "message": "No feasible solution or model failed"}


def solve_scheduling_readable(n_jobs, n_machines, processing_times, time_limit=None, verbose=True):
    """Solve the parallel machine scheduling problem with machine-grouped output.
    
    Assigns n jobs to m identical machines to minimize the maximum load (makespan).
    Returns assignments grouped by machine for easier readability.
    
    Args:
        n_jobs: Number of jobs to schedule
        n_machines: Number of identical machines
        processing_times: List of processing times for each job
        time_limit: Optional time limit in seconds for Gurobi
        verbose: Whether to show Gurobi output
        
    Returns:
        dict with keys:
            - status: Gurobi solution status
            - makespan: Objective value (maximum machine load)
            - machine_assignments: Dict mapping machine index to list of job indices
            - machine_loads: Dict mapping machine index to total load
            - model: Gurobi model object
            
    Raises:
        RuntimeError: If gurobipy is not available
    """
    if gp is None:
        raise RuntimeError("gurobipy is not available. Make sure Gurobi is installed and the Python environment is correct.")

    model = gp.Model("parallel_machines_makespan")
    model.setParam('OutputFlag', 1 if verbose else 0)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)

    # Decision variables: x[j,m] = 1 if job j assigned to machine m
    x = {}
    for j in range(n_jobs):
        for m in range(n_machines):
            x[j, m] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{m}")

    # Load on each machine and makespan
    load = {m: model.addVar(lb=0.0, name=f"load_{m}") for m in range(n_machines)}
    Cmax = model.addVar(lb=0.0, name="Cmax")

    model.update()

    # Each job assigned exactly once
    for j in range(n_jobs):
        model.addConstr(gp.quicksum(x[j, m] for m in range(n_machines)) == 1, name=f"assign_{j}")

    # Define machine loads
    for m in range(n_machines):
        model.addConstr(load[m] == gp.quicksum(processing_times[j] * x[j, m] for j in range(n_jobs)), name=f"load_def_{m}")
        model.addConstr(Cmax >= load[m], name=f"cmax_ge_load_{m}")

    # Objective: minimize makespan
    model.setObjective(Cmax, GRB.MINIMIZE)

    model.optimize()

    status = model.Status
    if status == GRB.OPTIMAL or status == GRB.TIME_LIMIT or status == GRB.SUBOPTIMAL:
        # Extract assignments grouped by machine
        machine_assignments = {m: [] for m in range(n_machines)}
        machine_loads = {m: 0.0 for m in range(n_machines)}
        
        for j in range(n_jobs):
            for m in range(n_machines):
                val = x[j, m].X
                if val > 0.5:
                    machine_assignments[m].append(j)
                    machine_loads[m] += processing_times[j]
                    break
        
        makespan = Cmax.X
        return {
            "status": status, 
            "makespan": makespan, 
            "machine_assignments": machine_assignments,
            "machine_loads": machine_loads,
            "model": model
        }
    else:
        return {"status": status, "message": "No feasible solution or model failed"}


def solve_leader_knapsack(items, budget, time_limit=None, verbose=False):
    """Solve unbounded knapsack to maximize total processing time (heuristic).
    
    The leader's heuristic: select items (job types) to maximize the
    total processing time within the budget, allowing multiple copies.
    This provides a good initial incumbent for branch-and-bound.
    
    Args:
        items: List of items with .duration and .price attributes
        budget: Budget constraint
        time_limit: Optional time limit in seconds for Gurobi
        verbose: Whether to show Gurobi output
        
    Returns:
        dict with keys:
            - x: List of counts (number of copies of each item)
            - obj: Objective value (total processing time)
            - status: Gurobi solution status
            - model: Gurobi model object
            
    Raises:
        RuntimeError: If gurobipy is not available
    """
    if gp is None:
        raise RuntimeError("gurobipy is not available. Install gurobipy into the active Python environment.")

    n = len(items)
    model = gp.Model("leader_knapsack")
    model.setParam('OutputFlag', 1 if verbose else 0)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)

    # Decision variables: x[i] = number of copies of item i
    x = {i: model.addVar(vtype=GRB.INTEGER, lb=0.0, name=f"x_{i}") for i in range(n)}
    model.update()

    # Budget constraint
    model.addConstr(gp.quicksum(items[i].price * x[i] for i in range(n)) <= budget, name="budget")

    # Objective: maximize total processing time (heuristic for leader)
    model.setObjective(gp.quicksum(items[i].duration * x[i] for i in range(n)), GRB.MAXIMIZE)

    model.optimize()

    status = model.Status
    if status == GRB.OPTIMAL or status == GRB.TIME_LIMIT or status == GRB.SUBOPTIMAL:
        sol = [int(x[i].X) for i in range(n)]
        obj = model.ObjVal
        return {"x": sol, "obj": obj, "status": status, "model": model}
    else:
        return {"status": status, "message": "No feasible solution or model failed"}


if __name__ == "__main__":
    # Simple test of the solvers

    'add a CLI flag to determine which test to run'
    args = sys.argv[1:]
    
    #solve knapsack problem 1
    if len(args) >= 1 and args[0] in ("test-knapsack", "knapsack"):
        print("Testing knapsack solver...")
        items = [type("_", (), {'duration': 10, 'price': 7})(),
                 type("_", (), {'duration': 20, 'price': 12})(),
                 type("_", (), {'duration': 15, 'price': 8})(),
                 type("_", (), {'duration': 1,  'price': 1})()
                 ]
        budget = 30
        leader_result = solve_leader_knapsack(items, budget, verbose=True)
        print("Leader knapsack result:", leader_result)
        sys.exit(0)

    
    #solve scheduling problem 1
    if len(args) >= 1 and args[0] in ("test-scheduling", "scheduling"):
        print("Testing scheduling solver...")
        n_jobs = 5
        n_machines = 2
        processing_times = [4, 2, 5, 3, 6]
        sched_result = solve_scheduling(n_jobs, n_machines, processing_times, verbose=True)
        print("Scheduling result:", sched_result)
        sys.exit(0)

    #solve scheduling problem using readable output
    #to run, use python solvers.py test-scheduling-readable
    if len(args) >= 1 and args[0] in ("test-scheduling-readable", "scheduling-readable"):
        print("Testing scheduling solver with readable output...")
        n_jobs = 5
        n_machines = 2
        processing_times = [4, 2, 5, 3, 6]
        sched_result = solve_scheduling_readable(n_jobs, n_machines, processing_times, verbose=True)
        print("Scheduling result:", sched_result)
        sys.exit(0)


    
    # If no test specified, exit
    print("No test specified. Use 'test-knapsack' or 'test-scheduling' as argument.")  
    sys.exit(0)