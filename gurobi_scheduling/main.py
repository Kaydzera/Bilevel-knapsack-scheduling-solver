# Instructions to run this example in PowerShell:
'''
cd 'C:\Users\oleda\.vscode\Solving stuff with Gurobi\gurobi_scheduling'

# 1) activate the venv (from inside gurobi_scheduling, venv is in parent)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& '..\.venv311\Scripts\Activate.ps1'

# 2) verify interpreter and gurobipy
python --version
python -c "import gurobipy as gp; print('gurobipy', gp.__version__); print('gurobi', gp.gurobi.version())"

# 3) run the examples
python main.py
'''


#Now that we have successfully used gurobipy, we can start solving the real problem we wanted to solve:
#The objective is to build a solver for a bilevel optimization problem.
#In this problem, there are two levels of optimization: the upper-level (leader) and the lower-level (follower).
#The leader makes decisions first, and then the follower optimizes their own objective based on the leader's decisions.
#The leader's objective is to maximize the objective functioion of the follower while considering the follower's optimal response to the leader's decisions.
#In this way, the leader anticipates the follower's reactions and makes decisions that lead to the best possible outcome for itself, given the follower's optimal behavior.
#Here, the leader essentially solves a knapsack problem to determine which items to include in a knapsack to maximize the total value, while the follower solves a scheduling problem to minimize the makespan based on the items selected by the leader.
#That means that the leader only considers how their selection will impact the scheduling efficiency of the follower he is trying to maximize while the follower is trying to minimize it.
# 
#The problem consists of two main components:
#1) The leader's knapsack problem:
#The leader is given a budget constraint and a set of items, each with a cost and a length.
# The leader must select a subset of these items to include in a knapsack, subject to the budget constraint.
# It is important to note that the leader can choose multiple copies of the same item, as long as the total cost does not exceed the budget.
# The goal of the leader is to choose an subset of items (again, possibly with multiple copies) that maximizes the makespan of the scheduling problem of agent2 while respecting the cost capacity.
#
#2) The follower's scheduling problem:
#The follower is given a set of jobs, each with a processing time determined by the items selected by the leader.
# The follower must assign these jobs to a set of identical machines in a way that minimizes the makespan (the maximum load on any machine).

#How we will solve this bilevel optimization problem:
#We will model the leader by building a branch and bound algorithm that enmumerates all possible selections of items for the knapsack.
#This branch and bound algorithm will explore the decision tree of possible item selections, branching on whether to include or exclude each item.
#We will start this tree from the root node, which represents the state where no items have been selected yet. After that, we will look at job number 0 and build a node for each amount of copies of item 0 that can be selected without exceeding the budget.
#For each of these nodes, we will then look at job number 1 and build child nodes for each amount of copies of item 1 that can be selected without exceeding the budget, and so on.
#At each node in this tree, we will keep track of the current selection of items and the remaining budget. Also, we will compute an upper bound on the objective function of the follower's scheduling problem given the current selection of items.
#After that, we will use this upper bound to prune branches of the tree that cannot lead to a better solution than the best one found so far.
#When reaching a leaf node (where all items have been considered), we will solve the follower's scheduling problem using gurobipy to obtain the exact makespan for the current selection of items.
#We will then compare this makespan to the best one found so far and update the best solution if necessary.
#By systematically exploring the decision tree and using bounding techniques to prune suboptimal branches, we will efficiently find the optimal selection of items for the leader's knapsack problem that maximizes the makespan of the follower's scheduling problem while respecting the budget constraint.

#Also, we will need to define the data structures and input formats for the items, their costs, lengths, and the budget constraint for the leader's knapsack problem.
#Given a starting heuristic solution for the leader's knapsack problem, we can use it to initialize the best solution found so far in the branch and bound algorithm.
#This will help in pruning branches of the decision tree that cannot lead to a better solution than the initial heuristic solution.
#Also, this starting solution cn help us finding a smart way to navigate the decision tree more efficiently, focusing on promising branches first.
#This can be done by using techniques such as best-first search or depth-first search with backtracking, guided by the quality of the solutions found so far.
#By incorporating a starting heuristic solution, we can improve the efficiency of the branch and bound algorithm and increase the chances of finding the optimal solution for the leader's knapsack problem in a shorter amount of time.
#The starting heuristic solution can be obtained by solving the leader's knapsack problem that maximizes the total processing time of the jobs assigned to the follower, without considering the scheduling aspect.
#This can be done using gurobipy to solve a standard knapsack problem where the objective is to maximize the sum of the processing times of the selected items, subject to the budget constraint while allowing multiple copies of the same item to be selected.
#Once we have this initial solution, we can use it to initialize the best solution found so far in the branch and bound algorithm for the bilevel optimization problem. 

#Steps to implement:
#1) Define the data structures and input formats for the items, their costs, lengths, and the budget constraint for the leader's knapsack problem.
#2) Implement a function to solve the leader's knapsack problem using gurobipy, allowing multiple copies of the same item to be selected, to obtain a starting heuristic solution.
#3) Initialize the best solution found so far using the starting heuristic solution obtained in step 2
#4) Implement the branch and bound algorithm to explore the decision tree of possible item selections for the leader's knapsack problem.
#5) At each node in the decision tree, compute an upper bound on the objective function of the follower's scheduling problem given the current selection of items.
#To find an upper bound on the objective function of the follower's scheduling problem given the current selection of items, solve the following:
"""
    Compute an *exact integer programming bound* using gurobipy.

    The goal is to maximize:
        sum_i [ duration[i] * ceil( x[i] / m) ]
    subject to:
        sum_i [ price[i] * x[i] ] <= remaining_budget
        x[i] ∈ ℕ₀
    where:
    - i: index over items, starting from n_committed to total number of items. n_committed is the number of jobs already committed by the leader at the current node in the decision tree.
    - x[i]: number of copies of item i selected by the leader
    - duration[i]: processing time of item i
    - price[i]: cost of item i
    - remaining_budget: budget left for the leader
    - m: number of identical machines available to the follower
    This integer program maximizes the total processing time allocated to the follower's machines, considering that each machine can handle multiple jobs.
    The ceil( x[i] / m ) term accounts for the distribution of jobs across m machines, ensuring that we consider the worst-case load on any single machine.
    By solving this integer program, are able to obtain an upper bound on the makespan that the follower can achieve given the current selection of items by the leader.
    To compute the final bound, we still have to add already_commited_jobs_length = sum_i [ duration[i] * ceil( x[i] / m) ] over the already committed items (from 0 to n_committed-1).
    Later, we can solve this integer program using dynmic programming but first we will use gurobipy.
""" 
#6) Use the upper bound to prune branches of the tree that cannot lead to a better solution than the best one found so far.
#7) When reaching a leaf node, solve the follower's scheduling problem using gurobipy to obtain the exact makespan for the current selection of items.
#8) Compare the makespan to the best one found so far and update the best solution if necessary.
#9) Test the implementation with various instances of the bilevel optimization problem to ensure correctness and efficiency.

import json
import os
import sys
import math


try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    gp = None
    GRB = None


def solve_scheduling(n_jobs, n_machines, processing_times, time_limit=None, verbose=True):
    """Build and solve the assignment model.

    Inputs:
      - n_jobs: int
      - n_machines: int
      - processing_times: list of ints/float, length n_jobs
      - time_limit: optional float seconds for Gurobi
    Returns:
      - dict with keys: status, makespan, assignment (list of machine indices per job)
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

    # load on each machine
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
        # extract assignment
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


# Simple CLI and bilevel data helpers (step 1)
from dataclasses import dataclass


@dataclass
class Item:
    name: str
    duration: int
    price: int


def create_sample_bilevel():
    """Return a small sample bilevel instance (items, budget, machines).

    Items: list[Item]
    budget: int
    m: int (number of identical machines for follower)
    """
    durations = [2, 14, 4, 16, 6, 5, 3, 7]
    prices =    [1,  8, 2, 10, 3, 3, 1, 4]
    items = [Item(name=f"item{i}", duration=durations[i], price=prices[i]) for i in range(len(durations))]
    budget = 15
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
    print(f"Bilevel problem: {len(items)} items, budget={budget}, machines={m}")
    for it in items:
        print(f"  {it.name}: duration={it.duration}, price={it.price}")






def solve_leader_knapsack(items, budget, time_limit=None, verbose=False):
    """Solve an unbounded knapsack where x[i] are integer >=0 copies of item i.

    Objective: maximize sum(duration[i] * x[i]) s.t. sum(price[i] * x[i]) <= budget

    Returns dict with keys 'x' (list of counts), 'obj', 'status', 'model'
    """
    if gp is None:
        raise RuntimeError("gurobipy is not available. Install gurobipy into the active Python environment.")

    n = len(items)
    model = gp.Model("leader_knapsack")
    model.setParam('OutputFlag', 1 if verbose else 0)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)

    x = {i: model.addVar(vtype=GRB.INTEGER, lb=0.0, name=f"x_{i}") for i in range(n)}
    model.update()

    # budget constraint
    model.addConstr(gp.quicksum(items[i].price * x[i] for i in range(n)) <= budget, name="budget")

    # objective: maximize total processing time (heuristic)
    model.setObjective(gp.quicksum(items[i].duration * x[i] for i in range(n)), GRB.MAXIMIZE)

    model.optimize()

    status = model.Status
    if status == GRB.OPTIMAL or status == GRB.TIME_LIMIT or status == GRB.SUBOPTIMAL:
        sol = [int(x[i].X) for i in range(n)]
        obj = model.ObjVal
        return {"x": sol, "obj": obj, "status": status, "model": model}
    else:
        return {"status": status, "message": "No feasible solution or model failed"}
    



# --- Class-based BnB skeleton ---
from collections import deque


def greedy_fractional_total(items, budget):
    """Fast fractional knapsack: maximize total duration per price under budget.

    Items is an iterable of objects with attributes 'duration' and 'price'.
    Returns the fractional total duration achievable within budget (float).
    """
    # sort by duration/price ratio descending
    good = sorted(items, key=lambda it: (it.duration / it.price) if it.price > 0 else float('inf'), reverse=True)
    rem = budget
    total = 0.0
    for it in good:
        if rem <= 0:
            break
        if it.price <= rem:
            # take integer count greedily
            count = rem // it.price
            total += count * it.duration
            rem -= count * it.price
        else:
            # fractional take
            frac = rem / it.price
            total += frac * it.duration
            rem = 0
    return total


def initialize_bnb_state_from_heuristic(items, budget, m, verbose=False):
    """Run the leader heuristic (maximize total processing time) and evaluate follower makespan.

    items: list of objects with .duration and .price
    Returns dict with best_obj (makespan), best_selection (counts list), and other metadata.
    """
    res = solve_leader_knapsack(items, budget, verbose=verbose)
    if 'x' not in res:
        return { 'best_obj': 0.0, 'best_selection': [0]*len(items) }
    sel = res['x']
    proc = []
    for i, cnt in enumerate(sel):
        proc += [items[i].duration] * cnt
    if len(proc) == 0:
        makespan = 0.0
    else:
        sched = solve_scheduling(len(proc), m, proc, verbose=False)
        makespan = sched.get('makespan', 0.0)
    return { 'best_obj': makespan, 'best_selection': sel, 'proc_len': len(proc) }


class MainProblem:
    """Simple container for the bilevel (leader) problem data."""
    def __init__(self, prices, durations, anzahl_maschinen, budget_total):
        assert len(prices) == len(durations)
        self.prices = list(prices)
        self.durations = list(durations)
        self.n_job_types = len(prices)
        self.machines = int(anzahl_maschinen)
        self.budget_total = int(budget_total)

    def __repr__(self):
        return f"MainProblem(n_types={self.n_job_types}, machines={self.machines}, budget={self.budget_total})"


class ProblemNode:
    """Node in the branch-and-bound tree.

    - job_occurrences: list of counts for each job type (length = n_job_types)
    - depth: index of current job type being decided (0..n_job_types)
    - remaining_budget: budget left
    - extendable: whether we can still add/commit decisions
    """
    def __init__(self, job_occurrences, depth, remaining_budget, n_job_types):
        if job_occurrences is None:
            self.job_occurrences = [0] * n_job_types
        else:
            self.job_occurrences = list(job_occurrences)
        self.depth = int(depth)
        self.remaining_budget = int(remaining_budget)
        self.n_job_types = int(n_job_types)

    @property
    def extendable(self):
        return self.depth < self.n_job_types

    def increment_current(self, price):
        """Return a new node where we add one more copy of the current job type (if budget allows).

        This corresponds to the 'right child' in your pseudocode: increase count for current
        job type and keep depth the same (so we can add more copies).
        """

        if self.remaining_budget < price:
            return None
        new_occ = list(self.job_occurrences)
        new_occ[self.depth] += 1
        return ProblemNode(new_occ, self.depth, self.remaining_budget - price, self.n_job_types)
    
    def decrement_current(self, price):
        """Return a new node where we remove one copy of the current job type (if possible).

        This corresponds to a hypothetical 'left child' that decreases count for current
        job type and keeps depth the same.
        """
        if self.job_occurrences[self.depth] <= 0:
            print("Cannot decrement current job occurrence below zero.")
            return None
        new_occ = list(self.job_occurrences)
        new_occ[self.depth] -= 1
        return ProblemNode(new_occ, self.depth, self.remaining_budget + price, self.n_job_types)


    def commit_current(self, optional_price=None, optional_n_initial_occurences=None):
        """Return a new node committing the current job type and moving to the next depth.

        This is the 'lower child' in your pseudocode: move to the next job type to be decided.
        """
        if not self.extendable:
            return None
        if optional_price is not None and optional_n_initial_occurences is not None:
            rem_budget = self.remaining_budget
            rem_budget -= optional_n_initial_occurences * optional_price
            if rem_budget < 0:
                print("Warning: committing initial occurrences exceeds remaining budget.")
                print("Returning a lower_child with no initial occurrences.")
                return ProblemNode(self.job_occurrences, self.depth + 1, self.remaining_budget, self.n_job_types)
            job_occ = list(self.job_occurrences)
            job_occ[self.depth] += optional_n_initial_occurences
            return ProblemNode(job_occ, self.depth + 1, rem_budget, self.n_job_types)
        #If no optional parameters are provided, just move to next depth
        return ProblemNode(self.job_occurrences, self.depth + 1, self.remaining_budget, self.n_job_types)



    def __repr__(self):
        return f"ProblemNode(depth={self.depth}, occ={self.job_occurrences}, rem_budget={self.remaining_budget}, extendable={self.extendable})"





#review needed
def compute_ip_bound_exact(problem: MainProblem, node: ProblemNode, time_limit=2.0):
    """Exact integer-program bound using grouping to model ceil(x_i / m).

    Extracts necessary data from problem and node.
    remaining_items: list of objects with .duration and .price for types from node.depth..
    rem_budget: int budget left
    Returns the objective value (sum duration * ceil(x_i/m)) maximized under budget.
    """
    if gp is None:
        # fallback to 0 if gurobipy isn't available
        #throw an exception instead
        raise ImportError("Gurobi is not available. Please install gurobipy to use this function.")

    model = gp.Model("exact_ip_bound")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)

    #compute necessary data
    remaining_items = problem.items[node.depth:]
    rem_budget = node.remaining_budget
    m = problem.machines

    # For each item type i (indexed relative to node.depth), compute max copies allowed
    K = []
    for it in remaining_items:
        if it.price <= 0:
            K.append(0)
        else:
            K.append(rem_budget // it.price)

    # decision binaries u_{i,k} for each possible copy k (1..K_i)
    u = {}
    w = {}
    for i, Ki in enumerate(K):
        #i is index over remaining_items
        #Ki is max copies allowed for item i

        # G is the number of groups (ceil(Ki / m))
        # each group represents a batch of up to m copies of item i
        # We use integer arithmetic to compute ceil(Ki/m)
        # G = ceil(Ki / m) = (Ki + m - 1) // m and represents the number of groups for item i
        G = (Ki + m - 1) // m if Ki > 0 else 0
        for k in range(1, Ki + 1):
            # decision variable u[i,k] = 1 if copy k of item i is selected
            u[i, k] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{k}")
        for g in range(1, G + 1):
            # decision variable w[i,g] = 1 if at least one copy in group g of item i is selected
            w[i, g] = model.addVar(vtype=GRB.BINARY, name=f"w_{i}_{g}")

    model.update()

    # link group binaries w and copy binaries u
    for i, Ki in enumerate(K):
        # G is the number of groups (ceil(Ki / m))
        G = (Ki + m - 1) // m if Ki > 0 else 0
        for g in range(1, G + 1):
            # copies indices in this group
            first = (g - 1) * m + 1 # first copy index in group g for item i
            last = min(g * m, Ki)   # last copy index in group g for item i
            if first > last:
                # empty group skip
                model.addConstr(w[i, g] == 0)
                continue
            # if any u in group is 1 then w must be 1; and if sum u == 0 then w must be 0
            # this follows because w[i,g] represents whether at least one copy in group g is selected
            model.addConstr(gp.quicksum(u[i, k] for k in range(first, last + 1)) >= w[i, g], name=f"w_lower_{i}_{g}")
            for k in range(first, last + 1):
                model.addConstr(u[i, k] <= w[i, g], name=f"u_le_w_{i}_{g}_{k}")

    # budget constraint: sum price_i * sum_k u[i,k] <= rem_budget
    budget_expr = gp.quicksum(remaining_items[i].price * u[i, k] for (i, k) in u.keys())
    model.addConstr(budget_expr <= rem_budget, name="budget")

    # objective: maximize sum_i duration[i] * sum_g w[i,g]
    obj = gp.quicksum(remaining_items[i].duration * w[i, g] for (i, g) in w.keys())
    model.setObjective(obj, GRB.MAXIMIZE)

    model.optimize()

    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.SUBOPTIMAL:
        return model.ObjVal
    else:
        return 0.0


def run_bnb_classic(problem: MainProblem, max_nodes=100000, verbose=False):
    """Branch-and-bound using ProblemNode and compute_ip_bound (skeleton implementation).

    This implements the loop you pasted: frontier (DFS), seen set, pruning using the bound.
    """
    n = problem.n_job_types
    frontier = deque()
    seen = set()

    init_node = ProblemNode(None, depth=0, remaining_budget=problem.budget_total, n_job_types=n)
    frontier.append(init_node)
    seen.add((tuple(init_node.job_occurrences), init_node.depth, init_node.remaining_budget))

    # initial incumbent: run heuristic
    init_state = initialize_bnb_state_from_heuristic([type("_", (), {'duration':d,'price':p})() for d,p in zip(problem.durations, problem.prices)], problem.budget_total, problem.machines, verbose=False)
    incumbent = init_state['best_obj']
    incumbent_sel = init_state['best_selection']
    if verbose:
        print(f"Initial incumbent from heuristic: makespan={incumbent}, selection={incumbent_sel}")

    #Todo: if a promising aolution exists, the branch and bound should start from there and explore adjacent nodes first
    nodes = 0
    while frontier:
        node = frontier.pop()  # DFS
        nodes += 1
        if verbose:
            print(f"Visiting node: {node}")

        # If node is not extendable, evaluate exactly (solve scheduling)
        if not node.extendable:
            # build processing times from node.job_occurrences
            proc = []
            for i, cnt in enumerate(node.job_occurrences):
                proc += [problem.durations[i]] * cnt
            if len(proc) == 0:
                makespan = 0.0
            else:
                sched = solve_scheduling(len(proc), problem.machines, proc, verbose=False)
                #def solve_scheduling(n_jobs, n_machines, processing_times, time_limit=None, verbose=True):
                makespan = sched.get('makespan', None)
            if makespan is not None and makespan > incumbent:
                incumbent = makespan
                incumbent_sel = list(node.job_occurrences)
                if verbose:
                    print(f"New incumbent makespan={incumbent} selection={incumbent_sel}")
            continue


        # Compute bound using compute_ip_bound
        try:
            # Solve the exact IP bound computation
            Solution_node_IP = compute_ip_bound_exact(problem, node, time_limit=2.0)


            # We still have to add already committed jobs length
            already_committed_length = 0.0
            for i in range(0, node.depth):
                occ = node.job_occurrences[i]
                dur = problem.durations[i]
                already_committed_length += dur * math.ceil(occ / problem.machines)


            bound = Solution_node_IP.get('bound') + already_committed_length
            if verbose:
                print(f"Computed bound (optimistic makespan) = {bound}")
            # prune if bound <= incumbent
            if bound <= incumbent:
                if verbose:
                    print(f"Pruning node: bound {bound} <= incumbent {incumbent}")
                continue
        except Exception as e:
            if verbose:
                print(f"Bound computation failed: {e}")

        # expand node: right_child (increment current), lower_child (commit current)
        price = problem.prices[node.depth]
        right_child = node.increment_current(price)
        lower_child = node.commit_current()
        left_child = node.decrement_current(price)

        def try_add(child):
            key = (tuple(child.job_occurrences), child.depth, child.remaining_budget)
            if key in seen:
                return False
            seen.add(key)
            frontier.append(child)
            return True

        if right_child is not None:
            added = try_add(right_child)
            if verbose and added:
                print("Added right child")
        if left_child is not None:
            added = try_add(left_child)
            if verbose and added:
                print("Added left child")
        if lower_child is not None:
            added = try_add(lower_child)
            if verbose and added:
                print("Added lower child")



        if nodes >= max_nodes:
            if verbose:
                print(f"Node limit {max_nodes} reached, stopping")
            break

    return { 'best_obj': incumbent, 'best_selection': incumbent_sel, 'nodes_explored': nodes }











if __name__ == "__main__":
    # Minimal CLI: run scheduling example by default; use 'bilevel-step1' to print sample data
    args = sys.argv[1:]
    if len(args) >= 1 and args[0] in ("bilevel-step1", "step1"):
        items, budget, m = create_sample_bilevel()
        print_bilevel(items, budget, m)
        print("\nStep 1 complete: data structures and sample instance created. Next: implement leader knapsack solver (step 2).")
        sys.exit(0)
    if len(args) >= 1 and args[0] in ("bilevel-step2", "step2"):
        # Step 2: solve leader knapsack (unbounded) to maximize total processing time (heuristic)
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
            # build follower jobs from selection
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
    
    else:
        # Default behavior: run the simple scheduling demo
        base = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base, "sample_data.json")

        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                data = json.load(f)
            n_jobs = int(data.get("n_jobs"))
            n_machines = int(data.get("n_machines"))
            processing_times = list(data.get("processing_times"))
        else:
            # fallback example
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
            print("Solver did not produce a makespan; status:", result.get('status'), result.get('message'))



    # Provide a convenient CLI entry to run the new class-based BnB
    if len(args) >= 1 and args[0] in ("bilevel-step4-class", "step4-class"):
        items, budget, m = create_sample_bilevel()
        prices = [it.price for it in items]
        durations = [it.duration for it in items]
        problem = MainProblem(prices, durations, m, budget)
        print("Running class-based BnB skeleton on:", problem)
        res = run_bnb_classic(problem, max_nodes=10000, verbose=True)
        print("BnB result:", res)
        sys.exit(0)

    if len(args) >= 1 and args[0] == "test-exact-bound":
        # Build sample problem and root node, then test compute_ip_bound_exact directly
        items, budget, m = create_sample_bilevel()
        prices = [it.price for it in items]
        durations = [it.duration for it in items]
        problem = MainProblem(prices, durations, m, budget)
        root = ProblemNode(None, depth=0, remaining_budget=budget, n_job_types=problem.n_job_types)
        # build remaining_items list matching compute_ip_bound_exact expectation
        remaining_items = [type("_", (), {'duration': durations[i], 'price': prices[i]})() for i in range(root.depth, problem.n_job_types)]
        print("Testing exact IP bound on root node:")
        exact_val = compute_ip_bound_exact(problem, root, remaining_items, root.remaining_budget, m, time_limit=5.0)
        print(f"compute_ip_bound_exact -> total_processing_upper (duration*ceil(x/m)) = {exact_val}")
        sys.exit(0)