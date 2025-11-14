"""Data structures for the bilevel optimization problem.

This module contains the core data classes used throughout the solver:
- Item: Represents a job type with duration and price
- MainProblem: Container for the bilevel problem instance data
- ProblemNode: Represents a node in the branch-and-bound search tree
"""

from dataclasses import dataclass


@dataclass
class Item:
    """Represents a job type that the leader can select.
    
    Attributes:
        name: Identifier for the item
        duration: Processing time for this job type
        price: Cost to include one copy of this item
    """
    name: str
    duration: int
    price: int


class MainProblem:
    """Container for the bilevel (leader) problem data.
    
    The leader selects items (job types) within a budget constraint,
    and the follower schedules the selected jobs to minimize makespan.
    
    Attributes:
        prices: List of costs for each job type
        durations: List of processing times for each job type
        n_job_types: Number of distinct job types
        machines: Number of identical machines available to the follower
        budget_total: Total budget available to the leader
    """
    
    def __init__(self, prices, durations, anzahl_maschinen, budget_total):
        """Initialize a bilevel problem instance.
        
        Args:
            prices: List of costs for each job type
            durations: List of processing times for each job type
            anzahl_maschinen: Number of identical machines
            budget_total: Total budget constraint
        """
        assert len(prices) == len(durations)
        self.prices = list(prices)
        self.durations = list(durations)
        self.n_job_types = len(prices)
        self.machines = int(anzahl_maschinen)
        self.budget_total = int(budget_total)
        # Store items as objects for compatibility with bounds computation
        self.items = [type("_", (), {'duration': durations[i], 'price': prices[i]})() 
                      for i in range(self.n_job_types)]

    def __repr__(self):
        return f"MainProblem(n_types={self.n_job_types}, machines={self.machines}, budget={self.budget_total})"


class ProblemNode:
    """Node in the branch-and-bound search tree.
    
    Each node represents a partial or complete selection of job types.
    The depth indicates which job type is currently being decided.
    
    Attributes:
        job_occurrences: List of counts for each job type (length = n_job_types)
        depth: Index of current job type being decided (0..n_job_types)
        remaining_budget: Budget remaining after selections so far
        n_job_types: Total number of job types in the problem
    """
    
    def __init__(self, job_occurrences, depth, remaining_budget, n_job_types):
        """Initialize a problem node.
        
        Args:
            job_occurrences: List of counts for each job type, or None to start with zeros
            depth: Current depth in the decision tree
            remaining_budget: Budget left after current selections
            n_job_types: Total number of job types
        """
        if job_occurrences is None:
            self.job_occurrences = [0] * n_job_types
        else:
            self.job_occurrences = list(job_occurrences)
        self.depth = int(depth)
        self.remaining_budget = int(remaining_budget)
        self.n_job_types = int(n_job_types)

    @property
    def extendable(self):
        """Check if this node can be further expanded.
        
        Returns:
            True if depth < n_job_types (more decisions to make)
        """
        return self.depth < self.n_job_types

    def increment_current(self, price):
        """Create a child node with one more copy of the current job type.
        
        This represents the 'right child' branch: add another copy
        of the current job type and stay at the same depth.
        
        Args:
            price: Cost of one copy of the current job type
            
        Returns:
            New ProblemNode with incremented count, or None if budget insufficient
        """
        if self.remaining_budget < price:
            return None
        new_occ = list(self.job_occurrences)
        new_occ[self.depth] += 1
        return ProblemNode(new_occ, self.depth, self.remaining_budget - price, self.n_job_types)
    
    def decrement_current(self, price):
        """Create a child node with one fewer copy of the current job type.
        
        This represents a hypothetical 'left child' that decreases
        the count for the current job type.
        
        Args:
            price: Cost of one copy of the current job type
            
        Returns:
            New ProblemNode with decremented count, or None if count is already zero
        """
        if self.job_occurrences[self.depth] <= 0:
            print("Cannot decrement current job occurrence below zero.")
            return None
        new_occ = list(self.job_occurrences)
        new_occ[self.depth] -= 1
        return ProblemNode(new_occ, self.depth, self.remaining_budget + price, self.n_job_types)

    def commit_current(self, optional_price=None, optional_n_initial_occurences=None):
        """Create a child node that moves to the next job type.
        
        This is the 'lower child' branch: finalize the current job type
        and move to deciding the next one.
        
        Args:
            optional_price: If provided with optional_n_initial_occurences, 
                           pre-commits that many copies
            optional_n_initial_occurences: Number of copies to pre-commit
            
        Returns:
            New ProblemNode at depth+1, or None if not extendable
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
        # If no optional parameters, just move to next depth
        return ProblemNode(self.job_occurrences, self.depth + 1, self.remaining_budget, self.n_job_types)

    def __repr__(self):
        return f"ProblemNode(depth={self.depth}, occ={self.job_occurrences}, rem_budget={self.remaining_budget}, extendable={self.extendable})"
