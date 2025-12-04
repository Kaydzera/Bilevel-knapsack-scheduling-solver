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
    The depth indicates up to which job type decisions have been made.
    Here, depth = 0 means no job types decided yet,
    depth = 1 means the first job type has been decided, etc.
    If depth = n_job_types, all job types have been decided and we flag the node as complete.
    The jobs after depth have not yet been considered and will be accounted for via bound approximations.
    
    Attributes:
        job_occurrences: List of counts for each job type (length = n_job_types)
        depth: Index of the jobs up to which decisions have been made
        remaining_budget: Budget remaining after selections so far
        n_job_types: Total number of job types in the problem, equal to len(job_occurrences)
        _machine_loads: Sorted list of loads on each machine (internal, length = m)
        _left_child_enabled: Whether to use exclusive mode (for left child support)
        _m: Number of machines
    """
    
    def __init__(self, job_occurrences, depth, remaining_budget, n_job_types, 
                 m=None, left_child_enabled=False, machine_loads=None):
        """Initialize a problem node.
        
        Args:
            job_occurrences: List of counts for each job type, or None to start with zeros
            depth: Current depth in the decision tree
            remaining_budget: Budget left after current selections
            n_job_types: Total number of job types
            m: Number of machines (required for machine load tracking)
            left_child_enabled: If True, use exclusive mode (items 0 to depth-1)
                               If False, use inclusive mode (items 0 to depth)
            machine_loads: Sorted list of machine loads (for internal use)
        """
        if job_occurrences is None:
            self.job_occurrences = [0] * n_job_types
        else:
            self.job_occurrences = list(job_occurrences)
        self.depth = int(depth)
        self.remaining_budget = int(remaining_budget)
        self.n_job_types = int(n_job_types)
        self._m = m
        self._left_child_enabled = left_child_enabled
        
        # Initialize machine loads
        if machine_loads is not None:
            self._machine_loads = list(machine_loads)
        elif m is not None:
            self._machine_loads = [0.0] * m
        else:
            self._machine_loads = None

    def get_already_committed_length(self, duration_at_depth=None):
        """Get the makespan contribution from committed jobs.
        
        In inclusive mode (left_child_enabled=False): includes items 0 to depth
        In exclusive mode (left_child_enabled=True): includes items 0 to depth-1,
            then distributes all occurrences of item at depth
        
        Args:
            duration_at_depth: Required in exclusive mode - duration of job at current depth
        
        Returns:
            Maximum load across all machines (makespan of committed jobs)
        """
        if self._machine_loads is None:
            return None
        
        if not self._left_child_enabled:
            # Inclusive mode: machine_loads already includes items 0 to depth
            return max(self._machine_loads) if self._machine_loads else 0.0
        else:
            # Exclusive mode: need to distribute item at depth
            if duration_at_depth is None:
                raise ValueError("duration_at_depth required in exclusive mode")
            # Make a copy to avoid modifying the stored loads
            loads = list(self._machine_loads)
            count_at_depth = self.job_occurrences[self.depth]
            
            # Distribute all occurrences onto machines
            for _ in range(count_at_depth):
                # Since loads is sorted, minimum is always first element
                loads[0] += duration_at_depth
                # Re-insert to maintain sorted order
                val = loads.pop(0)
                self._insert_sorted(loads, val)
            
            return max(loads) if loads else 0.0
    
    def _insert_sorted(self, loads, value):
        """Efficiently insert a value into a sorted list.
        
        Finds the position where value should go and inserts it,
        maintaining sorted order. O(m) time complexity.
        
        Args:
            loads: Sorted list of floats
            value: Value to insert
        """
        # Find insertion point (binary search would be O(log m) but list insertion is O(m) anyway)
        for i in range(len(loads)):
            if value <= loads[i]:
                loads.insert(i, value)
                return
        # If we get here, value is largest
        loads.append(value)
    
    @property
    def extendable(self):
        """Check if this node can be further expanded.
        
        A node is extendable if we haven't reached the last job type yet.
        Once we're at the last job type (depth = n_job_types), we can
        evaluate the solution directly without creating another level.
        
        Returns:
            True if depth < n_job_types  (more job types to decide)
        """
        return self.depth < self.n_job_types 

    def increment_current(self, price, duration=None):
        """Create a child node with one more copy of the current job type.
        
        This represents the 'right child' branch: add another copy
        of the current job type and stay at the same depth.
        
        Depth semantics: depth=N means we are DECIDING item N.
        Items 0..N-1 are already decided.
        
        In inclusive mode: Updates machine loads by adding duration to min machine.
        In exclusive mode: Doesn't update machine loads (items at depth not yet committed).
        
        Args:
            price: Cost of one copy of the current job type
            duration: Duration of the current job type (required if machine loads enabled)
            
        Returns:
            New ProblemNode with incremented count, or None if budget insufficient
        """
        if self.remaining_budget < price:
            return None
        new_occ = list(self.job_occurrences)
        new_occ[self.depth] += 1
        
        # Update machine loads if enabled and in inclusive mode
        new_loads = None
        if self._machine_loads is not None and not self._left_child_enabled:
            if duration is None:
                raise ValueError("duration required when machine loads are enabled")
            new_loads = list(self._machine_loads)
            # Since list is sorted, minimum is always the first element
            new_loads[0] += duration
            # Re-insert to maintain sorted order
            val = new_loads.pop(0)
            self._insert_sorted(new_loads, val)
        elif self._machine_loads is not None:
            # In exclusive mode, just copy the loads unchanged
            new_loads = list(self._machine_loads)
        
        return ProblemNode(new_occ, self.depth, self.remaining_budget - price, self.n_job_types,
                          m=self._m, left_child_enabled=self._left_child_enabled, machine_loads=new_loads)
    
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
        new_occ[self.depth-1] -= 1
        return ProblemNode(new_occ, self.depth, self.remaining_budget + price, self.n_job_types)

    def commit_current(self, duration=None, optional_price=None, optional_n_initial_occurences=None):
        """Create a child node that moves to the next job type.
        
        This is the 'lower child' branch: finalize the current job type
        and move to deciding the next one.
        
        In inclusive mode: Just copy machine loads (item at depth already included).
        In exclusive mode: Distribute all occurrences of item at depth onto machines.
        
        Args:
            duration: Duration of the current job type (required in exclusive mode)
            optional_price: If provided with optional_n_initial_occurences, 
                           pre-commits that many copies
            optional_n_initial_occurences: Number of copies to pre-commit
            
        Returns:
            New ProblemNode at depth+1, or None if not extendable
        """
        if not self.extendable:
            return None
        
        # Handle machine loads
        new_loads = None
        if self._machine_loads is not None:
            if self._left_child_enabled:
                # Exclusive mode: distribute all occurrences of item at depth
                if duration is None:
                    raise ValueError("duration required in exclusive mode")
                new_loads = list(self._machine_loads)
                count_at_depth = self.job_occurrences[self.depth]
                for _ in range(count_at_depth):
                    # Since list is sorted, minimum is always the first element
                    new_loads[0] += duration
                    # Re-insert to maintain sorted order
                    val = new_loads.pop(0)
                    self._insert_sorted(new_loads, val)
            else:
                # Inclusive mode: just copy
                new_loads = list(self._machine_loads)
        
        if optional_price is not None and optional_n_initial_occurences is not None:
            rem_budget = self.remaining_budget
            rem_budget -= optional_n_initial_occurences * optional_price
            if rem_budget < 0:
                print("Warning: committing initial occurrences exceeds remaining budget.")
                print("Returning a lower_child with no initial occurrences.")
                return ProblemNode(self.job_occurrences, self.depth + 1, self.remaining_budget, self.n_job_types,
                                  m=self._m, left_child_enabled=self._left_child_enabled, machine_loads=new_loads)
            job_occ = list(self.job_occurrences)
            job_occ[self.depth] += optional_n_initial_occurences
            return ProblemNode(job_occ, self.depth + 1, rem_budget, self.n_job_types,
                              m=self._m, left_child_enabled=self._left_child_enabled, machine_loads=new_loads)
        # If no optional parameters, just move to next depth
        return ProblemNode(self.job_occurrences, self.depth + 1, self.remaining_budget, self.n_job_types,
                          m=self._m, left_child_enabled=self._left_child_enabled, machine_loads=new_loads)

    def __repr__(self):
        return f"ProblemNode(depth={self.depth}, occ={self.job_occurrences}, rem_budget={self.remaining_budget}, extendable={self.extendable})"
