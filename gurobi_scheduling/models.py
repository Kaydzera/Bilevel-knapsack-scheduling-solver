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
        _branch_type: 'root' (can create left/right), 'right' (can create right/lower), 'left' (can create left/lower)
        _m: Number of machines
    """
    
    def __init__(self, job_occurrences, depth, remaining_budget, n_job_types, 
                 m=None, branch_type='root', machine_loads=None):
        """Initialize a problem node.
        
        Args:
            job_occurrences: List of counts for each job type, or None to start with zeros
            depth: Current depth in the decision tree
            remaining_budget: Budget left after current selections
            n_job_types: Total number of job types
            m: Number of machines (required for machine load tracking)
            branch_type: 'root' (can create left/right/lower), 'right' (can create right/lower), 
                        'left' (can create left/lower)
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
        self._branch_type = branch_type
        
        # Initialize machine loads
        if machine_loads is not None:
            self._machine_loads = list(machine_loads)
        elif m is not None:
            self._machine_loads = [0.0] * m
        else:
            self._machine_loads = None

    def get_already_committed_length(self, duration_at_depth=None):
        """Get the makespan contribution from committed jobs.
        
        For 'right' branch: machine_loads include items 0 to depth (inclusive)
        For 'root' or 'left' branch: machine_loads include items 0 to depth-1,
            need to distribute occurrences at depth for evaluation
        
        Args:
            duration_at_depth: Required for 'root'/'left' branches - duration of job at current depth
        
        Returns:
            Maximum load across all machines (makespan of committed jobs)
        """
        if self._machine_loads is None:
            return None
        
        if self._branch_type == 'right':
            # Right branch: machine_loads already includes items 0 to depth
            return max(self._machine_loads) if self._machine_loads else 0.0
        else:
            # Root or left branch: need to distribute item at depth
            if duration_at_depth is None:
                raise ValueError("duration_at_depth required for root/left branch nodes")
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
        
        If parent is 'root' or 'left': Distributes ALL parent.job_occurrences[depth] + 1 jobs
        If parent is 'right': Just adds +1 incrementally to min machine
        
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
        
        # Update machine loads
        new_loads = None
        if self._machine_loads is not None:
            if duration is None:
                raise ValueError("duration required when machine loads are enabled")
            new_loads = list(self._machine_loads)
            
            if self._branch_type == 'right':
                # Parent already has loads for jobs at depth, just add +1 incrementally
                new_loads[0] += duration
                val = new_loads.pop(0)
                self._insert_sorted(new_loads, val)
            else:
                # Parent is 'root' or 'left': distribute ALL occurrences (parent's + new one)
                total_jobs = new_occ[self.depth]
                for _ in range(total_jobs):
                    new_loads[0] += duration
                    val = new_loads.pop(0)
                    self._insert_sorted(new_loads, val)
        
        return ProblemNode(new_occ, self.depth, self.remaining_budget - price, self.n_job_types,
                          m=self._m, branch_type='right', machine_loads=new_loads)
    
    def decrement_current(self, price):
        """Create a child node with one fewer copy of the current job type.
        
        This represents the 'left child' branch: decrease the count
        for the current job type and stay at the same depth.
        
        Machine loads are copied unchanged (valid for jobs [0..depth-1]).
        Jobs at depth are not included in loads and will be distributed
        when evaluating get_already_committed_length().
        
        Args:
            price: Cost of one copy of the current job type
            
        Returns:
            New ProblemNode with decremented count, or None if count is already zero
        """
        if self.job_occurrences[self.depth] <= 0:
            return None
        
        new_occ = list(self.job_occurrences)
        new_occ[self.depth] -= 1
        
        # Copy machine loads unchanged (valid for jobs [0..depth-1])
        new_loads = None
        if self._machine_loads is not None:
            new_loads = list(self._machine_loads)
        
        return ProblemNode(new_occ, self.depth, self.remaining_budget + price, self.n_job_types,
                          m=self._m, branch_type='left', machine_loads=new_loads)

    def commit_current(self, duration=None, optional_price=None, optional_n_initial_occurences=None):
        """Create a child node that moves to the next job type.
        
        This is the 'lower child' branch: finalize the current job type
        and move to deciding the next one.
        
        Machine load handling depends on parent's branch type:
        - Right branch: Loads already include jobs at depth, just copy
        - Root/Left branch: Need to distribute jobs at depth before moving
        
        Lower child can create both left and right children at the new depth.
        
        Args:
            duration: Duration of the current job type (required for root/left branches)
            optional_price: If provided with optional_n_initial_occurences, 
                           pre-commits that many copies
            optional_n_initial_occurences: Number of copies to pre-commit
            
        Returns:
            New ProblemNode at depth+1, or None if not extendable
        """
        if not self.extendable:
            return None
        
        # Handle machine loads based on parent's branch type
        new_loads = None
        if self._machine_loads is not None:
            if self._branch_type == 'right':
                # Right branch: loads already include jobs at depth, just copy
                new_loads = list(self._machine_loads)
            else:
                # Root or left branch: need to distribute jobs at depth
                if duration is None:
                    raise ValueError("duration required for root/left branch nodes")
                new_loads = list(self._machine_loads)
                count_at_depth = self.job_occurrences[self.depth]
                # Distribute all occurrences at current depth
                for _ in range(count_at_depth):
                    new_loads[0] += duration
                    val = new_loads.pop(0)
                    self._insert_sorted(new_loads, val)
        
        if optional_price is not None and optional_n_initial_occurences is not None:
            rem_budget = self.remaining_budget
            rem_budget -= optional_n_initial_occurences * optional_price
            if rem_budget < 0:
                #This should not happen! Raise an exception
                raise ValueError("Insufficient budget for pre-committed occurrences")
            
            job_occ = list(self.job_occurrences)
            job_occ[self.depth] += optional_n_initial_occurences
            return ProblemNode(job_occ, self.depth + 1, rem_budget, self.n_job_types,
                              m=self._m, branch_type='root', machine_loads=new_loads)
        # If no optional parameters, just move to next depth
        return ProblemNode(self.job_occurrences, self.depth + 1, self.remaining_budget, self.n_job_types,
                          m=self._m, branch_type='root', machine_loads=new_loads)

    def can_create_children(self):
        """Determine which types of children this node can create.
        
        Returns:
            Dictionary with keys 'lower', 'right', 'left' and boolean values
        """
        can_lower = self.extendable
        can_right = False
        can_left = False
        
        if self._branch_type == 'root':
            # Root nodes can create both left and right children
            can_right = True
            can_left = self.job_occurrences[self.depth] > 0
        elif self._branch_type == 'right':
            # Right branch can only create right children
            can_right = True
        elif self._branch_type == 'left':
            # Left branch can only create left children
            can_left = self.job_occurrences[self.depth] > 0
        
        return {
            'lower': can_lower,
            'right': can_right,
            'left': can_left
        }

    def __repr__(self):
        return f"ProblemNode(depth={self.depth}, occ={self.job_occurrences}, rem_budget={self.remaining_budget}, branch={self._branch_type})"


# ============================================================================
# TESTS FOR LEFT/RIGHT CHILD BRANCHING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Left/Right Child Branching Implementation")
    print("=" * 70)
    
    # Test setup: 3 job types, 2 machines, budget 20
    # Prices: [2, 3, 4], Durations: [5, 6, 7]
    n_jobs = 3
    m = 2
    budget = 20
    prices = [2, 3, 4]
    durations = [5, 6, 7]
    
    print(f"\nTest Setup: {n_jobs} job types, {m} machines, budget {budget}")
    print(f"Prices: {prices}, Durations: {durations}")
    
    # Test 1: Root node creation and capabilities
    print("\n" + "-" * 70)
    print("TEST 1: Root Node Creation")
    print("-" * 70)
    root = ProblemNode([0, 0, 0], depth=0, remaining_budget=budget, n_job_types=n_jobs, m=m, branch_type='root')
    print(f"Root: {root}")
    print(f"Machine loads: {root._machine_loads}")
    print(f"Can create children: {root.can_create_children()}")
    assert root.can_create_children() == {'lower': True, 'right': True, 'left': False}, "Root with 0 jobs should not allow left child"
    print("PASS Root node created correctly")
    
    # Test 2: Create first right child from root
    print("\n" + "-" * 70)
    print("TEST 2: First Right Child from Root")
    print("-" * 70)
    right1 = root.increment_current(prices[0], durations[0])
    print(f"Right1: {right1}")
    print(f"Machine loads: {right1._machine_loads}")
    print(f"Expected: [5.0, 0.0] (1 job distributed)")
    assert right1._branch_type == 'right', "Should be right branch"
    assert right1.job_occurrences == [1, 0, 0], "Should have 1 job at depth 0"
    assert sorted(right1._machine_loads) == [0.0, 5.0], "Should have distributed 1 job"
    print(f"Can create children: {right1.can_create_children()}")
    assert right1.can_create_children() == {'lower': True, 'right': True, 'left': False}, "Right branch can only create right/lower"
    print("PASS First right child distributes parent's jobs + 1 correctly")
    
    # Test 3: Create second right child (incremental)
    print("\n" + "-" * 70)
    print("TEST 3: Second Right Child (Incremental)")
    print("-" * 70)
    right2 = right1.increment_current(prices[0], durations[0])
    print(f"Right2: {right2}")
    print(f"Machine loads: {right2._machine_loads}")
    print(f"Expected: [5.0, 5.0] (added 1 job incrementally)")
    assert right2.job_occurrences == [2, 0, 0], "Should have 2 jobs at depth 0"
    assert sorted(right2._machine_loads) == [5.0, 5.0], "Should have added 1 job incrementally"
    print("PASS Subsequent right child adds incrementally")
    
    # Test 4: Create lower child from root
    print("\n" + "-" * 70)
    print("TEST 4: Lower Child from Root")
    print("-" * 70)
    lower1 = root.commit_current(durations[0])
    print(f"Lower1: {lower1}")
    print(f"Machine loads: {lower1._machine_loads}")
    print(f"Expected: [0.0, 0.0] (0 jobs at depth 0, nothing to distribute)")
    assert lower1._branch_type == 'root', "Lower child should be root type"
    assert lower1.depth == 1, "Should be at depth 1"
    assert lower1._machine_loads == [0.0, 0.0], "Should have no load (0 jobs at depth 0)"
    print(f"Can create children: {lower1.can_create_children()}")
    assert lower1.can_create_children() == {'lower': True, 'right': True, 'left': False}, "Lower child can create both left/right"
    print("PASS Lower child distributes jobs at depth and resets to root type")
    
    # Test 5: Create first right child from lower node with existing jobs
    print("\n" + "-" * 70)
    print("TEST 5: First Right Child from Lower Node with Jobs")
    print("-" * 70)
    # Create root with 2 jobs at depth 0
    root_with_jobs = ProblemNode([2, 0, 0], depth=0, remaining_budget=budget-4, n_job_types=n_jobs, m=m, branch_type='root')
    print(f"Root with jobs: {root_with_jobs}")
    lower2 = root_with_jobs.commit_current(durations[0])
    print(f"Lower2 (jobs=[2,0,0], depth=1): {lower2}")
    print(f"Machine loads: {lower2._machine_loads}")
    print(f"Expected: [5.0, 5.0] (2 jobs at depth 0 distributed during commit)")
    assert sorted(lower2._machine_loads) == [5.0, 5.0], "Lower should have distributed jobs at depth 0"
    
    # Now create first right child from lower2 at depth=1
    right_from_lower = lower2.increment_current(prices[1], durations[1])
    print(f"Right from lower: {right_from_lower}")
    print(f"Machine loads: {right_from_lower._machine_loads}")
    print(f"Expected: [5.0, 11.0] (1 job at depth 1 distributed onto min machine)")
    assert right_from_lower.job_occurrences == [2, 1, 0], "Should have 1 job at depth 1"
    assert sorted(right_from_lower._machine_loads) == [5.0, 11.0], "Should distribute 1 job at new depth"
    print("PASS First right child from lower node distributes correctly")
    
    # Test 6: Create left child from root
    print("\n" + "-" * 70)
    print("TEST 6: Left Child from Root")
    print("-" * 70)
    # Start with root that has 3 jobs at depth 0
    root_for_left = ProblemNode([3, 0, 0], depth=0, remaining_budget=budget-6, n_job_types=n_jobs, m=m, branch_type='root')
    print(f"Root for left: {root_for_left}")
    left1 = root_for_left.decrement_current(prices[0])
    print(f"Left1: {left1}")
    print(f"Machine loads: {left1._machine_loads}")
    print(f"Expected: [0.0, 0.0] (just copied, no distribution)")
    assert left1._branch_type == 'left', "Should be left branch"
    assert left1.job_occurrences == [2, 0, 0], "Should have decremented to 2"
    assert left1._machine_loads == [0.0, 0.0], "Should just copy loads"
    print(f"Can create children: {left1.can_create_children()}")
    assert left1.can_create_children() == {'lower': True, 'right': False, 'left': True}, "Left branch can only create left/lower"
    print("PASS Left child copies loads and sets left branch type")
    
    # Test 7: Create second left child
    print("\n" + "-" * 70)
    print("TEST 7: Second Left Child")
    print("-" * 70)
    left2 = left1.decrement_current(prices[0])
    print(f"Left2: {left2}")
    assert left2.job_occurrences == [1, 0, 0], "Should have decremented to 1"
    print("PASS Can create multiple left children")
    
    # Test 8: Cannot decrement below 0
    print("\n" + "-" * 70)
    print("TEST 8: Cannot Decrement Below Zero")
    print("-" * 70)
    left3 = left2.decrement_current(prices[0])
    print(f"Left3: {left3}")
    left4 = left3.decrement_current(prices[0]) if left3 else None
    print(f"Left4 (should be None): {left4}")
    assert left4 is None, "Should not allow decrement below 0"
    print("PASS Correctly prevents decrement below 0")
    
    # Test 9: get_already_committed_length for different branch types
    print("\n" + "-" * 70)
    print("TEST 9: get_already_committed_length()")
    print("-" * 70)
    
    # Right branch: loads already include jobs at depth
    test_right = ProblemNode([2, 0, 0], depth=0, remaining_budget=10, n_job_types=n_jobs, m=m, 
                            branch_type='right', machine_loads=[5.0, 5.0])
    committed_right = test_right.get_already_committed_length(durations[0])
    print(f"Right branch committed length: {committed_right} (expected: 5.0)")
    assert committed_right == 5.0, "Right branch should return max of loads directly"
    
    # Root/Left branch: needs to distribute jobs at depth
    test_root = ProblemNode([2, 0, 0], depth=0, remaining_budget=10, n_job_types=n_jobs, m=m,
                           branch_type='root', machine_loads=[0.0, 0.0])
    committed_root = test_root.get_already_committed_length(durations[0])
    print(f"Root branch committed length: {committed_root} (expected: 5.0)")
    assert committed_root == 5.0, "Root branch should distribute jobs at depth"
    print("PASS get_already_committed_length works for all branch types")
    
    # Test 10: Complex tree scenario
    print("\n" + "-" * 70)
    print("TEST 10: Complex Tree Scenario")
    print("-" * 70)
    print("Building: Root -> Right -> Lower -> Right")
    root_complex = ProblemNode([0, 0, 0], depth=0, remaining_budget=20, n_job_types=n_jobs, m=m, branch_type='root')
    r1 = root_complex.increment_current(prices[0], durations[0])
    r2 = r1.increment_current(prices[0], durations[0])
    lower = r2.commit_current(durations[0])
    r3 = lower.increment_current(prices[1], durations[1])
    
    print(f"Root: {root_complex}")
    print(f"R1 (right): {r1}, loads: {r1._machine_loads}")
    print(f"R2 (right): {r2}, loads: {r2._machine_loads}")
    print(f"Lower: {lower}, loads: {lower._machine_loads}")
    print(f"R3 (right from lower): {r3}, loads: {r3._machine_loads}")
    
    assert r1._branch_type == 'right', "R1 should be right"
    assert r2._branch_type == 'right', "R2 should be right"
    assert lower._branch_type == 'root', "Lower should be root"
    assert r3._branch_type == 'right', "R3 should be right"
    assert r2.job_occurrences == [2, 0, 0], "R2 should have 2 jobs at depth 0"
    assert lower.job_occurrences == [2, 0, 0] and lower.depth == 1, "Lower should keep jobs and move to depth 1"
    assert r3.job_occurrences == [2, 1, 0], "R3 should have 1 job at depth 1"
    assert sorted(lower._machine_loads) == [5.0, 5.0], "Lower should have distributed 2 jobs"
    print("PASS Complex tree builds correctly")

    # Test setup: 3 job types, 2 machines, budget 20
    # Prices: [2, 3, 4], Durations: [5, 6, 7]
    
    # Test 11: Complex starting point with all node types and get_already_committed_length
    print("\n" + "-" * 70)
    print("TEST 11: Complex Tree with [1,2,1] and get_already_committed_length")
    print("-" * 70)
    print("Starting with job_occurrences=[1,2,1] at depth=0")
    
    # Start with a root node that has [1,2,1] jobs
    start = ProblemNode([1, 2, 1], depth=0, remaining_budget=20, n_job_types=n_jobs, m=m, branch_type='root')
    print(f"\n1. Start (root): {start}")
    print(f"   Branch type: {start._branch_type}")
    print(f"   Machine loads: {start._machine_loads}")
    committed = start.get_already_committed_length(durations[0])
    print(f"   get_already_committed_length(): {committed} (expected: 5.0, 1 job at depth 0)")
    assert committed == 5.0, "Should distribute 1 job at depth 0"
    
    # Create right child from start (increment at depth 0)
    right1 = start.increment_current(prices[0], durations[0])
    print(f"\n2. Right1 from start: {right1}")
    print(f"   Branch type: {right1._branch_type}")
    print(f"   Machine loads: {right1._machine_loads}")
    committed = right1.get_already_committed_length(durations[0])
    print(f"   get_already_committed_length(): {committed} (expected: 5.0)")
    assert sorted(right1._machine_loads) == [5.0, 5.0], "Should have distributed 2 jobs at depth 0"
    assert committed == 5.0, "Right branch: loads already include depth"
    
    # Create left child from start (decrement at depth 0)
    left1 = start.decrement_current(prices[0])
    print(f"\n3. Left1 from start: {left1}")
    print(f"   Branch type: {left1._branch_type}")
    print(f"   Machine loads: {left1._machine_loads}")
    committed = left1.get_already_committed_length(durations[0])
    print(f"   get_already_committed_length(): {committed} (expected: 0.0, 0 jobs at depth 0)")
    assert left1.job_occurrences == [0, 2, 1], "Should have decremented to 0"
    assert committed == 0.0, "Should have 0 jobs at depth 0"
    
    # Create lower child from start (move to depth 1)
    lower1 = start.commit_current(durations[0])
    print(f"\n4. Lower1 from start: {lower1}")
    print(f"   Branch type: {lower1._branch_type}")
    print(f"   Machine loads: {lower1._machine_loads}")
    committed = lower1.get_already_committed_length(durations[1])
    print(f"   get_already_committed_length(): {committed} (expected: 11.0)")
    assert sorted(lower1._machine_loads) == [0.0, 5.0], "Should have distributed 1 job from depth 0"
    # Distribute 2 jobs of duration 6 onto [0.0, 5.0] -> [6.0, 11.0]
    expected = 11.0
    assert committed == expected, "Should distribute 2 jobs at depth 1"
    
    # From right1, create another right child
    right2 = right1.increment_current(prices[0], durations[0])
    print(f"\n5. Right2 from right1: {right2}")
    print(f"   Branch type: {right2._branch_type}")
    print(f"   Machine loads: {right2._machine_loads}")
    committed = right2.get_already_committed_length(durations[0])
    print(f"   get_already_committed_length(): {committed} (expected: 10.0)")
    assert right2.job_occurrences == [3, 2, 1], "Should have 3 jobs at depth 0"
    assert sorted(right2._machine_loads) == [5.0, 10.0], "Should have added 1 job incrementally"
    assert committed == 10.0, "Right branch loads include depth"
    
    # From right1, create lower child (move to depth 1)
    lower2 = right1.commit_current(durations[0])
    print(f"\n6. Lower2 from right1: {lower2}")
    print(f"   Branch type: {lower2._branch_type}")
    print(f"   Machine loads: {lower2._machine_loads}")
    committed = lower2.get_already_committed_length(durations[1])
    print(f"   get_already_committed_length(): {committed} (expected: 11.0)")
    assert sorted(lower2._machine_loads) == [5.0, 5.0], "Should just copy loads from right branch"
    # At depth 1, need to distribute 2 jobs of duration 6 onto [5.0, 5.0]
    expected = 11.0  # 5.0 + 6 + 0 vs 5.0 + 6 = max(11.0, 11.0)
    assert committed == expected, "Should distribute 2 jobs at depth 1"
    
    # From left1, create another left child
    left2 = left1.decrement_current(prices[1])
    print(f"\n7. Left2 from left1: {left2}")
    if left2:
        print(f"   ERROR: Should not be able to decrement at depth 0 (already 0)")
        assert False, "Cannot decrement below 0"
    else:
        print(f"   Correctly returns None (cannot decrement below 0)")
    
    # From left1, create lower child (move to depth 1)
    lower3 = left1.commit_current(durations[0])
    print(f"\n8. Lower3 from left1: {lower3}")
    print(f"   Branch type: {lower3._branch_type}")
    print(f"   Machine loads: {lower3._machine_loads}")
    committed = lower3.get_already_committed_length(durations[1])
    expected = 6.0  # 2 jobs of duration 6 distributed evenly: [6, 6]
    print(f"   get_already_committed_length(): {committed} (expected: {expected})")
    assert lower3.job_occurrences == [0, 2, 1], "Should keep jobs and move to depth 1"
    assert lower3._machine_loads == [0.0, 0.0], "Should have 0 jobs at depth 0"
    assert committed == expected, "Should distribute 2 jobs at depth 1"
    
    # From lower1, create right child at depth 1
    right3 = lower1.increment_current(prices[1], durations[1])
    print(f"\n9. Right3 from lower1: {right3}")
    print(f"   Branch type: {right3._branch_type}")
    print(f"   Machine loads: {right3._machine_loads}")
    committed = right3.get_already_committed_length(durations[1])
    expected = 12.0  # Distributed 3 jobs of duration 6 on [0, 5]: [0,5]→[6,5]→[6,11]→[12,11]
    print(f"   get_already_committed_length(): {committed} (expected: {expected})")
    assert right3.job_occurrences == [1, 3, 1], "Should have 3 jobs at depth 1"
    assert sorted(right3._machine_loads) == [11.0, 12.0], "Should distribute 3 jobs at depth 1"
    assert committed == expected, "Right branch loads include depth"
    
    # From lower1, create left child at depth 1
    left3 = lower1.decrement_current(prices[1])
    print(f"\n10. Left3 from lower1: {left3}")
    print(f"   Branch type: {left3._branch_type}")
    print(f"   Machine loads: {left3._machine_loads}")
    committed = left3.get_already_committed_length(durations[1])
    expected = 6.0  # Distributes 1 job of duration 6 on [0, 5]: [0,5]→[6,5], max=6
    print(f"   get_already_committed_length(): {committed} (expected: {expected})")
    assert left3.job_occurrences == [1, 1, 1], "Should have 1 job at depth 1"
    assert sorted(left3._machine_loads) == [0.0, 5.0], "Should copy loads unchanged"
    assert committed == expected, "Should distribute 1 job at depth 1"
    
    # From lower1, create another lower child (move to depth 2)
    lower4 = lower1.commit_current(durations[1])
    print(f"\n11. Lower4 from lower1: {lower4}")
    print(f"   Branch type: {lower4._branch_type}")
    print(f"   Machine loads: {lower4._machine_loads}")
    committed = lower4.get_already_committed_length(durations[2])
    # Distributes 2 jobs at depth 1 on [0, 5]: [0,5]→[6,5]→[6,11], then 1 job at depth 2: [6,11]→[13,11]
    expected = 13.0
    print(f"   get_already_committed_length(): {committed} (expected: {expected})")
    assert lower4.depth == 2, "Should be at depth 2"
    assert sorted(lower4._machine_loads) == [6.0, 11.0], "Should distribute 2 jobs at depth 1"
    assert committed == expected, "Should distribute 1 job at depth 2"
    
    print("\nPASS Complex tree with [1,2,1] and all get_already_committed_length calls work correctly")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! PASS")
    print("=" * 70)


'''
----------------------------------------------------------------------
TEST 11: Complex Tree with [1,2,1] and get_already_committed_length
----------------------------------------------------------------------
Starting with job_occurrences=[1,2,1] at depth=0

1. Start (root): ProblemNode(depth=0, occ=[1, 2, 1], rem_budget=20, branch=root)
   Machine loads: [0.0, 0.0]
   get_already_committed_length(): 5.0 (expected: 5.0, 1 job at depth 0)

2. Right1 from start: ProblemNode(depth=0, occ=[2, 2, 1], rem_budget=18, branch=right)
   Machine loads: [5.0, 5.0]
   get_already_committed_length(): 5.0 (expected: 5.0)

3. Left1 from start: ProblemNode(depth=0, occ=[0, 2, 1], rem_budget=22, branch=left)
   Machine loads: [0.0, 0.0]
   get_already_committed_length(): 0.0 (expected: 0.0, 0 jobs at depth 0)

4. Lower1 from start: ProblemNode(depth=1, occ=[1, 2, 1], rem_budget=20, branch=root)
   Machine loads: [0.0, 5.0]
   get_already_committed_length(): 11.0 (expected: 11.0)

5. Right2 from right1: ProblemNode(depth=0, occ=[3, 2, 1], rem_budget=16, branch=right)
   Machine loads: [5.0, 10.0]
   get_already_committed_length(): 10.0 (expected: 10.0)

6. Lower2 from right1: ProblemNode(depth=1, occ=[2, 2, 1], rem_budget=18, branch=root)
   Machine loads: [5.0, 5.0]
   get_already_committed_length(): 11.0 (expected: 11.0)

7. Left2 from left1: None
   Correctly returns None (cannot decrement below 0)

8. Lower3 from left1: ProblemNode(depth=1, occ=[0, 2, 1], rem_budget=22, branch=root)
   Machine loads: [0.0, 0.0]
   get_already_committed_length(): 6.0 (expected: 6.0)

9. Right3 from lower1: ProblemNode(depth=1, occ=[1, 3, 1], rem_budget=17, branch=right)
   Machine loads: [11.0, 12.0]
   get_already_committed_length(): 12.0 (expected: 12.0)

10. Left3 from lower1: ProblemNode(depth=1, occ=[1, 1, 1], rem_budget=23, branch=left)
   Machine loads: [0.0, 5.0]
   get_already_committed_length(): 6.0 (expected: 6.0)

11. Lower4 from lower1: ProblemNode(depth=2, occ=[1, 2, 1], rem_budget=20, branch=root)
   Machine loads: [6.0, 11.0]
   get_already_committed_length(): 13.0 (expected: 13.0)

PASS Complex tree with [1,2,1] and all get_already_committed_length calls work correctly

======================================================================
ALL TESTS PASSED! PASS
======================================================================
PS C:\Users\oleda\.vscode\Solving stuff with Gurobi> cd "c:\Users\oleda\.vscode\Solving stuff with Gurobi" ; .\.venv311\Scripts\python.exe gurobi_scheduling\models.py 2>&1 | Select-Object -Last 80
TEST 9: get_already_committed_length()
----------------------------------------------------------------------
Right branch committed length: 5.0 (expected: 5.0)
Root branch committed length: 5.0 (expected: 5.0)
PASS get_already_committed_length works for all branch types

----------------------------------------------------------------------
TEST 10: Complex Tree Scenario
----------------------------------------------------------------------
Building: Root -> Right -> Lower -> Right
Root: ProblemNode(depth=0, occ=[0, 0, 0], rem_budget=20, branch=root)
R1 (right): ProblemNode(depth=0, occ=[1, 0, 0], rem_budget=18, branch=right), loads: [0.0, 5.0]
R2 (right): ProblemNode(depth=0, occ=[2, 0, 0], rem_budget=16, branch=right), loads: [5.0, 5.0]
Lower: ProblemNode(depth=1, occ=[2, 0, 0], rem_budget=16, branch=root), loads: [5.0, 5.0]
R3 (right from lower): ProblemNode(depth=1, occ=[2, 1, 0], rem_budget=13, branch=right), loads: [5.0, 11.0]
PASS Complex tree builds correctly

----------------------------------------------------------------------
TEST 11: Complex Tree with [1,2,1] and get_already_committed_length
----------------------------------------------------------------------
Starting with job_occurrences=[1,2,1] at depth=0

1. Start (root): ProblemNode(depth=0, occ=[1, 2, 1], rem_budget=20, branch=root)
   Branch type: root
   Machine loads: [0.0, 0.0]
   get_already_committed_length(): 5.0 (expected: 5.0, 1 job at depth 0)

2. Right1 from start: ProblemNode(depth=0, occ=[2, 2, 1], rem_budget=18, branch=right)
   Branch type: right
   Machine loads: [5.0, 5.0]
   get_already_committed_length(): 5.0 (expected: 5.0)

3. Left1 from start: ProblemNode(depth=0, occ=[0, 2, 1], rem_budget=22, branch=left)
   Branch type: left
   Machine loads: [0.0, 0.0]
   get_already_committed_length(): 0.0 (expected: 0.0, 0 jobs at depth 0)

4. Lower1 from start: ProblemNode(depth=1, occ=[1, 2, 1], rem_budget=20, branch=root)
   Branch type: root
   Machine loads: [0.0, 5.0]
   get_already_committed_length(): 11.0 (expected: 11.0)

5. Right2 from right1: ProblemNode(depth=0, occ=[3, 2, 1], rem_budget=16, branch=right)
   Branch type: right
   Machine loads: [5.0, 10.0]
   get_already_committed_length(): 10.0 (expected: 10.0)

6. Lower2 from right1: ProblemNode(depth=1, occ=[2, 2, 1], rem_budget=18, branch=root)
   Branch type: root
   Machine loads: [5.0, 5.0]
   get_already_committed_length(): 11.0 (expected: 11.0)

7. Left2 from left1: None
   Correctly returns None (cannot decrement below 0)

8. Lower3 from left1: ProblemNode(depth=1, occ=[0, 2, 1], rem_budget=22, branch=root)
   Branch type: root
   Machine loads: [0.0, 0.0]
   get_already_committed_length(): 6.0 (expected: 6.0)

9. Right3 from lower1: ProblemNode(depth=1, occ=[1, 3, 1], rem_budget=17, branch=right)
   Branch type: right
   Machine loads: [11.0, 12.0]
   get_already_committed_length(): 12.0 (expected: 12.0)

10. Left3 from lower1: ProblemNode(depth=1, occ=[1, 1, 1], rem_budget=23, branch=left)
   Branch type: left
   Machine loads: [0.0, 5.0]
   get_already_committed_length(): 6.0 (expected: 6.0)

11. Lower4 from lower1: ProblemNode(depth=2, occ=[1, 2, 1], rem_budget=20, branch=root)
   Branch type: root
   Machine loads: [6.0, 11.0]
   get_already_committed_length(): 13.0 (expected: 13.0)

PASS Complex tree with [1,2,1] and all get_already_committed_length calls work correctly

======================================================================
ALL TESTS PASSED! PASS
'''