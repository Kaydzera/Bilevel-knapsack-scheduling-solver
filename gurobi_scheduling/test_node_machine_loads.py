"""Test script for ProblemNode with machine load tracking."""

from models import ProblemNode

print("="*70)
print("TESTING ProblemNode with Machine Load Tracking")
print("="*70)

# Test scenario: 4 job types, 2 machines, budget=12
# durations = [4, 6, 8, 10], prices = [2, 3, 4, 5]
n_jobs = 4
m = 2
budget = 12
durations = [4, 6, 8, 10]
prices = [2, 3, 4, 5]

print("\nTest Setup:")
print(f"Jobs: {n_jobs}, Machines: {m}, Budget: {budget}")
print(f"Durations: {durations}")
print(f"Prices: {prices}")

# Test 1: Inclusive mode (left_child_enabled=False)
print("\n" + "="*70)
print("TEST 1: Inclusive Mode (items 0 to depth)")
print("="*70)

root = ProblemNode(None, depth=0, remaining_budget=budget, n_job_types=n_jobs, 
                   m=m, left_child_enabled=False)
print(f"\nRoot node: depth={root.depth}, occ={root.job_occurrences}")
print(f"Machine loads: {root._machine_loads}")
print(f"Already committed length: {root.get_already_committed_length()}")

# Add one job of type 0 (duration=4, price=2)
print(f"\n--- Adding job type 0 (duration={durations[0]}, price={prices[0]}) ---")
child1 = root.increment_current(prices[0], duration=durations[0])
print(f"Child node: depth={child1.depth}, occ={child1.job_occurrences}")
print(f"Machine loads: {child1._machine_loads}")
print(f"Already committed length: {child1.get_already_committed_length()}")
assert child1._machine_loads == [0.0, 4.0], f"Expected [0.0, 4.0], got {child1._machine_loads}"

# Add another job of type 0
print(f"\n--- Adding another job type 0 ---")
child2 = child1.increment_current(prices[0], duration=durations[0])
print(f"Child node: depth={child2.depth}, occ={child2.job_occurrences}")
print(f"Machine loads: {child2._machine_loads}")
print(f"Already committed length: {child2.get_already_committed_length()}")
assert child2._machine_loads == [4.0, 4.0], f"Expected [4.0, 4.0], got {child2._machine_loads}"

# Add one more job of type 0
print(f"\n--- Adding third job type 0 ---")
child3 = child2.increment_current(prices[0], duration=durations[0])
print(f"Child node: depth={child3.depth}, occ={child3.job_occurrences}")
print(f"Machine loads: {child3._machine_loads}")
print(f"Already committed length: {child3.get_already_committed_length()}")
assert child3._machine_loads == [4.0, 8.0], f"Expected [4.0, 8.0], got {child3._machine_loads}"
assert child3.get_already_committed_length() == 8.0, f"Expected 8.0, got {child3.get_already_committed_length()}"

# Commit and move to next depth
print(f"\n--- Committing depth 0, moving to depth 1 ---")
child4 = child3.commit_current()
print(f"Child node: depth={child4.depth}, occ={child4.job_occurrences}")
print(f"Machine loads: {child4._machine_loads}")
print(f"Already committed length: {child4.get_already_committed_length()}")
assert child4._machine_loads == [4.0, 8.0], f"Expected [4.0, 8.0], got {child4._machine_loads}"
assert child4.depth == 1

# Add job of type 1 (duration=6, price=3)
print(f"\n--- At depth 1, adding job type 1 (duration={durations[1]}, price={prices[1]}) ---")
child5 = child4.increment_current(prices[1], duration=durations[1])
print(f"Child node: depth={child5.depth}, occ={child5.job_occurrences}")
print(f"Machine loads: {child5._machine_loads}")
print(f"Already committed length: {child5.get_already_committed_length()}")
assert child5._machine_loads == [8.0, 10.0], f"Expected [8.0, 10.0], got {child5._machine_loads}"
assert child5.get_already_committed_length() == 10.0

print("\n✓ Inclusive mode tests passed!")


























# Test 2: Exclusive mode (left_child_enabled=True)
print("\n" + "="*70)
print("TEST 2: Exclusive Mode (items 0 to depth-1)")
print("="*70)

root2 = ProblemNode(None, depth=0, remaining_budget=budget, n_job_types=n_jobs,
                    m=m, left_child_enabled=True)
print(f"\nRoot node: depth={root2.depth}, occ={root2.job_occurrences}")
print(f"Machine loads: {root2._machine_loads}")

# Add jobs at depth 0 - these should NOT update machine loads yet
print(f"\n--- Adding 2 jobs of type 0 (should NOT update loads) ---")
ex_child1 = root2.increment_current(prices[0], duration=durations[0])
ex_child2 = ex_child1.increment_current(prices[0], duration=durations[0])
print(f"Child node: depth={ex_child2.depth}, occ={ex_child2.job_occurrences}")
print(f"Machine loads: {ex_child2._machine_loads}")
assert ex_child2._machine_loads == [0.0, 0.0], f"Expected [0.0, 0.0], got {ex_child2._machine_loads}"
print(f"Already committed length: {ex_child2.get_already_committed_length(duration_at_depth=durations[0])}")
print(f"job occurrences:", ex_child2.job_occurrences)
assert ex_child2.get_already_committed_length(duration_at_depth=durations[0]) == 4.0 
print(f"remaining budget: {ex_child2.remaining_budget}")

# Commit - this should distribute the 2 jobs of type 0
print(f"\n--- Committing depth 0 (should distribute 2 jobs of duration {durations[0]}) ---")
ex_child3 = ex_child2.commit_current(duration=durations[0])
print(f"Child node: depth={ex_child3.depth}, occ={ex_child3.job_occurrences}")
print(f"Machine loads: {ex_child3._machine_loads}")
assert ex_child3._machine_loads == [4.0, 4.0], f"Expected [4.0, 4.0], got {ex_child3._machine_loads}"
assert ex_child3.depth == 1

# Add 1 job at depth 1
#we have 12 - 2*2 = 8 remaining budget
print(f"\n--- At depth 1, adding 1 job of type 1 (remaining budget should be {ex_child3.remaining_budget}) ---")
ex_child4 = ex_child3.increment_current(prices[1], duration=durations[1])
assert ex_child4.remaining_budget == 5, f"Expected remaining budget 5, got {ex_child4.remaining_budget}"
print(f"Child node: depth={ex_child4.depth}, occ={ex_child4.job_occurrences}")
print(f"Machine loads: {ex_child4._machine_loads}")
assert ex_child4._machine_loads == [4.0, 4.0], f"Expected [4.0, 4.0], got {ex_child4._machine_loads}"
assert ex_child4.get_already_committed_length(duration_at_depth=durations[1]) == 10.0 # we added one job of duration 6 to previous 4, this calculation is done in get_already_committed_length
print(f"Already committed length: {ex_child4.get_already_committed_length(duration_at_depth=durations[1])}")

# Commit - distribute 1 job of type 1 (duration=6)
print(f"\n--- Committing depth 1 (should distribute 1 job of duration {durations[1]}) ---")
ex_child5 = ex_child4.commit_current(duration=durations[1])
print(f"Child node: depth={ex_child5.depth}, occ={ex_child5.job_occurrences}")
print(f"Machine loads: {ex_child5._machine_loads}")
# Should be: [4, 4] -> add 6 to min -> [4, 10]
assert ex_child5._machine_loads == [4.0, 10.0], f"Expected [4.0, 10.0], got {ex_child5._machine_loads}"
assert ex_child5.depth == 2

# Test get_already_committed_length in exclusive mode
print(f"\n--- Testing get_already_committed_length in exclusive mode ---")
print(f"At depth 1 with 1 job (before commit):")
print(f"  Stored loads: {ex_child4._machine_loads}")
print(f"  Computed length: {ex_child4.get_already_committed_length(duration_at_depth=durations[1])}")
assert ex_child4.get_already_committed_length(duration_at_depth=durations[1]) == 10.0, \
    f"Expected 10.0, got {ex_child4.get_already_committed_length(duration_at_depth=durations[1])}"

print("\n✓ Exclusive mode tests passed!")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
