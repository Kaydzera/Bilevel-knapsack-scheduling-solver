from bilevel_gurobi import solve_bilevel_simpler

# Instance details from the cache key
items = [(3, 2), (5, 3), (7, 4), (9, 5), (11, 6), (13, 7), (15, 8), (17, 9)]
m = 8
budget = 40

# Convert to the expected item format (duration, price)
item_objs = [type('Item', (object,), {'duration': d, 'price': p})() for d, p in items]

# Run the solver
makespan, selection, schedule, nodes_evaluated, runtime = solve_bilevel_simpler(item_objs, m, budget, time_limit=3600.0, verbose=True)

print(f"Makespan: {makespan}")
print(f"Selection: {selection}")
print(f"Schedule: {schedule}")
print(f"Nodes evaluated: {nodes_evaluated}")
print(f"Runtime: {runtime}")

#"Complex_008_J8_M8_B40_8_40_[(3, 2), (5, 3), (7, 4), (9, 5), (11, 6), (13, 7), (15, 8), (17, 9)]"