print("Hello, world!")
# py -3.13 ....


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
python hello.py
python main.py
'''


from gurobipy import *
from gurobipy import Model, GRB, quicksum


print("Solving knapsack problem using Gurobi")
w = [4, 2, 5, 4, 5, 1, 3, 5]
v = [10, 5, 18, 12, 15, 1, 2, 8]
C = 15
N = len(w)

knapsack_model = Model("knapsack")

'Decision variables: x[i] = 1 if item i is included in the knapsack, 0 otherwise'
#addVars = ( *indices, ub = float('inf'), obj = 0.0, vtype = GRB.CONTINUOUS, name="" )


x = knapsack_model.addVars(N, vtype=GRB.BINARY, name="x")
obj_fn = sum(v[i] * x[i] for i in range(N))
knapsack_model.setObjective(obj_fn, GRB.MAXIMIZE)

knapsack_model.addConstr(sum(w[i] * x[i] for i in range(N)) <= C)     

knapsack_model.setParam('OutputFlag', False)
knapsack_model.optimize()
print("Selected items:")
print("Optimal value:", knapsack_model.ObjVal)
for v in knapsack_model.getVars():
    print(f"{v.varName}: {v.x}")

