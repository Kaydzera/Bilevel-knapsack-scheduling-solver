Gurobi Scheduling Example

This small project demonstrates using Gurobi (Python API) to solve a simple scheduling problem: assign n jobs to m identical machines to minimize the makespan (maximum machine load).

Files:
- `main.py` — the solver and example runner.
- `sample_data.json` — example job processing times and number of machines.
- `requirements.txt` — notes on Python and Gurobi.

Quick start (Windows PowerShell):

1. Ensure Gurobi is installed and the license is set up (you mentioned this is already done).
2. Run the example:

```powershell
python main.py
```

If `gurobipy` is not on your Python PATH, start the appropriate Python environment (for example the one Gurobi installed) or adjust `PYTHONPATH`.

What the model does
- Decision variables x[j,m] assign each job to one machine.
- `load[m]` is the total processing time assigned to machine `m`.
- `Cmax` is the makespan; objective is to minimize `Cmax`.

Notes
- Gurobi's Python package is usually installed as part of the Gurobi installation (not via pip). If `import gurobipy` fails, ensure the Python you run is the one Gurobi installed into or use the Gurobi-provided instructions to add `gurobipy` to your environment.
