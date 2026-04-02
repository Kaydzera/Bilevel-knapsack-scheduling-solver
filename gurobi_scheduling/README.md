# Bilevel Scheduling Optimizer

A bilevel optimization framework for scheduling jobs on identical parallel machines. The **leader** selects job types under a budget constraint to maximize the follower's makespan, while the **follower** schedules the selected jobs to minimize makespan. The project implements a custom branch-and-bound solver with two bounding strategies (ceiling bound and Max-LPT bound) and includes comprehensive sensitivity analysis tooling.

# Acknowledegement

- English Version: 
Parts of this code were generated, revised, and/or improved using AI-powered tools (GitHub Copilot (GPT-4.1), Anthropic Claude (Claude Opus 4.6, Claude Sonnet 4, Claude Sonnet 4.6)). 
AI was used for various tasks, including creating initial versions, revising algorithms, fixing bugs, and optimizing code. All generated or modified sections were carefully reviewed, understood, and, if necessary, adapted by me. The code reflects my understanding and responsibility.

- Deutsche Version: 
Teile dieses Codes wurden mithilfe von KI-gestützten Werkzeugen (GitHub Copilot (GPT-4.1), Anthropic Claude (Claude Opus 4.6, Claude Sonnet 4, Claude Sonnet 4.6)) generiert,überarbeitet und/oder verbessert. Die KI wurde für verschiedene Aufgaben eingesetzt, darunter das Erstellen
von Grundversionen, das Überarbeiten von Algorithmen, das Beheben von Fehlern und das Optimieren von Code.
Alle generierten oder modifizierten Abschnitte wurden von mir sorgfältig geprüft, nachvollzogen und ggf.
angepasst. Der Code entspricht meinem Verständnis und meiner Verantwortung.

## Problem Description

- **Leader (upper level):** Chooses how many copies of each job type to include, constrained by a total budget. Goal: maximize the follower's makespan.
- **Follower (lower level):** Given the selected jobs, assigns them to `m` identical machines to minimize the makespan (maximum machine load).


## Project Structure

### Core Modules

| File | Description |
|------|-------------|
| `models.py` | Data classes: `Item`, `MainProblem`, `ProblemNode` |
| `solvers.py` | Gurobi-based IP solvers for follower scheduling and leader knapsack heuristic |
| `bnb.py` | Branch-and-bound algorithm with configurable bound types (ceiling vs Max-LPT) |
| `maxlpt_bound.py` | Max-LPT upper bound using unbounded knapsack DP (3/4-approximation) |
| `knapsack_dp.py` | DP solvers: 0/1, unbounded, and bounded knapsack |
| `bilevel_gurobi.py` | Complete enumeration baseline solver (practical for small instances only) |
| `logger.py` | Structured logging with `.log` and `_metrics.json` output |

### Test Suites

| File | Scope | Description |
|------|-------|-------------|
| `test_small.py` | 5 instances, 4 job types, 2–4 machines | Sanity checks, verifies BnB vs enumeration |
| `test_middle.py` | 10+ instances, 6–7 job types, 3–6 machines | Medium complexity with varied characteristics |
| `test_big.py` | 140 instances, 6–12 job types, 3–8 machines | tests the Algorithm on Instances of seven different Patterns |
| `test_grid_sensitivity.py` | Up to 540 configs | **Main experiment**: 2D grid over machines × jobs × budget levels |
| `test_sensitivity_maxlpt.py` | Ceiling vs Max-LPT comparison | Head-to-head bound comparison across parameter sweeps |

### Analysis & Results

| Path | Description |
|------|-------------|
| `results/` | CSV output from test runs, organized by timestamp |
| `analyze_logs.py` | Analyze BnB log files for performance metrics |
| `GRID_SENSITIVITY_README.md` | Detailed documentation for the grid sensitivity experiment |
| `LOGGING_GUIDE.md` | Guide to the logging system |

## Setup

### Requirements

- **Python 3.8+**
- **Gurobi** with a valid license (installed via the official Gurobi installer)
- Optional: `pandas` (for analysis scripts)

Gurobi's Python package (`gurobipy`) is typically installed as part of the Gurobi installation, not via pip. If `import gurobipy` fails, ensure your Python environment matches the one Gurobi was installed into.

### Quick Start

```powershell
cd gurobi_scheduling

run:
cd "Solving stuff with Gurobi/.venv311/Scripts/python.exe" test_small.py
to solve python test_small.py      # Fast, ~5 instances
```

## Running the Key Tests

### 1. Small / Medium / Big Instance Tests

These validate correctness and measure performance across instance sizes:

```powershell
python test_middle.py     # Medium, ~10 instances
python test_big.py        # Large, 140 instances (uses enumeration cache)
```

### 2. Grid Sensitivity Analysis (Main Experiment)

The grid sensitivity test is the **most important experiment**. It evaluates both ceiling and Max-LPT bounds across a 2D grid of machines × job types at multiple budget levels.

For more information, we refer to the dedicated README
NOte: It is possible to resume an interrupted run:

```powershell
python test_grid_sensitivity.py --resume-timestamp 20260302_112308 --resume-machines 8 --resume-jobs 16 --resume-multiplier 2.5 --resume-rep 3
```

Results are saved to `results/sensitivity_grid/grid_TIMESTAMP/sensitivity_grid.csv` and flushed after each test so you can monitor progress. See [GRID_SENSITIVITY_README.md](GRID_SENSITIVITY_README.md) for full documentation.

> **Runtime note:** The full 540-test grid can take 1–5 days depending on hardware. Many tests hit the 3600s timeout at larger sizes. Run on a dedicated machine and consider starting with only the low budget multiplier (180 tests).

```

### 3. Ceiling vs Max-LPT Bound Comparison

Compare the two bounding strategies head-to-head:

```powershell
python test_sensitivity_maxlpt.py --parameter jobs --start 4 --step 2 --repetitions 5
python test_sensitivity_maxlpt.py --parameter machines --start 2 --step 1 --repetitions 5
python test_sensitivity_maxlpt.py --parameter budget --start 1.2 --step 0.2 --repetitions 5
```



## Architecture

```
main.py (CLI)
  │
  ├── bilevel_gurobi.py ── enumeration baseline (small instances)
  │
  └── bnb.py ── branch-and-bound solver
        ├── solvers.py ── Gurobi IP for follower scheduling
        ├── maxlpt_bound.py ── Max-LPT upper bound (unbounded knapsack DP)
        ├── knapsack_dp.py ── DP solvers (0/1, unbounded, bounded)
        └── logger.py ── structured performance logging
```
