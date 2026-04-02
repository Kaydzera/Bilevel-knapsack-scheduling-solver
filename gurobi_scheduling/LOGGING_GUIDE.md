# Logging System for Branch-and-Bound Solver

## Overview

This project includes a comprehensive logging system that tracks all important metrics during branch-and-bound execution. The system creates two types of files for each run:

1. **`.log` file**: Human-readable text log with timestamped messages
2. **`_metrics.json` file**: Structured JSON data with all performance metrics

## Quick Start

### Running with Logging

The logging system is automatically enabled when you run the BnB solver:

```bash
python main.py bilevel-step4-class
```

Log files are saved in the `logs/` directory with timestamps:
- `logs/sample_bilevel_20251124_104713.log`
- `logs/sample_bilevel_20251124_104713_metrics.json`

### Analyzing Logs

Use the analysis script to generate a summary:

```bash
python analyze_logs.py logs/sample_bilevel_20251124_104713_metrics.json
```

Compare multiple runs:

```bash
python analyze_logs.py logs/*.json
```

## What Gets Logged

### Performance Metrics
- **Total runtime**: Wall-clock time for the entire run
- **Nodes explored**: Total number of nodes visited in the search tree
- **Nodes pruned**: Number of nodes at which we pruned by bounds
- **Nodes evaluated**: Number of leaf nodes where scheduling was solved
- **Pruning rate**: Percentage of nodes at which we pruned


### Solution Quality
- **Incumbent updates**: Track when new best solutions are found
  - Makespan value
  - Job selection
  - Node count when found
  - Timestamp
- **Final solution**: Best makespan and selection found

### Bound Computation
- **Bound values**: Upper bounds computed at each node
- **Bound type**: Type of bound (e.g., "exact_ip", "fractional")
- **Computation time**: Time taken to compute each bound
- **Bound by depth**: Track bound quality at different tree depths

### Pruning Analysis
- **Pruning reasons**: Categorize why nodes were pruned
  - `bound_dominated`: Bound ≤ incumbent
  - `budget_infeasible`: Not enough budget for child node
  - etc.

### Problem Characteristics
- Number of job types
- Number of machines
- Budget constraint
- Job prices and durations

## Using the Logger in Your Code

### Basic Usage

```python
from logger import create_logger
from models import MainProblem
from bnb import run_bnb_classic

# Create logger
logger = create_logger(instance_name="my_instance")

# Create problem
problem = MainProblem(prices, durations, machines, budget)

# Run BnB with logging
result = run_bnb_classic(problem, max_nodes=10000, 
                        logger=logger, instance_name="my_instance")
```



## Log File Formats

### Text Log (.log)

```
2025-11-24 10:47:13 - INFO - Starting branch-and-bound solver
2025-11-24 10:47:13 - INFO - Problem: {'n_job_types': 8, ...}
2025-11-24 10:47:13 - DEBUG - Node 1: {'depth': 0, ...}
2025-11-24 10:47:13 - DEBUG - Bound computed: 70.00 (exact_ip) at depth 0 in 0.0030s
2025-11-24 10:47:13 - INFO - NEW INCUMBENT: 24.0
```

### Metrics JSON (_metrics.json)

```json
{
  "instance_name": "sample_bilevel",
  "timestamp": "20251124_104713",
  "total_runtime": 19.646,
  "nodes_explored": 36022,
  "nodes_pruned": 11141,
  "best_bound_updates": [
    {
      "incumbent": 24.0,
      "selection": [0, 0, 0, 3, 0, 0, 0, 0],
      "node_count": 0,
      "timestamp": 0.006
    }
  ],
  "bound_computations": [ ... ],
  "pruning_reasons": {
    "bound_dominated": 10500,
    "budget_infeasible": 641
  }
}
```
