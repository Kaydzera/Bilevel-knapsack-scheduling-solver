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
- **Nodes pruned**: Number of nodes pruned by bounds
- **Nodes evaluated**: Number of leaf nodes where scheduling was solved
- **Pruning rate**: Percentage of nodes pruned
- **Nodes per second**: Throughput metric

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

### Custom Logging

You can also log custom events:

```python
# Log informational messages
logger.info("Starting preprocessing...")

# Log debug messages (only in .log file)
logger.debug(f"Node details: {node_info}")

# Log warnings
logger.warning("Bound computation took longer than expected")

# Log errors
logger.error("Failed to solve subproblem")
```

## Extending the Logger

The logging system is designed to be easily extended. Here are common extensions:

### 1. Adding New Metrics

Edit `logger.py` and add fields to `self.metrics` in `__init__`:

```python
self.metrics = {
    # ... existing metrics ...
    "custom_metric": 0,
    "custom_list": [],
}
```

### 2. Logging New Events

Add methods to the `BnBLogger` class:

```python
def log_branching_rule(self, rule_name: str, decision_info: dict):
    """Log a branching rule decision."""
    if "branching_decisions" not in self.metrics:
        self.metrics["branching_decisions"] = []
    
    self.metrics["branching_decisions"].append({
        "rule": rule_name,
        "info": decision_info,
        "node_count": self.metrics["nodes_explored"]
    })
    self.logger.debug(f"Branching rule '{rule_name}': {decision_info}")
```

### 3. Tracking Different Bound Types

The system already supports different bound types. To add a new one:

```python
# In your bound computation code
bound_value = compute_dynamic_programming_bound(...)
logger.log_bound_computation(
    bound_value=bound_value,
    bound_type="dynamic_prog",  # New bound type
    node_depth=node.depth,
    computation_time=time_taken
)
```

### 4. Analyzing Branching Rules

Track which branching rules are used:

```python
self.metrics["branching_rule_usage"] = {}  # In __init__

# When branching:
logger.log_branching_decision("most_fractional", node_info)
```

## Example Extensions for Your Research

### Track Bound Quality Over Time

```python
def analyze_bound_tightness(self):
    """Compute bound tightness metrics."""
    bounds = self.metrics["bound_computations"]
    incumbent_updates = self.metrics["best_bound_updates"]
    
    # Compare bounds to actual incumbent at that time
    for bound_info in bounds:
        node_count = bound_info["node_count"]
        # Find incumbent at that node count
        # ... compute gap ...
```

### Compare Branching Strategies

```python
# Add to metrics
self.metrics["branching_strategy"] = strategy_name

# In analysis:
def compare_strategies(metrics_files):
    for file in metrics_files:
        metrics = load(file)
        print(f"{metrics['branching_strategy']}: "
              f"{metrics['nodes_explored']} nodes, "
              f"{metrics['total_runtime']:.2f}s")
```

### Track Node Processing Time

```python
def log_node_processing(self, node_id, processing_time):
    """Log time spent processing a single node."""
    if "node_times" not in self.metrics:
        self.metrics["node_times"] = []
    
    self.metrics["node_times"].append({
        "node": node_id,
        "time": processing_time,
        "depth": current_depth
    })
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

## Tips for Research

1. **Run experiments in batches**: Generate multiple instances and compare logs
2. **Track what matters**: Focus on metrics relevant to your research questions
3. **Visualize trends**: Use the JSON data to create plots (runtime vs. problem size, etc.)
4. **Version your experiments**: Include instance parameters in the log files
5. **Document changes**: When modifying the algorithm, note it in the logs

## Next Steps

- [ ] Implement different branching rules and log which one is used
- [ ] Add dynamic programming bound and compare with exact IP bound
- [ ] Generate hard instances and track performance differences
- [ ] Visualize bound progression over time
- [ ] Compare pruning efficiency across instance types
