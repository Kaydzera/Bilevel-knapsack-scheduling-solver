# Grid Sensitivity Analysis

## Overview

This test suite evaluates solver performance across a comprehensive grid of problem configurations, simultaneously varying:
- **Number of machines** (m_machines)
- **Number of job types** (n_jobs)  
- **Budget multiplier** (low/medium/high)

The key innovation is **proportional budget scaling**: when machines increase by X%, the budget also increases by X%, allowing us to observe how the solver scales with problem size under different resource constraints.

## Test Design

### Parameter Grid

| Parameter | Range | Values |
|-----------|-------|--------|
| **m_machines** | 2 to 12 (step 2) | 6 values: 2, 4, 6, 8, 10, 12 |
| **n_jobs** | 4 to 24 (step 4) | 6 values: 4, 8, 12, 16, 20, 24 |
| **budget_multiplier** | Low/Med/High | 3 values: 1.3, 2.5, 5.0 |
| **repetitions** | - | 5 per configuration |

**Total Tests**: 6 × 6 × 3 × 5 = **540 tests**

### Budget Calculation

Budget scales proportionally with the number of machines:

```
base_cost = sum(prices) / n_jobs     # Average price per job type
budget = base_cost × m_machines × budget_multiplier
```

**Example**: If we have 10 machines and increase to 11 machines (+10%), the budget automatically increases by 10%.

### Budget Multiplier Scenarios

- **Low (1.3)**: Tight budget - severely constrains job selection
- **Medium (2.5)**: Balanced scenario - reasonable budget for scheduling
- **High (5.0)**: Generous budget - less constrained by cost

## Research Questions

This test suite helps answer:

1. **Scalability**: How do ceiling vs MaxLPT bounds perform as problem size increases?

2. **Budget Impact**: Does budget tightness affect the relative performance of the two bounding strategies?

3. **Interaction Effects**: Are there interesting interactions between:
   - Machine count and job type count?
   - Budget level and problem size?
   - Specific size combinations where one bound excels?

4. **Practical Thresholds**: At what problem sizes do we hit:
   - Timeout limits (3600s)?
   - Node limits (500,000)?
   - Optimality gaps?

## Usage

### Basic Run

Start the full grid analysis:

```bash
python test_grid_sensitivity.py
```

### Custom Parameters

```bash
python test_grid_sensitivity.py \
  --machines-min 2 \
  --machines-max 10 \
  --machines-step 2 \
  --jobs-min 4 \
  --jobs-max 20 \
  --jobs-step 4 \
  --budget-low 1.5 \
  --budget-medium 2.5 \
  --budget-high 4.0 \
  --repetitions 3 \
  --time-limit 1800.0
```

### Resume Interrupted Run

If the analysis is interrupted, resume from a specific configuration:

```bash
python test_grid_sensitivity.py \
  --resume-timestamp 20260225_143022 \
  --resume-multiplier 2.5 \
  --resume-machines 7 \
  --resume-jobs 14 \
  --resume-rep 2
```

This will resume from:
- Budget multiplier 2.5
- 7 machines
- 14 job types  
- Repetition 3 (0-indexed rep=2)

## Output

### Directory Structure

```
results/sensitivity_grid/
└── grid_YYYYMMDD_HHMMSS/
    └── sensitivity_grid.csv
```

### CSV Columns

| Column | Description |
|--------|-------------|
| timestamp | When the test was run |
| m_machines | Number of machines |
| n_jobs | Number of job types |
| budget_multiplier | Budget multiplier (1.3, 2.5, or 5.0) |
| budget | Calculated budget value |
| repetition | Repetition number (1-5) |
| seed | Random seed for reproducibility |
| ceiling_status | Status of ceiling bound (optimal/timeout/node_limit) |
| ceiling_nodes | Nodes explored with ceiling bound |
| ceiling_time | Runtime with ceiling bound (seconds) |
| ceiling_initial | Initial makespan before BnB search (ceiling) |
| ceiling_final | Final makespan after BnB search (ceiling) |
| ceiling_improvement | Improvement (initial - final) with ceiling |
| maxlpt_status | Status of MaxLPT bound |
| maxlpt_nodes | Nodes explored with MaxLPT bound |
| maxlpt_time | Runtime with MaxLPT bound (seconds) |
| maxlpt_initial | Initial makespan before BnB search (MaxLPT) |
| maxlpt_final | Final makespan after BnB search (MaxLPT) |
| maxlpt_improvement | Improvement (initial - final) with MaxLPT |

## Expected Runtime

**Conservative estimate** (assuming many tests hit timeout):
- 540 tests × 2 bounds × 3600s timeout = ~1,100 hours worst case
- In practice, many tests will complete much faster
- Estimated realistic runtime: **30-120 hours** (1.3-5 days)

**For low budget only (180 tests)**:
- Estimated runtime: **10-40 hours** (0.4-1.7 days)

**Recommendations**:
- Run on a dedicated machine
- Use screen/tmux session to prevent disconnection
- Monitor progress via CSV file growth
- Consider parallelization for faster completion

## Analysis Ideas

After completion, analyze:

1. **Performance comparison**: Plot nodes/time for ceiling vs MaxLPT across the grid
2. **Scaling trends**: How does runtime grow with m_machines and n_jobs?
3. **Budget sensitivity**: Does one bound handle tight budgets better?
4. **Heatmaps**: Visualize success/timeout rates across the grid
5. **Statistical tests**: Formal comparison of bound effectiveness

## Notes

- Each test uses a deterministic seed based on (n_jobs, m_machines, budget_mult, rep)
- Job types are automatically sorted by price (descending) via MainProblem
- Both bounds use the same timeout (3600s) and node limit (500,000)
- Results are flushed to CSV after each test for resume capability
- **Initial makespan**: Both bound types start from the same Max-LPT heuristic solution
- **Improvement**: Measures how much the BnB search improves upon the initial solution (positive = better)
