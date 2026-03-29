# MibS Comparison Project Structure

## Directory Layout

```
mibs_comparison/
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
├── setup_guide.md                     # Installation instructions
│
├── formulation/
│   ├── __init__.py
│   ├── bilevel_model.py              # Abstract bilevel problem representation
│   └── mps_generator.py              # Convert instances to MPS format
│
├── solvers/
│   ├── __init__.py
│   ├── mibs_solver.py                # MibS wrapper
│   └── bnb_wrapper.py                # Your BnB solver wrapper
│
├── instances/
│   ├── load_from_grid.py             # Load instances from test_grid_sensitivity
│   ├── small_test_instances.json     # Hand-picked small instances
│   └── generated/                    # MPS files for MibS
│       ├── instance_2m_4j_rep1.mps
│       ├── instance_2m_4j_rep1.aux   # Auxiliary file for bilevel structure
│       └── ...
│
├── experiments/
│   ├── __init__.py
│   ├── run_comparison.py             # Main comparison script
│   ├── run_small_tests.py            # Test on small instances first
│   └── estimate_runtime.py           # Predict full run time
│
├── results/
│   ├── small_test_comparison.csv     # Results from small instances
│   ├── full_comparison.csv           # Results from all 180 instances
│   └── analysis/
│       ├── solver_comparison.ipynb   # Jupyter notebook for visualization
│       └── statistical_tests.py      # Significance testing
│
├── tests/
│   ├── test_mps_generation.py        # Unit tests
│   ├── test_mibs_interface.py
│   └── test_result_parsing.py
│
└── docs/
    ├── mibs_formulation.md           # Mathematical formulation details
    ├── installation_log.md           # Track your installation process
    └── troubleshooting.md            # Common issues and solutions
```

## Key Files and Their Purpose

### 1. **formulation/bilevel_model.py**

Abstract representation of your bilevel problem:

```python
class BilevelInstance:
    def __init__(self, n_jobs, n_machines, durations, prices, budget, seed):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.durations = durations
        self.prices = prices
        self.budget = budget
        self.seed = seed
    
    def to_dict(self):
        # Serialize for JSON storage
        pass
    
    @classmethod
    def from_grid_sensitivity(cls, csv_row):
        # Load from test_grid_sensitivity.py results
        pass
```

### 2. **formulation/mps_generator.py**

Converts instances to MPS format that MibS can read:

```python
def generate_mps_file(instance: BilevelInstance, output_path: str):
    """
    Generate MPS and AUX files for MibS.
    
    MPS file: Standard LP format with all variables and constraints
    AUX file: Specifies which variables/constraints belong to leader vs follower
    """
    pass
```

### 3. **solvers/mibs_solver.py**

Wrapper for MibS:

```python
class MibSSolver:
    def solve(self, instance: BilevelInstance, time_limit: float = 3600):
        """
        Solve using MibS and return standardized results.
        
        Returns:
            {
                'makespan': float,
                'leader_selection': list,
                'follower_assignment': dict,
                'runtime': float,
                'status': str,  # 'optimal', 'timeout', 'infeasible', etc.
                'nodes_explored': int,
                'gap': float
            }
        """
        pass
```

### 4. **solvers/bnb_wrapper.py**

Standardized interface for your existing solver:

```python
from gurobi_scheduling.bnb import run_bnb_classic
from gurobi_scheduling.models import MainProblem

class BnBSolver:
    def solve(self, instance: BilevelInstance, time_limit: float = 3600):
        """
        Solve using your BnB and return standardized results.
        
        Returns same format as MibSSolver for easy comparison.
        """
        pass
```

### 5. **experiments/run_small_tests.py**

Test on hand-picked small instances:

```python
# Select 5 representative instances:
# 1. Tiny (2m, 4j) - should be instant
# 2. Small easy (2m, 8j) - optimal case
# 3. Small hard (2m, 16j) - completed case
# 4. Medium (4m, 12j) - some search needed
# 5. Difficult (6m, 16j) - timeout case

# Run both solvers, compare results
# Estimate MibS runtime for larger instances
```

### 6. **experiments/estimate_runtime.py**

Predict how long full comparison will take:

```python
# Based on small test results, predict:
# - Average time per instance for MibS
# - Expected success rate (optimal vs timeout)
# - Total wall-clock time for 180 instances
```

### 7. **experiments/run_comparison.py**

Main comparison script:

```python
def run_full_comparison():
    # Load all 180 instances from cleaned CSV
    # For each instance:
    #   - Run BnB solver (you already have results)
    #   - Run MibS solver (new)
    #   - Record metrics
    # Save to comparison CSV
```

## Phase-by-Phase Implementation

### Phase 1: Setup (Day 1)
- [ ] Create project directory structure
- [ ] Install MibS (try conda first, then source if needed)
- [ ] Test MibS on example from their GitHub
- [ ] Document installation in `docs/installation_log.md`

### Phase 2: Small Instance Testing (Day 2)
- [ ] Implement `BilevelInstance` class
- [ ] Implement `mps_generator.py`
- [ ] Select 5 small instances from your cleaned CSV
- [ ] Generate MPS files manually
- [ ] Run MibS on these 5 instances
- [ ] Document results and learnings

### Phase 3: Automation (Day 3-4)
- [ ] Implement `MibSSolver` wrapper
- [ ] Implement `BnBSolver` wrapper
- [ ] Create `run_small_tests.py`
- [ ] Implement result parsing
- [ ] Create comparison CSV format

### Phase 4: Scaling Analysis (Day 5)
- [ ] Run on 20-30 diverse instances
- [ ] Analyze MibS performance patterns
- [ ] Estimate time for full 180 instances
- [ ] Decide: full run or subset?

### Phase 5: Full Comparison (Days 6+)
- [ ] Run full comparison if feasible
- [ ] Statistical analysis
- [ ] Visualization (plots, tables)
- [ ] Write up findings

## Data Flow

```
test_grid_sensitivity.py results (CSV)
    ↓
instances/load_from_grid.py
    ↓
BilevelInstance objects
    ↓
    ├→ formulation/mps_generator.py → MPS files → MibS → Results
    └→ solvers/bnb_wrapper.py → Your BnB → Results
         ↓
    experiments/run_comparison.py
         ↓
    results/full_comparison.csv
         ↓
    results/analysis/ (plots, statistics)
```

## Expected Output

### Comparison Metrics

For each instance, record:
- **Solver:** "BnB" or "MibS"
- **Instance ID:** (m_machines, n_jobs, budget_mult, rep)
- **Status:** optimal, timeout, node_limit, error
- **Makespan:** Objective value found
- **Runtime:** Seconds
- **Nodes:** Nodes explored
- **Gap:** Optimality gap (if not optimal)
- **Leader selection:** Job counts
- **Memory:** Peak memory usage

### Comparison Analysis

Generate:
1. **Performance profiles**: Time to optimality
2. **Win/loss/tie**: Which solver found better solutions
3. **Scalability**: How performance degrades with instance size
4. **Robustness**: Success rate on hard instances

## Critical Questions to Answer

1. **Is MibS faster than your BnB?** (Probably not, given your excellent bounds)
2. **Does MibS find better solutions?** (Should be same if both reach optimality)
3. **Where does each solver struggle?** (Different instance characteristics)
4. **Can you learn from MibS?** (Branching strategies, cutting planes)

## Resources Needed

- **Compute time**: Estimate 2-10 hours per instance for MibS on hard cases
- **Storage**: ~1MB per instance for MPS files
- **Python packages**: pandas, numpy, matplotlib, seaborn

---

**Next Steps:**
1. Read `MIBS_SETUP_GUIDE.md`
2. Install MibS
3. Test on example instance
4. Report back with installation status
