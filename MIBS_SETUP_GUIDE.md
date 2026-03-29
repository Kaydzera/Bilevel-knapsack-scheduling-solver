# MibS Solver Setup and Comparison Guide

## Overview

This guide explains how to set up MibS (Mixed Integer Bilevel Solver) to solve your bilevel knapsack-scheduling problem and compare results with your custom branch-and-bound implementation.

## Problem Formulation

### Your Bilevel Problem

**Leader (Upper Level):**
- **Decision:** Select x_i ≥ 0 (integer) copies of job type i
- **Objective:** MAXIMIZE the makespan (follower's objective value)
- **Constraint:** Σ price_i × x_i ≤ budget

**Follower (Lower Level):**
- **Decision:** Assign jobs to machines (y_j,m ∈ {0,1})
- **Objective:** MINIMIZE makespan (maximum machine load)
- **Constraints:**
  - Each job assigned to exactly one machine
  - Makespan ≥ load on each machine

## MibS Installation

### Option 1: Using Conda (Recommended)

MibS is available through conda-forge:

```bash
# Create new environment for MibS
conda create -n mibs_env python=3.11
conda activate mibs_env

# Install MibS
conda install -c conda-forge mibssolver

# Install additional dependencies
conda install -c gurobi gurobi
conda install numpy pandas
```

### Option 2: Building from Source

If conda doesn't work, build from source:

```bash
# Prerequisites
# - C++ compiler (Visual Studio 2019+ on Windows)
# - CMake 3.15+
# - Git

# Clone repository
git clone https://github.com/coin-or/MibS.git
cd MibS

# Build dependencies (COIN-OR projects)
git clone https://github.com/coin-or-tools/coinbrew
cd coinbrew
./coinbrew build MibS --prefix=C:/MibS-install

# Python bindings
cd ../MibS/python
pip install -e .
```

### Option 3: Docker (Cross-platform)

```bash
# Pull COIN-OR container with MibS
docker pull coinor/coin-or-optimization-suite

# Run container with volume mount
docker run -it -v C:/Users/oleda:/workspace coinor/coin-or-optimization-suite
```

## Understanding MibS

### What MibS Does

MibS solves bilevel mixed-integer linear programs (BLPs) of the form:

```
max/min  c1^T x + d1^T y
s.t.     A1 x + B1 y ≤ b1
         x ≥ 0, x integer

         where y solves:
         max/min  c2^T x + d2^T y
         s.t.     A2 x + B2 y ≤ b2
                  y ≥ 0, y integer
```

Key features:
- Handles mixed-integer variables
- Supports conflicting objectives (leader MAX, follower MIN)
- Uses branch-and-cut algorithms
- Can exploit problem structure

### Input Format

MibS accepts:
1. **MPS format**: Standard LP/MIP file format with auxiliary file for bilevel structure
2. **AMPL format**: Algebraic modeling language
3. **Python API**: Direct problem construction (if available)

## Bilevel Formulation for MibS

Your problem has a **linear bilevel formulation** that works directly with MibS!

### Variables

**Upper Level (Leader):**
- `x_i ∈ ℤ₊`: Number of jobs of type i to purchase (i = 1,...,n)
- `y_hat ∈ ℝ₊`: Makespan (shared with lower level)

**Lower Level (Follower):**
- `y_ik ∈ ℤ₊`: Number of jobs of type i assigned to machine k (i = 1,...,n; k = 1,...,m)
- `y_hat ∈ ℝ₊`: Makespan (shared with upper level)

### Complete Bilevel MILP

**Upper Level (Leader - Maximize Makespan):**
```
max  y_hat
s.t. Σᵢ pᵢ × xᵢ ≤ B              [Budget constraint]
     xᵢ ∈ ℤ₊                      [Integer job counts]
```

**Lower Level (Follower - Minimize Makespan):**
```
min  y_hat
s.t. Σₖ yᵢₖ = xᵢ                 [Assign all jobs of type i] ∀i ∈ {1,...,n}
     y_hat ≥ Σᵢ dᵢ × yᵢₖ          [Makespan ≥ load on machine k] ∀k ∈ {1,...,m}
     yᵢₖ ∈ ℤ₊                     [Integer assignments]
```

### Key Properties

✅ **Linear objectives**: Both levels have linear objectives
✅ **Linear constraints**: All constraints are linear
✅ **Shared variable**: y_hat links the two levels naturally
✅ **MibS compatible**: Can be directly translated to MPS/AUX format

### Notation Summary

- `n`: Number of job types
- `m`: Number of machines
- `pᵢ`: Price of job type i
- `dᵢ`: Duration of job type i
- `B`: Budget
- `xᵢ`: Jobs of type i purchased by leader
- `yᵢₖ`: Jobs of type i assigned to machine k by follower
- `y_hat`: Makespan (minimized by follower, maximized by leader)

### How It Works

1. **Leader** selects job quantities `x_i` within budget, maximizing `y_hat`
2. **Follower** assigns those jobs to machines (satisfying `Σₖ yᵢₖ = xᵢ`), minimizing `y_hat`
3. **Makespan constraint** ensures `y_hat ≥ load_k` for all machines
4. At optimum, follower's minimization forces `y_hat = max_k(load_k)`

## Next Steps

### Phase 1: Installation & Testing (1-2 hours)
1. Install MibS using one of the methods above
2. Run MibS on standard test instances
3. Verify installation works

### Phase 2: Small Instance Testing (2-3 hours)
1. Select 3-5 small instances from your grid sensitivity
2. Manually create MPS files
3. Run MibS and compare with your BnB results

### Phase 3: Automated Pipeline (4-6 hours)
1. Create Python script to generate MPS files from your instances
2. Run MibS via command line or Python interface
3. Parse results and create comparison CSV

### Phase 4: Full Comparison (depends on MibS performance)
1. Run on all 180 instances from test_grid_sensitivity
2. Collect timing, optimality gaps, etc.
3. Statistical analysis and visualization

## Expected Challenges

1. **MibS may be slow**: Bilevel MIPs are extremely hard
2. **Formulation size**: Large instances may exceed memory
3. **Time limits**: May need hours per instance
4. **No solution guarantee**: May not find optimal within time limit

## Alternative: Simplified Comparison

If MibS is too slow for direct comparison, consider:
1. **Small instances only**: Test on 2-4 machines, 4-8 jobs
2. **Relaxations**: Compare LP relaxation bounds
3. **Single-level approximation**: Run leader heuristic + follower as baseline

## Resources

- **MibS Documentation**: https://github.com/coin-or/MibS/wiki
- **MPS Format Guide**: http://lpsolve.sourceforge.net/5.5/mps-format.htm
- **Bilevel Optimization Primer**: See MibS paper by Tahernejad et al. (2020)

## Contact & Support

- MibS GitHub Issues: https://github.com/coin-or/MibS/issues
- COIN-OR Forum: https://github.com/coin-or/COIN-OR-OptimizationSuite/discussions

---

**Next:** See `MIBS_PROJECT_STRUCTURE.md` for the project setup and code organization.
