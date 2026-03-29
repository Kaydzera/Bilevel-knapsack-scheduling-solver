# Bilevel Formulation for MibS

**Problem Type:** Bilevel Mixed-Integer Linear Program (MILP)  
**Solver:** MibS (Mixed Integer Bilevel Solver)  
**Application:** Adversarial Knapsack-Scheduling

---

## Mathematical Formulation

### Notation

**Indices:**
- `i ∈ {1,...,n}`: Job types
- `k ∈ {1,...,m}`: Machines

**Parameters:**
- `n`: Number of job types
- `m`: Number of machines
- `pᵢ`: Price of job type i
- `dᵢ`: Duration of job type i
- `B`: Budget (total budget available)

**Variables:**
- `xᵢ ∈ ℤ₊`: Number of jobs of type i purchased (upper level decision)
- `yᵢₖ ∈ ℤ₊`: Number of jobs of type i assigned to machine k (lower level decision)
- `y_hat ∈ ℝ₊`: Makespan (shared variable between levels)

---

## Complete Bilevel MILP

### Upper Level Problem (Leader - Maximize Makespan)

```
max  y_hat

s.t. Σᵢ₌₁ⁿ pᵢ × xᵢ ≤ B              [Budget constraint]
     
     xᵢ ∈ ℤ₊, ∀i ∈ {1,...,n}        [Integer job counts]
```

**Leader's goal:** Select job quantities to maximize the makespan (make follower's job harder)

---

### Lower Level Problem (Follower - Minimize Makespan)

```
min  y_hat

s.t. Σₖ₌₁ᵐ yᵢₖ = xᵢ                 [All jobs of type i must be assigned]
                                    ∀i ∈ {1,...,n}
     
     y_hat ≥ Σᵢ₌₁ⁿ dᵢ × yᵢₖ          [Makespan ≥ load on machine k]
                                    ∀k ∈ {1,...,m}
     
     yᵢₖ ∈ ℤ₊, ∀i ∈ {1,...,n}, ∀k ∈ {1,...,m}
```

**Follower's goal:** Assign jobs to machines to minimize makespan (load balancing)

---

## Key Properties

✅ **Linear Objectives:** Both levels have linear objectives (maximize/minimize `y_hat`)

✅ **Linear Constraints:** All constraints are linear

✅ **Shared Variable:** `y_hat` appears in both levels, linking them naturally

✅ **Integer Variables:** All decision variables are integer or continuous

✅ **MibS Compatible:** This formulation can be directly translated to MPS/AUX format

---

## How It Works

### 1. Leader's Strategy
- Leader chooses quantities `xᵢ` within budget `B`
- Leader wants to maximize `y_hat` (worse outcome for follower)
- Budget constraint: `Σᵢ pᵢ × xᵢ ≤ B`

### 2. Follower's Response
- Follower receives job counts `xᵢ` from leader
- Follower assigns jobs to machines: `Σₖ yᵢₖ = xᵢ` (must assign all jobs)
- Follower minimizes `y_hat` (best load balancing possible)

### 3. Makespan Definition
- Constraint: `y_hat ≥ loadₖ` for each machine k
- Where `loadₖ = Σᵢ dᵢ × yᵢₖ` (total duration on machine k)
- At optimum: `y_hat = max{load₁, load₂, ..., loadₘ}` (highest machine load)

### 4. Equilibrium
The solution is a Stackelberg equilibrium:
- Leader moves first (selects `xᵢ`)
- Follower moves second (selects `yᵢₖ`, minimizes `y_hat`)
- Leader anticipates follower's rational response

---

## Example Instance

**Parameters:**
- n = 4 job types
- m = 2 machines
- Prices: `p = [58, 19, 39, 24]`
- Durations: `d = [12, 33, 34, 9]`
- Budget: `B = 91`

**Upper Level Decision (example):**
- `x₁ = 1` (buy 1 job of type 1, cost 58)
- `x₂ = 1` (buy 1 job of type 2, cost 19)
- `x₃ = 0` (buy 0 jobs of type 3)
- `x₄ = 0` (buy 0 jobs of type 4)
- Total cost: 58 + 19 = 77 ≤ 91 ✓

**Lower Level Decision (follower's response):**
- `y₁₁ = 1, y₁₂ = 0` (assign type-1 job to machine 1)
- `y₂₁ = 0, y₂₂ = 1` (assign type-2 job to machine 2)
- Load on machine 1: `12 × 1 = 12`
- Load on machine 2: `33 × 1 = 33`
- Makespan: `y_hat = max(12, 33) = 33`

---

## Translation to MPS/AUX Format

### Variable Types

**Upper Level Variables (Leader):**
- `x[1], x[2], ..., x[n]`: Integer, ≥ 0
- `y_hat`: Continuous, ≥ 0 (shared with lower level)

**Lower Level Variables (Follower):**
- `y[1,1], y[1,2], ..., y[n,m]`: Integer, ≥ 0
- `y_hat`: Continuous, ≥ 0 (shared with upper level)

### Constraint Structure

**Upper Level Constraints:**
1. Budget: `Σᵢ pᵢ × xᵢ ≤ B`

**Lower Level Constraints:**
1. Assignment: `Σₖ yᵢₖ = xᵢ` for all i
2. Makespan: `y_hat ≥ Σᵢ dᵢ × yᵢₖ` for all k

**Linking Variables:**
- `xᵢ` appears in upper-level budget constraint
- `xᵢ` appears in lower-level assignment constraint (right-hand side)
- `y_hat` appears in both objectives

---

## MPS File Structure (Sketch)

```
NAME          BILEVEL_SCHEDULING
ROWS
 N  OBJUL         (Upper level objective)
 N  OBJLL         (Lower level objective)
 L  BUDGET        (Upper level budget constraint)
 E  ASSIGN_1      (Lower level: Σₖ y₁ₖ = x₁)
 ...
 E  ASSIGN_n      (Lower level: Σₖ yₙₖ = xₙ)
 G  MAKESPAN_1    (Lower level: y_hat ≥ load₁)
 ...
 G  MAKESPAN_m    (Lower level: y_hat ≥ loadₘ)
COLUMNS
    x[1]      OBJUL      0.0
    x[1]      BUDGET     p[1]
    x[1]      ASSIGN_1   -1.0
    ...
    y_hat     OBJUL      1.0      (maximize in upper level)
    y_hat     OBJLL      1.0      (minimize in lower level)
    ...
    y[1,1]    ASSIGN_1   1.0
    y[1,1]    MAKESPAN_1 d[1]
    ...
RHS
    RHS       BUDGET     B
    RHS       ASSIGN_1   0.0
    ...
BOUNDS
 LI BOUND     x[1]       0
 LI BOUND     x[2]       0
 ...
 LI BOUND     y[1,1]     0
 ...
 LO BOUND     y_hat      0
ENDATA
```

### AUX File (Bilevel Structure)

```
N 1                    (1 upper-level objective)
M 1                    (1 upper-level constraint: budget)
N 4                    (4 job types → 4 xᵢ variables)
LC 1                   (1 shared variable: y_hat)
LR 0                   (0 shared constraints)
LO 1                   (1 leader objective row: OBJUL)
LL 1                   (1 lower-level objective row: OBJLL)
LV 4                   (4 leader variables: x₁, x₂, x₃, x₄)
LV 1                   (1 shared variable: y_hat)
```

---

## Implementation Notes

### Variable Ordering
For MibS, order variables as:
1. Leader variables: `x[1], x[2], ..., x[n]`
2. Shared variables: `y_hat`
3. Follower variables: `y[1,1], y[1,2], ..., y[n,m]`

### Constraint Ordering
1. Upper-level constraints (budget)
2. Lower-level constraints (assignment, makespan)

### Objective Rows
- Upper objective: Maximize `y_hat` → coefficient +1.0
- Lower objective: Minimize `y_hat` → coefficient +1.0 (MibS handles min/max)

---

## Next Steps

1. **Implement MPS Generator** (`formulation/mps_generator.py`)
   - Input: `BilevelInstance` object
   - Output: `.mps` and `.aux` files
   
2. **Test on Small Instance**
   - Generate MPS for 2m_4j_rep1
   - Run MibS manually
   - Verify solution matches BnB result

3. **Automate**
   - Wrapper to call MibS
   - Parse output
   - Compare with BnB

---

## References

- **MibS Documentation:** https://github.com/coin-or/MibS/wiki
- **MPS Format:** http://lpsolve.sourceforge.net/5.5/mps-format.htm
- **AUX Format:** See MibS examples in GitHub repo
- **Bilevel Optimization:** Colson, Marcotte, Savard (2007) "An overview of bilevel optimization"
