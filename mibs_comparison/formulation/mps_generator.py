"""
MPS and AUX file generator for bilevel scheduling instances.

Generates files compatible with MibS (Mixed Integer Bilevel Solver).
"""

from typing import Tuple
import os
from .bilevel_model import BilevelInstance


def generate_mps_aux_files(instance: BilevelInstance, output_dir: str) -> Tuple[str, str]:
    """Generate MPS and AUX files for a bilevel instance.
    
    Args:
        instance: BilevelInstance to convert
        output_dir: Directory to write files to
        
    Returns:
        Tuple of (mps_path, aux_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    instance_id = instance.get_instance_id()
    mps_path = os.path.join(output_dir, f"{instance_id}.mps")
    aux_path = os.path.join(output_dir, f"{instance_id}.txt")
    
    # Generate MPS file
    mps_content = _generate_mps(instance)
    with open(mps_path, 'w', newline='\n') as f:
        f.write(mps_content)
    
    # Generate AUX file using legacy index format (working)
    aux_content = _generate_aux(instance)
    with open(aux_path, 'w', newline='\n') as f:
        f.write(aux_content)
    
    return mps_path, aux_path


def _generate_mps(instance: BilevelInstance) -> str:
    """Generate MPS file content.
    
    Uses single objective row format where objective coefficients are placed
    in the OBJ row, then AUX file specifies which belong to UL/LL.
    """
    n = instance.n_job_types
    m = instance.n_machines
    durations = instance.durations
    prices = instance.prices
    budget = instance.budget

    # Keep UL objective bounded for MibS by capping y_hat with a safe worst-case bound.
    min_price = min(prices)
    max_jobs = int(budget // min_price) if min_price > 0 else 0
    max_duration = max(durations)
    yhat_ub = max(1, max_jobs * max_duration)
    
    lines = []
    
    # NAME section
    lines.append(f"NAME          C{instance.get_instance_id().replace('_', '')[:7]}")
    
    # ROWS section
    lines.append("ROWS")
    lines.append(" N  OBJ")  # Single objective row
    lines.append(" L  BUDG")
    # Replace equality constraints with two inequalities for each assignment
    # Try using all L (≤) constraints like moore90
    for i in range(n):
        lines.append(f" L  AU{i+1}")
    for i in range(n):
        lines.append(f" L  AL{i+1}")
    for k in range(m):
        lines.append(f" L  MS{k+1}")    # -y_hat + Σᵢ dᵢ×yᵢₖ <= 0 (was ≥, flipped)
    
    # COLUMNS section
    lines.append("COLUMNS")
    lines.append("    INT1      'MARKER'                 'INTORG'")
    
    # x[i] variables (leader)
    # ALL variables must appear in OBJ row (like generalExample)
    # Reformulate: Σ_k y_ik - x_i <= 0  and  -Σ_k y_ik + x_i <= 0 (was x_i - Σ_k y_ik >= 0)
    for i in range(n):
        var_name = f"X{i+1}"
        lines.append(f"    {var_name:<10s} OBJ        0.0")  # Must appear in OBJ!
        lines.append(f"    {var_name:<10s} BUDG       {prices[i]}")
        lines.append(f"    {var_name:<10s} AU{i+1}      -1.0")
        lines.append(f"    {var_name:<10s} AL{i+1}      1.0")
    
    # y_hat variable (shared)
    # Without IC entries, put UL objective in MPS OBJ row
    # UL wants to maximize y_hat, so coefficient should be -1 (for minimization problem)
    lines.append(f"    YH         OBJ        -1.0")
    # y_hat appears in MAKESPAN constraints: y_hat >= Σᵢ dᵢ×yᵢₖ
    # Reformulated as L: -y_hat + Σᵢ dᵢ×yᵢₖ <= 0
    for k in range(m):
        lines.append(f"    YH         MS{k+1}      -1.0")
    
    # y[i,k] variables (follower)
    for i in range(n):
        for k in range(m):
            var_name = f"Y_{i+1}_{k+1}"
            print(f"DEBUG: Generating variable {var_name}")
            # Add to OBJ with coefficient 0 so MibS recognizes it as a variable
            lines.append(f"    {var_name:<10s} OBJ        0.0")
            lines.append(f"    {var_name:<10s} AU{i+1}      1.0")
            lines.append(f"    {var_name:<10s} AL{i+1}      -1.0")
            # Positive coefficient because: -y_hat + Σᵢ dᵢ×yᵢₖ <= 0
            lines.append(f"    {var_name:<10s} MS{k+1}      {durations[i]}")
    # NOTE: If you are still seeing old variable names, delete any __pycache__ folders and .pyc files in mibs_comparison/formulation to avoid stale bytecode.
    
    lines.append("    INT1END   'MARKER'                 'INTEND'")
    
    # RHS section
    lines.append("RHS")
    lines.append(f"    RHS        BUDG       {budget}")
    for i in range(n):
        lines.append(f"    RHS        AU{i+1}      0.0")
        lines.append(f"    RHS        AL{i+1}      0.0")
    for k in range(m):
        lines.append(f"    RHS        MS{k+1}      0.0")
    
    # BOUNDS section
    lines.append("BOUNDS")
    # All variables have lower bound 0 (integer non-negative)
    for i in range(n):
        lines.append(f" LI BND        X{i+1}        0")
    # Keep y_hat continuous (shared makespan variable).
    lines.append(f" LO BND        YH         0")
    lines.append(f" UP BND        YH         {yhat_ub}")
    for i in range(n):
        for k in range(m):
            lines.append(f" LI BND        Y_{i+1}_{k+1}      0")
    
    lines.append("ENDATA")
    
    return '\n'.join(lines) + '\n'


def _generate_aux(instance: BilevelInstance) -> str:
    """Generate AUX file using index-based format (like knapsack/linderoth).
    
    Key rules discovered from working examples:
    - N: number of upper-level variables
    - M: number of lower-level constraints
    - LC: column indices of lower-level controlled variables
    - LR: row indices of lower-level constraints (0-indexed)
    - LO: lower-level objective coefficients (ONE PER LC VARIABLE!)
    - OS: objective sense (1 = minimize)
    - IC: upper-level objective coefficients (ONE PER LC VARIABLE!)
    - IB: budget constraint RHS
    """
    n = instance.n_job_types
    m = instance.n_machines
    n_ll_vars = 1 + (n * m)  # y_hat + y_ij variables
    
    lines = []
    
    # N: Number of upper-level variables (x_1..x_n)
    # This should be the count of UL-only variables (not in LC list)
    lines.append(f"N {n}")
    
    # M: Number of lower-level constraints (ONLY LL constraints, NOT BUDGET!)
    # Like generalExample: only LL constraints in LR, not UL constraints
    n_ll_constraints = 2 * n + m  # ASSIGN + MAKESPAN (no +1 for BUDGET!)
    lines.append(f"M {n_ll_constraints}")
    
    # LC: Linking columns (lower-level controlled variables)
    # y_hat is column n, y_ij are columns n+1 onwards
    for col_idx in range(n, n + n_ll_vars):
        lines.append(f"LC {col_idx}")
    
    # LR: Lower-level constraint rows (0-indexed after objective row)
    # BUDGET is row 0 - it's a UL constraint, so SKIP IT!
    # Start from row 1 (ASSIGN_UP_1) like generalExample skips R0, R1
    for i in range(1, n_ll_constraints + 1):  # Start from 1, not 0!
        lines.append(f"LR {i}")
    
    # LO: Lower-level objective coefficients FOR EACH LC VARIABLE.
    # Use only y_hat in LL objective (robust pattern validated with MibS).
    lines.append("LO 1")
    for _ in range(n * m):
        lines.append("LO 0")
    
    # OS: Objective sense (1 = minimize)
    lines.append("OS 1")
    
    # IC: Upper-level objective coefficients FOR EACH LC VARIABLE
    # Try without IC to match moore90 format
    # lines.append("IC -1")
    # for _ in range(n * m):
    #     lines.append("IC 0")
    
    # IB: Budget constraint RHS
    # Try without IB to match moore90 format
    # lines.append(f"IB {int(instance.budget)}")
    
    return '\n'.join(lines) + '\n'


def _generate_aux_legacy_names(instance: BilevelInstance) -> str:
    """Generate AUX file using legacy format with NAMES (like moore90WithName).
    
    This uses N, M, LC, LR, LO, OS keywords but references variables/constraints
    by NAME instead of index. This format actually works in MibS!
    """
    n = instance.n_job_types
    m = instance.n_machines
    n_ll_vars = 1 + (n * m)  # y_hat + y_ij variables
    
    lines = []
    
    # N: Number of upper-level variables
    lines.append(f"N {n}")
    
    # M: Number of lower-level constraints
    n_ll_constraints = 1 + (2 * n) + m  # BUDGET + ASSIGN + MAKESPAN
    lines.append(f"M {n_ll_constraints}")
    
    # LC: Lower-level controlled variables (by NAME)
    lines.append("LC y_hat")
    for i in range(1, n + 1):
        for k in range(1, m + 1):
            lines.append(f"LC y_{i}_{k}")
    
    # LR: Lower-level constraint rows (by NAME)
    lines.append("LR BUDGET")
    for i in range(1, n + 1):
        lines.append(f"LR ASSIGN_UP_{i}")
        lines.append(f"LR ASSIGN_LO_{i}")
    for k in range(1, m + 1):
        lines.append(f"LR MAKESPAN_{k}")
    
    # LO: Lower-level objective coefficients (ONE PER LC VARIABLE)
    for _ in range(n_ll_vars):
        lines.append("LO 1")
    
    # OS: Objective sense (1 = minimize)
    lines.append("OS 1")
    
    return '\n'.join(lines) + '\n'


def _generate_aux_namebased(instance: BilevelInstance, mps_filename: str) -> str:
    """Generate AUX file using name-based format (RECOMMENDED by MibS docs).
    
    The name-based format is more robust than the legacy index-based format.
    It references variables and constraints by their names from the MPS file.
    
    Format:
        @NUMVARS - followed by count of LL variables
        @NUMCONSTR - followed by count of LL constraints
        @VARSBEGIN ... @VARSEND - list of LL variable names with coefficients
        @CONSTRSBEGIN ... @CONSTRSEND - list of LL constraint names
        @MPS - MPS filename
    """
    n = instance.n_job_types
    m = instance.n_machines
    
    lines = []
    
    # Number of lower-level variables: y_hat + all y_ij
    n_ll_vars = 1 + (n * m)
    lines.append("@NUMVARS")
    lines.append(str(n_ll_vars))
    
    # Number of lower-level constraints: BUDGET + ASSIGN_UP + ASSIGN_LO + MAKESPAN
    n_ll_constraints = 1 + (2 * n) + m
    lines.append("@NUMCONSTRS")
    lines.append(str(n_ll_constraints))
    
    # Lower-level variables section
    lines.append("@VARSBEGIN")
    
    # y_hat: coefficient 1.0 (LL minimizes y_hat)
    lines.append("y_hat 1.0")
    
    # y_ij: coefficient 1.0 (or 0.0 if not in LL objective)
    # For now, give them all coefficient 1.0
    for i in range(1, n + 1):
        for k in range(1, m + 1):
            lines.append(f"y_{i}_{k} 1.0")
    
    lines.append("@VARSEND")
    
    # Lower-level constraints section
    lines.append("@CONSTRBEGIN")
    
    # Include BUDGET constraint
    lines.append("BUDGET")
    
    # ASSIGN constraints (both UP and LO)
    for i in range(1, n + 1):
        lines.append(f"ASSIGN_UP_{i}")
        lines.append(f"ASSIGN_LO_{i}")
    
    # MAKESPAN constraints
    for k in range(1, m + 1):
        lines.append(f"MAKESPAN_{k}")
    
    lines.append("@CONSTREND")
    
    # MPS filename
    lines.append("@MPS")
    lines.append(mps_filename)
    
    return '\n'.join(lines) + '\n'


def generate_mps_aux_files_strict_template(instance: BilevelInstance, output_dir: str) -> Tuple[str, str]:
    """Generate MPS/AUX files in a strict hand-crafted style.

    This mirrors the line structure used by the known-good manual files that
    solved reliably in MibS.
    """
    os.makedirs(output_dir, exist_ok=True)

    instance_id = instance.get_instance_id()
    mps_path = os.path.join(output_dir, f"{instance_id}.mps")
    aux_path = os.path.join(output_dir, f"{instance_id}.txt")

    mps_content = _generate_mps_strict_template(instance)
    with open(mps_path, 'w', newline='\n') as f:
        f.write(mps_content)

    aux_content = _generate_aux_strict_template(instance)
    with open(aux_path, 'w', newline='\n') as f:
        f.write(aux_content)

    return mps_path, aux_path


def _generate_mps_strict_template(instance: BilevelInstance) -> str:
    """Generate MPS with strict formatting matching the known-good template."""
    n = instance.n_job_types
    m = instance.n_machines
    durations = instance.durations
    prices = instance.prices
    budget = int(instance.budget)

    min_price = min(prices)
    max_jobs = int(budget // min_price) if min_price > 0 else 0
    max_duration = max(durations)
    yhat_ub = max(1, max_jobs * max_duration)

    lines = []
    lines.append(f"NAME          C{instance.get_instance_id().replace('_', '')[:7]}")
    lines.append("ROWS")
    lines.append(" N  OBJ")
    lines.append(" L  BUDG")
    for i in range(n):
        lines.append(f" L  AU{i+1}")
    for i in range(n):
        lines.append(f" L  AL{i+1}")
    for k in range(m):
        lines.append(f" L  MS{k+1}")

    lines.append("COLUMNS")
    lines.append("    INT1      'MARKER'                 'INTORG'")

    for i in range(n):
        var = f"X{i+1}"
        lines.append(f"    {var:<9} OBJ       0")
        lines.append(f"    {var:<9} BUDG      {int(prices[i])}")
        lines.append(f"    {var:<9} AU{i+1}       -1")
        lines.append(f"    {var:<9} AL{i+1}       1")

    lines.append("    YH        OBJ       -1")
    for k in range(m):
        lines.append(f"    YH        MS{k+1}       -1")

    for i in range(n):
        for k in range(m):
            var = f"Y_{i+1}_{k+1}"
            lines.append(f"    {var:<9} OBJ       0")
            lines.append(f"    {var:<9} AU{i+1}       1")
            lines.append(f"    {var:<9} AL{i+1}       -1")
            lines.append(f"    {var:<9} MS{k+1}       {int(durations[i])}")

    lines.append("    INT1END   'MARKER'                 'INTEND'")
    lines.append("RHS")
    lines.append(f"    B         BUDG      {budget}")
    for i in range(n):
        lines.append(f"    B         AU{i+1}       0")
    for i in range(n):
        lines.append(f"    B         AL{i+1}       0")
    for k in range(m):
        lines.append(f"    B         MS{k+1}       0")

    lines.append("BOUNDS")
    for i in range(n):
        lines.append(f" LI BND       X{i+1:<9} 0")
    lines.append(" LO BND       YH        0")
    lines.append(f" UP BND       YH        {yhat_ub}")
    for i in range(n):
        for k in range(m):
            lines.append(f" LI BND       Y_{i+1}_{k+1:<7} 0")

    lines.append("ENDATA")
    return "\n".join(lines) + "\n"


def _generate_aux_strict_template(instance: BilevelInstance) -> str:
    """Generate AUX in strict index format matching known-good pattern."""
    n = instance.n_job_types
    m = instance.n_machines
    n_ll_vars = 1 + (n * m)
    first_ll_col = n
    last_ll_col = n + n_ll_vars - 1
    n_ll_constraints = (2 * n) + m

    lines = []
    lines.append(f"N {n_ll_vars}")
    lines.append(f"M {n_ll_constraints}")

    for col_idx in range(first_ll_col, last_ll_col + 1):
        lines.append(f"LC {col_idx}")

    for row_idx in range(1, n_ll_constraints + 1):
        lines.append(f"LR {row_idx}")

    lines.append("LO 1")
    for _ in range(n * m):
        lines.append("LO 0")
    lines.append("OS 1")

    return "\n".join(lines) + "\n"
