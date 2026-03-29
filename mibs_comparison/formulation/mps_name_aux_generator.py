"""
MPS + name-based AUX generator for MibS.

This module generates deterministic bilevel instances in the recommended
name-based AUX format documented by MibS.
"""

from __future__ import annotations

import os
from typing import Tuple

from .bilevel_model import BilevelInstance


def generate_mps_name_aux_files(instance: BilevelInstance, output_dir: str) -> Tuple[str, str]:
    """Generate MPS and name-based AUX files for one instance.

    Args:
        instance: Bilevel instance to export.
        output_dir: Directory where files are written.

    Returns:
        Tuple of (mps_path, aux_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    instance_id = instance.get_instance_id()
    mps_path = os.path.join(output_dir, f"{instance_id}.mps")
    aux_path = os.path.join(output_dir, f"{instance_id}.aux")

    mps_content = _generate_mps(instance)
    with open(mps_path, "w", encoding="utf-8") as mps_file:
        mps_file.write(mps_content)

    aux_content = _generate_name_aux(instance, os.path.basename(mps_path), instance_id)
    with open(aux_path, "w", encoding="utf-8") as aux_file:
        aux_file.write(aux_content)

    return mps_path, aux_path


def _generate_mps(instance: BilevelInstance) -> str:
    """Build MPS content.

    Formulation used:
      Upper level: minimize -y_hat
      s.t. budget, x integer >= 0

      Lower level: variables and constraints are identified in AUX
      Lower objective in AUX: minimize y_hat
    """
    n = instance.n_job_types
    m = instance.n_machines

    lines: list[str] = []

    lines.append(f"NAME          {instance.get_instance_id()}")

    lines.append("ROWS")
    lines.append(" N  OBJ")
    lines.append(" L  BUDGET")

    for i in range(n):
        lines.append(f" L  ASSIGN_UP_{i+1}")
        lines.append(f" G  ASSIGN_LO_{i+1}")

    for machine in range(m):
        lines.append(f" G  MAKESPAN_{machine+1}")

    lines.append("COLUMNS")
    lines.append("    MARK0000  'MARKER'                 'INTORG'")

    # Upper-level integer variables x_i
    for i, price in enumerate(instance.prices, start=1):
        var_name = f"x_{i}"
        lines.append(f"    {var_name:<10s} BUDGET     {price}")
        lines.append(f"    {var_name:<10s} ASSIGN_UP_{i}   -1")
        lines.append(f"    {var_name:<10s} ASSIGN_LO_{i}   -1")

    lines.append("    MARK0001  'MARKER'                 'INTEND'")

    # Shared variable y_hat (continuous, nonnegative)
    # Upper-level objective maximize y_hat => minimize -y_hat
    lines.append("    y_hat      OBJ        -1")
    for machine in range(1, m + 1):
        lines.append(f"    y_hat      MAKESPAN_{machine}   1")

    # Lower-level integer assignment variables y_i_k
    lines.append("    MARK0002  'MARKER'                 'INTORG'")
    for i, duration in enumerate(instance.durations, start=1):
        for machine in range(1, m + 1):
            var_name = f"y_{i}_{machine}"
            lines.append(f"    {var_name:<10s} ASSIGN_UP_{i}   1")
            lines.append(f"    {var_name:<10s} ASSIGN_LO_{i}   1")
            lines.append(f"    {var_name:<10s} MAKESPAN_{machine}   -{duration}")
    lines.append("    MARK0003  'MARKER'                 'INTEND'")

    lines.append("RHS")
    lines.append(f"    RHS        BUDGET     {instance.budget}")
    for i in range(1, n + 1):
        lines.append(f"    RHS        ASSIGN_UP_{i}   0")
        lines.append(f"    RHS        ASSIGN_LO_{i}   0")
    for machine in range(1, m + 1):
        lines.append(f"    RHS        MAKESPAN_{machine}   0")

    lines.append("BOUNDS")
    for i in range(1, n + 1):
        lines.append(f" LI BND        x_{i}       0")
    lines.append(" LO BND        y_hat      0")
    for i in range(1, n + 1):
        for machine in range(1, m + 1):
            lines.append(f" LI BND        y_{i}_{machine}     0")

    lines.append("ENDATA")
    return "\n".join(lines) + "\n"


def _generate_name_aux(instance: BilevelInstance, mps_filename: str, instance_name: str) -> str:
    """Build name-based AUX content using MibS recommended format."""
    n = instance.n_job_types
    m = instance.n_machines

    lower_variables: list[tuple[str, float]] = [("y_hat", 1.0)]
    for i in range(1, n + 1):
        for machine in range(1, m + 1):
            lower_variables.append((f"y_{i}_{machine}", 0.0))

    lower_constraints = [f"ASSIGN_UP_{i}" for i in range(1, n + 1)] + [
        f"ASSIGN_LO_{i}" for i in range(1, n + 1)
    ] + [
        f"MAKESPAN_{machine}" for machine in range(1, m + 1)
    ]

    lines: list[str] = []
    lines.append("@NUMVARS")
    lines.append(str(len(lower_variables)))

    lines.append("@NUMCONSTR")
    lines.append(str(len(lower_constraints)))

    lines.append("@VARSBEGIN")
    for var_name, coefficient in lower_variables:
        lines.append(f"{var_name} {coefficient:g}")
    lines.append("@VARSEND")

    lines.append("@CONSTRBEGIN")
    for row_name in lower_constraints:
        lines.append(row_name)
    lines.append("@CONSTREND")

    lines.append("@NAME")
    lines.append(instance_name)

    lines.append("@MPS")
    lines.append(mps_filename)

    return "\n".join(lines) + "\n"
