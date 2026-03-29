"""Grid sensitivity runner for MibS using strict MPS/AUX export.

Adapted from gurobi_scheduling/test_grid_sensitivity.py, but this version:
- Generates bilevel instances directly in formulation format.
- Exports strict-template MPS/AUX files for MibS.
- Runs MibS via WSL with per-instance timeout.
- Writes a CSV with key generation + solver metrics.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

try:
    from .bilevel_model import BilevelInstance
    from .mps_generator import generate_mps_aux_files_strict_template
except ImportError:
    from bilevel_model import BilevelInstance
    from mps_generator import generate_mps_aux_files_strict_template


def generate_random_instance(
    n_jobs: int,
    m_machines: int,
    budget_multiplier: float,
    seed: int,
) -> BilevelInstance:
    """Generate one random bilevel instance with budget scaling by machines."""
    random.seed(seed)

    prices = [random.randint(10, 100) for _ in range(n_jobs)]
    durations = [random.randint(5, 50) for _ in range(n_jobs)]

    base_cost = sum(prices) / n_jobs
    budget = base_cost * m_machines * budget_multiplier

    metadata = {
        "source": "grid_sensitivity_mibs",
        "budget_multiplier": budget_multiplier,
    }

    return BilevelInstance(
        n_job_types=n_jobs,
        n_machines=m_machines,
        durations=durations,
        prices=prices,
        budget=budget,
        seed=seed,
        metadata=metadata,
    )


def windows_to_wsl_path(path: Path) -> str:
    """Convert a Windows path to a /mnt/<drive>/... WSL path."""
    p = path.resolve()
    drive = p.drive.rstrip(":").lower()
    tail = p.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def parse_mibs_output(text: str, timeout_hit: bool) -> Dict[str, object]:
    """Extract key metrics from MibS output text."""
    lower = text.lower()

    status = "failed"
    if any(token in lower for token in ["malloc", "aborted", "core dumped", "segmentation fault", "free()"]):
        status = "crash"
    elif timeout_hit:
        status = "timeout"
    elif "optimal solution:" in lower:
        status = "optimal"
    elif "search completed" in lower:
        status = "completed"

    cost_match = re.search(r"Cost\s*=\s*(-?\d+(?:\.\d+)?)", text)
    cost = float(cost_match.group(1)) if cost_match else None
    makespan = -cost if cost is not None else None

    best_quality_match = re.search(r"Best solution found had quality\s+(-?\d+(?:\.\d+)?)", text)
    best_quality = float(best_quality_match.group(1)) if best_quality_match else None

    nodes_match = re.search(r"Number of nodes processed:\s*(\d+)", text)
    nodes_processed = int(nodes_match.group(1)) if nodes_match else None

    tree_depth_match = re.search(r"Tree depth:\s*(\d+)", text)
    tree_depth = int(tree_depth_match.group(1)) if tree_depth_match else None

    search_wall_match = re.search(r"Search wall-clock time:\s*([0-9.]+) seconds", text)
    search_wall_seconds = float(search_wall_match.group(1)) if search_wall_match else None

    ll_vars_match = re.search(r"Number of LL Variables:\s*(\d+)", text)
    ll_vars = int(ll_vars_match.group(1)) if ll_vars_match else None

    return {
        "mibs_status": status,
        "mibs_cost": cost,
        "mibs_makespan": makespan,
        "mibs_best_quality": best_quality,
        "mibs_nodes_processed": nodes_processed,
        "mibs_tree_depth": tree_depth,
        "mibs_search_wall_seconds": search_wall_seconds,
        "mibs_ll_vars": ll_vars,
    }


def run_mibs_via_wsl(
    mps_path: Path,
    aux_path: Path,
    timeout_seconds: int,
    mibs_binary: str,
    wsl_run_dir: str,
) -> Dict[str, object]:
    """Run MibS in WSL by copying files to a no-space Linux directory first."""
    wsl_mps_src = windows_to_wsl_path(mps_path)
    wsl_aux_src = windows_to_wsl_path(aux_path)
    mps_name = mps_path.name
    aux_name = aux_path.name

    command = (
        "set -e; "
        f"mkdir -p '{wsl_run_dir}'; "
        f"cp '{wsl_mps_src}' '{wsl_run_dir}/{mps_name}'; "
        f"cp '{wsl_aux_src}' '{wsl_run_dir}/{aux_name}'; "
        f"cd '{wsl_run_dir}'; "
        f"timeout {timeout_seconds} '{mibs_binary}' "
        f"-Alps_instance '{mps_name}' -MibS_auxiliaryInfoFile '{aux_name}'"
    )

    t0 = time.time()
    proc = subprocess.run(
        ["wsl", "bash", "-lc", command],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    timeout_hit = proc.returncode == 124
    parsed = parse_mibs_output(output, timeout_hit=timeout_hit)

    # Always include full output and command for debugging
    parsed.update(
        {
            "mibs_return_code": proc.returncode,
            "mibs_timeout_hit": timeout_hit,
            "mibs_wall_time_seconds": elapsed,
            "mibs_output_excerpt": output[-2000:].replace("\n", "\\n"),
            "mibs_output_full": output.replace("\n", "\\n"),
            "mibs_command": command,
            "mibs_mps_path": str(mps_path),
            "mibs_aux_path": str(aux_path),
        }
    )
    if parsed["mibs_status"] == "failed" or parsed["mibs_status"] == "crash":
        print("\n[DEBUG] MibS run failed or crashed!")
        print(f"  Command: {command}")
        print(f"  MPS path: {mps_path}")
        print(f"  AUX path: {aux_path}")
        print(f"  Return code: {proc.returncode}")
        print(f"  Timeout hit: {timeout_hit}")
        print(f"  Full output:\n{output}")
    return parsed


def run_grid_sensitivity_mibs(
    output_dir: str,
    machines_range: Tuple[int, int, int],
    jobs_range: Tuple[int, int, int],
    budget_multiplier: float,
    repetitions: int,
    timeout_seconds: int,
    mibs_binary: str,
    wsl_run_dir: str,
    stop_on_timeout: bool,
    max_timeouts: int,
    max_runs: int,
    skip_job_counts: Tuple[int, ...],
    start_machines: int,
    start_jobs: int,
    start_repetition: int,
) -> Path:
    """Generate instances, run MibS, and write CSV results."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_dir) / f"grid_mibs_{ts}"
    model_dir = out_root / "models"
    out_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "grid_mibs_results.csv"

    machines_values = list(range(machines_range[0], machines_range[1] + 1, machines_range[2]))
    jobs_values = list(range(jobs_range[0], jobs_range[1] + 1, jobs_range[2]))
    skipped_jobs = set(skip_job_counts)
    jobs_values = [n for n in jobs_values if n not in skipped_jobs]

    start_tuple = None
    if start_machines > 0 and start_jobs > 0:
        start_tuple = (start_machines, start_jobs, max(1, start_repetition))

    total_planned = 0
    for m_machines in machines_values:
        for n_jobs in jobs_values:
            for rep in range(repetitions):
                if start_tuple and (m_machines, n_jobs, rep + 1) < start_tuple:
                    continue
                total_planned += 1

    total = min(total_planned, max_runs) if max_runs > 0 else total_planned
    print(f"Total runs: {total}")
    print(f"Output dir: {out_root}")
    if skipped_jobs:
        print(f"Skipping hardcoded job counts: {sorted(skipped_jobs)}")
    if start_tuple:
        print(f"Starting from: m={start_tuple[0]}, n={start_tuple[1]}, rep={start_tuple[2]}")

    fields = [
        "timestamp",
        "instance_name",
        "m_machines",
        "n_jobs",
        "budget_multiplier",
        "budget",
        "repetition",
        "seed",
        "mibs_status",
        "mibs_return_code",
        "mibs_timeout_hit",
        "mibs_wall_time_seconds",
        "mibs_search_wall_seconds",
        "mibs_nodes_processed",
        "mibs_tree_depth",
        "mibs_makespan",
        "mibs_best_quality",
    ]

    run_idx = 0
    timeout_count = 0
    stop_requested = False
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for m_machines in machines_values:
            if stop_requested:
                break
            for n_jobs in jobs_values:
                if stop_requested:
                    break
                for rep in range(repetitions):
                    if start_tuple and (m_machines, n_jobs, rep + 1) < start_tuple:
                        continue

                    if max_runs > 0 and run_idx >= max_runs:
                        stop_requested = True
                        break

                    run_idx += 1
                    seed = hash((n_jobs, m_machines, budget_multiplier, rep)) % (2**32)
                    instance = generate_random_instance(
                        n_jobs=n_jobs,
                        m_machines=m_machines,
                        budget_multiplier=budget_multiplier,
                        seed=seed,
                    )

                    instance_name = f"grid_m{m_machines}_j{n_jobs}_r{rep+1}"
                    instance.metadata["repetition"] = rep + 1
                    instance.metadata["custom_name"] = instance_name

                    # Generate strict-template model files.
                    mps_path, aux_path = generate_mps_aux_files_strict_template(instance, str(model_dir))
                    mps_path = Path(mps_path)
                    aux_path = Path(aux_path)

                    # Rename generic file names to explicit grid names.
                    target_mps = model_dir / f"{instance_name}.mps"
                    target_aux = model_dir / f"{instance_name}.txt"
                    mps_path.replace(target_mps)
                    aux_path.replace(target_aux)

                    print(
                        f"[{run_idx}/{total}] m={m_machines}, n={n_jobs}, rep={rep+1} "
                        f"budget={instance.budget:.2f}"
                    )


                    mibs = run_mibs_via_wsl(
                        mps_path=target_mps,
                        aux_path=target_aux,
                        timeout_seconds=timeout_seconds,
                        mibs_binary=mibs_binary,
                        wsl_run_dir=wsl_run_dir,
                    )

                    row = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "instance_name": instance_name,
                        "m_machines": m_machines,
                        "n_jobs": n_jobs,
                        "budget_multiplier": budget_multiplier,
                        "budget": f"{instance.budget:.4f}",
                        "repetition": rep + 1,
                        "seed": seed,
                    }
                    for k in fields:
                        if k in mibs:
                            row[k] = mibs[k]

                    writer.writerow(row)
                    f.flush()

                    if mibs["mibs_status"] == "timeout":
                        timeout_count += 1
                        if stop_on_timeout:
                            print("  -> stopping early due to --stop-on-timeout")
                            stop_requested = True
                            break
                        if max_timeouts > 0 and timeout_count >= max_timeouts:
                            print(f"  -> stopping early after reaching --max-timeouts={max_timeouts}")
                            stop_requested = True
                            break

                    # Extra debug for failed/crash
                    if mibs["mibs_status"] == "failed" or mibs["mibs_status"] == "crash":
                        print("\n[DEBUG] Full MibS output for failed/crash:")
                        print(mibs.get("mibs_output_full", "<no output>"))
                        print(f"[DEBUG] Command: {mibs.get('mibs_command')}")
                        print(f"[DEBUG] MPS: {mibs.get('mibs_mps_path')}")
                        print(f"[DEBUG] AUX: {mibs.get('mibs_aux_path')}")

                    print(
                        f"  -> {mibs['mibs_status']} | cost={mibs['mibs_cost']} "
                        f"| makespan={mibs['mibs_makespan']} | t={mibs['mibs_wall_time_seconds']:.2f}s"
                    )

    print(f"Timeouts observed: {timeout_count}")
    print(f"Done. CSV: {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MibS grid sensitivity with strict MPS/AUX export")
    parser.add_argument("--output-dir", type=str, default="mibs_comparison/experiments/test_output")
    parser.add_argument("--machines-min", type=int, default=2)
    parser.add_argument("--machines-max", type=int, default=12)
    parser.add_argument("--machines-step", type=int, default=2)
    parser.add_argument("--jobs-min", type=int, default=4)
    parser.add_argument("--jobs-max", type=int, default=24)
    parser.add_argument("--jobs-step", type=int, default=4)
    parser.add_argument("--budget-multiplier", type=float, default=1.3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--stop-on-timeout", action="store_true", help="Stop immediately after first timeout")
    parser.add_argument("--max-timeouts", type=int, default=0, help="Stop after this many timeouts (0 = no limit)")
    parser.add_argument("--max-runs", type=int, default=0, help="Maximum total runs to execute (0 = no limit)")
    parser.add_argument("--start-machines", type=int, default=0, help="Start from this machine count (0 = from beginning)")
    parser.add_argument("--start-jobs", type=int, default=0, help="Start from this job count (0 = from beginning)")
    parser.add_argument("--start-repetition", type=int, default=1, help="Start from this repetition index (1-based)")
    parser.add_argument("--mibs-binary", type=str, default="/home/ole/mibs_build/dist/bin/mibs")
    parser.add_argument("--wsl-run-dir", type=str, default="/home/ole/mibs_grid_sensitivity")

    args = parser.parse_args()

    run_grid_sensitivity_mibs(
        output_dir=args.output_dir,
        machines_range=(args.machines_min, args.machines_max, args.machines_step),
        jobs_range=(args.jobs_min, args.jobs_max, args.jobs_step),
        budget_multiplier=args.budget_multiplier,
        repetitions=args.repetitions,
        timeout_seconds=args.timeout_seconds,
        mibs_binary=args.mibs_binary,
        wsl_run_dir=args.wsl_run_dir,
        stop_on_timeout=args.stop_on_timeout,
        max_timeouts=args.max_timeouts,
        max_runs=args.max_runs,
        skip_job_counts=(12,),
        start_machines=args.start_machines,
        start_jobs=args.start_jobs,
        start_repetition=args.start_repetition,
    )


if __name__ == "__main__":
    main()


'''
& "c:/Users/oleda/.vscode/Solving stuff with Gurobi/.venv311/Scripts/python.exe" -m mibs_comparison.formulation.test_grid_sensitivity_mibs --budget-multiplier 2.5 --machines-min 8 --machines-max 12 --machines-step 4 --jobs-min 4 --jobs-max 20 --jobs-step 4 --repetitions 5 --timeout-seconds 3600 --output-dir "mibs_comparison/experiments/test_output"

'''