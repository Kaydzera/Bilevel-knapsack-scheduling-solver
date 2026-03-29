"""Convert cached test_big enumeration instances to MibS MPS/AUX files.

Reads gurobi_scheduling/enumeration_results_cache_big.json and generates one
MPS/AUX pair per cached instance using the existing formulation generator.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add mibs_comparison root to import path.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from formulation.bilevel_model import BilevelInstance
from formulation.mps_generator import generate_mps_aux_files


ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "gurobi_scheduling" / "enumeration_results_cache_big.json"
OUT_DIR = Path(__file__).resolve().parent / "test_output" / "test_big_cache_mibs"
SUMMARY_PATH = OUT_DIR / "conversion_summary.json"


def parse_cache_key(key: str) -> Tuple[str, int, int, List[Tuple[int, int]]]:
    """Parse cache key: <name>_<m>_<budget>_[(duration, price), ...]."""
    match = re.match(r"^(.*)_(\d+)_(\d+)_\[(.*)\]$", key)
    if match is None:
        raise ValueError(f"Unsupported cache key format: {key}")

    name = match.group(1)
    n_machines = int(match.group(2))
    budget = int(match.group(3))
    items_raw = "[" + match.group(4) + "]"
    item_specs = ast.literal_eval(items_raw)

    if not isinstance(item_specs, list) or not item_specs:
        raise ValueError(f"No item specs parsed from key: {key}")

    parsed_specs: List[Tuple[int, int]] = []
    for item in item_specs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Invalid item spec in key: {key}")
        duration, price = int(item[0]), int(item[1])
        parsed_specs.append((duration, price))

    return name, n_machines, budget, parsed_specs


def sanitize_filename(text: str) -> str:
    """Create filesystem-safe filename stem."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def build_instance(cache_key: str, cache_entry: Dict[str, object], index: int) -> BilevelInstance:
    """Build BilevelInstance from one cache key/entry pair."""
    name, n_machines, budget, specs = parse_cache_key(cache_key)

    durations = [d for d, _ in specs]
    prices = [p for _, p in specs]

    metadata = {
        "source": "test_big_cache",
        "cache_key": cache_key,
        "cache_name": name,
        "selection": cache_entry.get("selection"),
        "enumeration_makespan": cache_entry.get("makespan"),
        "nodes_evaluated": cache_entry.get("nodes_evaluated"),
        "runtime": cache_entry.get("runtime"),
        "timed_out": bool(cache_entry.get("timed_out", False)),
        "repetition": index,
        "filename_stem": sanitize_filename(name),
    }

    return BilevelInstance(
        n_job_types=len(specs),
        n_machines=n_machines,
        durations=durations,
        prices=prices,
        budget=float(budget),
        seed=index,
        metadata=metadata,
    )


def main() -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Cache file not found: {CACHE_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with CACHE_PATH.open("r", encoding="utf-8") as f:
        cache = json.load(f)

    if not isinstance(cache, dict):
        raise ValueError("Cache JSON must be a dictionary.")

    summary = {
        "cache_path": str(CACHE_PATH),
        "output_dir": str(OUT_DIR),
        "generated": [],
        "skipped": [],
    }

    for idx, (cache_key, cache_entry) in enumerate(cache.items()):
        if not isinstance(cache_entry, dict):
            summary["skipped"].append({"cache_key": cache_key, "reason": "entry_not_dict"})
            continue

        try:
            instance = build_instance(cache_key, cache_entry, idx)
            mps_path, aux_path = generate_mps_aux_files(instance, str(OUT_DIR))

            safe_stem = sanitize_filename(instance.metadata.get("filename_stem", instance.get_instance_id()))
            target_mps = OUT_DIR / f"{safe_stem}.mps"
            target_aux = OUT_DIR / f"{safe_stem}.txt"

            # Rename generic generator output to stable names based on instance key.
            Path(mps_path).replace(target_mps)
            Path(aux_path).replace(target_aux)

            summary["generated"].append(
                {
                    "cache_key": cache_key,
                    "mps": str(target_mps),
                    "aux": str(target_aux),
                    "n_jobs": instance.n_job_types,
                    "n_machines": instance.n_machines,
                    "budget": instance.budget,
                    "timed_out": bool(cache_entry.get("timed_out", False)),
                    "enumeration_makespan": cache_entry.get("makespan"),
                }
            )
        except Exception as exc:
            summary["skipped"].append({"cache_key": cache_key, "reason": str(exc)})

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Cache entries: {len(cache)}")
    print(f"Generated: {len(summary['generated'])}")
    print(f"Skipped: {len(summary['skipped'])}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
