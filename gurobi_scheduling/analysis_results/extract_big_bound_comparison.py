"""Extract and compare ceiling vs Max-LPT BnB results with enumeration cache.

Outputs:
- analysis_ceil_maxLPT_results/big_bounds_comparison.csv
- analysis_ceil_maxLPT_results/big_bounds_summary.txt
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs" / "test_big"
CACHE_FILE = ROOT / "enumeration_results_cache_big.json"
OUTPUT_DIR = ROOT / "analysis_ceil_maxLPT_results"
OUTPUT_CSV = OUTPUT_DIR / "big_bounds_comparison.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "big_bounds_summary.txt"

MIN_INSTANCE_ID = 1
MAX_INSTANCE_ID = 140

BASE_RE = re.compile(r"^(Complex_\d+_J\d+_M\d+_B\d+)$")
KEY_RE = re.compile(r"^(Complex_\d+_J\d+_M\d+_B\d+)_\d+_\d+_\[")
PARSE_RE = re.compile(r"^Complex_(\d+)_J(\d+)_M(\d+)_B(\d+)$")


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_best_makespan(metrics: Dict[str, Any]) -> Optional[float]:
    updates = metrics.get("best_bound_updates", [])
    incumbents = [u.get("incumbent") for u in updates if "incumbent" in u]
    if not incumbents:
        return None
    return float(max(incumbents))


def parse_scheme_from_log(log_path: Path) -> str:
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Scheme:" in line:
                    return line.split("Scheme:", 1)[1].strip()
    except Exception:
        return "unknown"
    return "unknown"


def load_bnb_metrics() -> Dict[str, Dict[str, Dict[str, Any]]]:
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for metrics_path in LOGS_DIR.glob("*_metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        instance_name = metrics.get("instance_name", "")
        bound = None
        base_name = None
        if instance_name.endswith("_ceiling"):
            bound = "ceiling"
            base_name = instance_name[: -len("_ceiling")]
        elif instance_name.endswith("_maxlpt"):
            bound = "maxlpt"
            base_name = instance_name[: -len("_maxlpt")]
        if not base_name or not bound:
            continue
        if not BASE_RE.match(base_name):
            continue
        results.setdefault(base_name, {})[bound] = {
            "nodes_explored": metrics.get("nodes_explored"),
            "runtime": safe_float(metrics.get("total_runtime")),
            "best_makespan": extract_best_makespan(metrics),
        }
    return results


def load_enum_cache() -> Dict[str, Dict[str, Any]]:
    if not CACHE_FILE.exists():
        return {}
    with CACHE_FILE.open("r", encoding="utf-8") as f:
        cache = json.load(f)
    results: Dict[str, Dict[str, Any]] = {}
    for key, entry in cache.items():
        match = KEY_RE.match(key)
        if not match:
            continue
        base_name = match.group(1)
        results[base_name] = entry
    return results


def parse_instance_fields(base_name: str) -> Dict[str, Any]:
    match = PARSE_RE.match(base_name)
    if not match:
        return {"instance_id": None, "n_jobs": None, "machines": None, "budget": None, "scheme": "unknown"}
    instance_id = int(match.group(1))
    # Compute scheme from instance ID based on test_big.py generation logic
    scheme_map = {
        0: "uniform_ratios",
        1: "high_variance",
        2: "increasing",
        3: "random_realistic",
        4: "extreme",
        5: "strong_correlation",
        6: "subset_sum",
    }
    scheme = scheme_map.get((instance_id - 1) % 7, "unknown")
    return {
        "instance_id": instance_id,
        "n_jobs": int(match.group(2)),
        "machines": int(match.group(3)),
        "budget": int(match.group(4)),
        "scheme": scheme,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bnb = load_bnb_metrics()
    enum = load_enum_cache()

    all_instances = sorted(set(bnb.keys()) | set(enum.keys()))
    filtered_instances = []
    for base_name in all_instances:
        if base_name not in bnb:
            continue
        if "ceiling" not in bnb[base_name] or "maxlpt" not in bnb[base_name]:
            continue
        fields = parse_instance_fields(base_name)
        instance_id = fields["instance_id"]
        if instance_id is None:
            continue
        if MIN_INSTANCE_ID <= instance_id <= MAX_INSTANCE_ID:
            filtered_instances.append(base_name)

    rows = []
    missing_bnb = 0
    missing_enum = 0
    mismatch_ceiling_maxlpt = 0
    mismatch_enum_ceiling = 0
    mismatch_enum_maxlpt = 0
    enum_timeouts = 0

    for base_name in filtered_instances:
        fields = parse_instance_fields(base_name)
        ceiling = bnb.get(base_name, {}).get("ceiling", {})
        maxlpt = bnb.get(base_name, {}).get("maxlpt", {})
        enum_entry = enum.get(base_name)

        if not ceiling or not maxlpt:
            missing_bnb += 1

        if enum_entry is None:
            missing_enum += 1

        # Get scheme from parsed fields
        scheme = fields.get("scheme", "unknown")

        ceiling_mk = safe_float(ceiling.get("best_makespan"))
        maxlpt_mk = safe_float(maxlpt.get("best_makespan"))
        enum_mk = safe_float(enum_entry.get("makespan") if enum_entry else None)

        match_c_m = None
        if ceiling_mk is not None and maxlpt_mk is not None:
            match_c_m = abs(ceiling_mk - maxlpt_mk) < 0.01
            if not match_c_m:
                mismatch_ceiling_maxlpt += 1

        match_enum_ceiling = None
        match_enum_maxlpt = None
        if enum_mk is not None and ceiling_mk is not None:
            match_enum_ceiling = abs(enum_mk - ceiling_mk) < 0.01
            if not match_enum_ceiling:
                mismatch_enum_ceiling += 1
        if enum_mk is not None and maxlpt_mk is not None:
            match_enum_maxlpt = abs(enum_mk - maxlpt_mk) < 0.01
            if not match_enum_maxlpt:
                mismatch_enum_maxlpt += 1

        enum_timed_out = bool(enum_entry.get("timed_out")) if enum_entry else None
        if enum_timed_out:
            enum_timeouts += 1

        ceiling_nodes = ceiling.get("nodes_explored")
        maxlpt_nodes = maxlpt.get("nodes_explored")
        enum_nodes = enum_entry.get("nodes_evaluated") if enum_entry else None

        ceiling_runtime = safe_float(ceiling.get("runtime"))
        maxlpt_runtime = safe_float(maxlpt.get("runtime"))
        enum_runtime = safe_float(enum_entry.get("runtime") if enum_entry else None)

        ratio_nodes_m_to_c = None
        if ceiling_nodes and maxlpt_nodes is not None and ceiling_nodes != 0:
            ratio_nodes_m_to_c = maxlpt_nodes / ceiling_nodes

        ratio_runtime_m_to_c = None
        if ceiling_runtime and maxlpt_runtime is not None and ceiling_runtime != 0:
            ratio_runtime_m_to_c = maxlpt_runtime / ceiling_runtime

        speedup_enum_over_ceiling = None
        if enum_runtime and ceiling_runtime and ceiling_runtime != 0:
            speedup_enum_over_ceiling = enum_runtime / ceiling_runtime

        speedup_enum_over_maxlpt = None
        if enum_runtime and maxlpt_runtime and maxlpt_runtime != 0:
            speedup_enum_over_maxlpt = enum_runtime / maxlpt_runtime

        row = {
            "instance": base_name,
            "instance_id": fields["instance_id"],
            "n_jobs": fields["n_jobs"],
            "machines": fields["machines"],
            "budget": fields["budget"],
            "scheme": scheme,
            "ceiling_makespan": ceiling_mk,
            "ceiling_nodes": ceiling_nodes,
            "ceiling_runtime": ceiling_runtime,
            "maxlpt_makespan": maxlpt_mk,
            "maxlpt_nodes": maxlpt_nodes,
            "maxlpt_runtime": maxlpt_runtime,
            "enum_makespan": enum_mk,
            "enum_nodes": enum_nodes,
            "enum_runtime": enum_runtime,
            "enum_timed_out": enum_timed_out,
            "match_ceiling_maxlpt": match_c_m,
            "match_enum_ceiling": match_enum_ceiling,
            "match_enum_maxlpt": match_enum_maxlpt,
            "ratio_nodes_maxlpt_to_ceiling": ratio_nodes_m_to_c,
            "ratio_runtime_maxlpt_to_ceiling": ratio_runtime_m_to_c,
            "speedup_enum_over_ceiling": speedup_enum_over_ceiling,
            "speedup_enum_over_maxlpt": speedup_enum_over_maxlpt,
        }
        rows.append(row)

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with OUTPUT_CSV.open("w", encoding="utf-8") as f:
            f.write(",".join(fieldnames) + "\n")
            for row in rows:
                values = []
                for k in fieldnames:
                    v = row.get(k)
                    if v is None:
                        values.append("")
                    else:
                        values.append(str(v))
                f.write(",".join(values) + "\n")

    # Write summary
    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as f:
        f.write("Big Test Bounds Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Instances (with BnB logs): {len(filtered_instances)}\n")
        f.write(f"Instance id range: {MIN_INSTANCE_ID}-{MAX_INSTANCE_ID}\n")
        f.write(f"Missing BnB metrics: {missing_bnb}\n")
        f.write(f"Missing enumeration: {missing_enum}\n")
        f.write(f"Enumeration timeouts: {enum_timeouts}\n")
        f.write("\nMatch counts:\n")
        f.write(f"  Ceiling vs Max-LPT mismatches: {mismatch_ceiling_maxlpt}\n")
        f.write(f"  Enum vs Ceiling mismatches: {mismatch_enum_ceiling}\n")
        f.write(f"  Enum vs Max-LPT mismatches: {mismatch_enum_maxlpt}\n")

    print(f"Wrote CSV: {OUTPUT_CSV}")
    print(f"Wrote summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
