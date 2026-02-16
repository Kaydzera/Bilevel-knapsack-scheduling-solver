"""Generate readable text comparison from big_bounds_comparison.csv.

Reads:
  - big_bounds_comparison.csv

Writes:
  - big_bounds_comparison_readable.txt
"""

import csv
from pathlib import Path
from collections import defaultdict

INPUT_CSV = Path(__file__).parent / "big_bounds_comparison.csv"
OUTPUT_TXT = Path(__file__).parent / "big_bounds_comparison_readable.txt"


def safe_float(val):
    try:
        return float(val) if val and val != '' else None
    except (ValueError, TypeError):
        return None


def safe_int(val):
    try:
        return int(val) if val and val != '' else None
    except (ValueError, TypeError):
        return None


def main():
    # Read CSV
    rows = []
    with INPUT_CSV.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Group by scheme
    by_scheme = defaultdict(list)
    for row in rows:
        scheme = row.get('scheme', 'unknown')
        by_scheme[scheme].append(row)
    
    # Write readable output
    with OUTPUT_TXT.open('w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("CEILING vs MAX-LPT BOUND COMPARISON - READABLE FORMAT\n")
        f.write("=" * 120 + "\n\n")
        
        f.write(f"Total instances: {len(rows)}\n")
        f.write(f"Schemes: {len(by_scheme)}\n\n")
        
        # Overall summary
        f.write("=" * 120 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 120 + "\n\n")
        
        total_match_c_m = sum(1 for r in rows if r.get('match_ceiling_maxlpt') == 'True')
        total_match_enum_c = sum(1 for r in rows if r.get('match_enum_ceiling') == 'True')
        total_match_enum_m = sum(1 for r in rows if r.get('match_enum_maxlpt') == 'True')
        total_enum_timeout = sum(1 for r in rows if r.get('enum_timed_out') == 'True')
        
        f.write(f"Ceiling vs Max-LPT matches:     {total_match_c_m}/{len(rows)}\n")
        f.write(f"Enum vs Ceiling matches:        {total_match_enum_c}/{len([r for r in rows if r.get('enum_makespan')])}\n")
        f.write(f"Enum vs Max-LPT matches:        {total_match_enum_m}/{len([r for r in rows if r.get('enum_makespan')])}\n")
        f.write(f"Enumeration timeouts:           {total_enum_timeout}\n\n")
        
        # Average metrics
        ceiling_nodes = [safe_int(r.get('ceiling_nodes')) for r in rows]
        maxlpt_nodes = [safe_int(r.get('maxlpt_nodes')) for r in rows]
        ceiling_runtime = [safe_float(r.get('ceiling_runtime')) for r in rows]
        maxlpt_runtime = [safe_float(r.get('maxlpt_runtime')) for r in rows]
        
        ceiling_nodes = [x for x in ceiling_nodes if x is not None]
        maxlpt_nodes = [x for x in maxlpt_nodes if x is not None]
        ceiling_runtime = [x for x in ceiling_runtime if x is not None]
        maxlpt_runtime = [x for x in maxlpt_runtime if x is not None]
        
        if ceiling_nodes:
            f.write(f"Average ceiling nodes:          {sum(ceiling_nodes)/len(ceiling_nodes):.1f}\n")
        if maxlpt_nodes:
            f.write(f"Average Max-LPT nodes:          {sum(maxlpt_nodes)/len(maxlpt_nodes):.1f}\n")
        if ceiling_runtime:
            f.write(f"Average ceiling runtime:        {sum(ceiling_runtime)/len(ceiling_runtime):.4f}s\n")
        if maxlpt_runtime:
            f.write(f"Average Max-LPT runtime:        {sum(maxlpt_runtime)/len(maxlpt_runtime):.4f}s\n")
        
        f.write("\n")
        
        # By scheme breakdown
        f.write("=" * 120 + "\n")
        f.write("BY SCHEME BREAKDOWN\n")
        f.write("=" * 120 + "\n\n")
        
        for scheme in sorted(by_scheme.keys()):
            scheme_rows = by_scheme[scheme]
            f.write("-" * 120 + "\n")
            f.write(f"Scheme: {scheme} ({len(scheme_rows)} instances)\n")
            f.write("-" * 120 + "\n\n")
            
            # Scheme summary
            scheme_match_c_m = sum(1 for r in scheme_rows if r.get('match_ceiling_maxlpt') == 'True')
            scheme_enum_timeout = sum(1 for r in scheme_rows if r.get('enum_timed_out') == 'True')
            
            f.write(f"  Ceiling vs Max-LPT matches: {scheme_match_c_m}/{len(scheme_rows)}\n")
            f.write(f"  Enumeration timeouts:       {scheme_enum_timeout}\n\n")
            
            # Averages
            s_ceiling_nodes = [safe_int(r.get('ceiling_nodes')) for r in scheme_rows]
            s_maxlpt_nodes = [safe_int(r.get('maxlpt_nodes')) for r in scheme_rows]
            s_ceiling_runtime = [safe_float(r.get('ceiling_runtime')) for r in scheme_rows]
            s_maxlpt_runtime = [safe_float(r.get('maxlpt_runtime')) for r in scheme_rows]
            
            s_ceiling_nodes = [x for x in s_ceiling_nodes if x is not None]
            s_maxlpt_nodes = [x for x in s_maxlpt_nodes if x is not None]
            s_ceiling_runtime = [x for x in s_ceiling_runtime if x is not None]
            s_maxlpt_runtime = [x for x in s_maxlpt_runtime if x is not None]
            
            if s_ceiling_nodes:
                f.write(f"  Avg ceiling nodes:          {sum(s_ceiling_nodes)/len(s_ceiling_nodes):.1f}\n")
            if s_maxlpt_nodes:
                f.write(f"  Avg Max-LPT nodes:          {sum(s_maxlpt_nodes)/len(s_maxlpt_nodes):.1f}\n")
            if s_ceiling_runtime:
                f.write(f"  Avg ceiling runtime:        {sum(s_ceiling_runtime)/len(s_ceiling_runtime):.4f}s\n")
            if s_maxlpt_runtime:
                f.write(f"  Avg Max-LPT runtime:        {sum(s_maxlpt_runtime)/len(s_maxlpt_runtime):.4f}s\n")
            
            # Node reduction
            if s_ceiling_nodes and s_maxlpt_nodes:
                avg_reduction = 100 * (1 - sum(s_maxlpt_nodes)/sum(s_ceiling_nodes))
                f.write(f"  Node reduction:             {avg_reduction:.1f}%\n")
            
            f.write("\n")
            
            # Instance details
            f.write("  Instance Details:\n")
            f.write("  " + "-" * 116 + "\n")
            f.write(f"  {'Instance':<35} {'J':<3} {'M':<3} {'B':<5} {'C_nodes':<8} {'M_nodes':<8} {'Match':<8} {'C_time':<8} {'M_time':<8}\n")
            f.write("  " + "-" * 116 + "\n")
            
            for row in sorted(scheme_rows, key=lambda r: safe_int(r.get('instance_id', 0)) or 0):
                inst = row.get('instance', '')
                n_jobs = row.get('n_jobs', '')
                machines = row.get('machines', '')
                budget = row.get('budget', '')
                c_nodes = row.get('ceiling_nodes', '')
                m_nodes = row.get('maxlpt_nodes', '')
                match = 'OK' if row.get('match_ceiling_maxlpt') == 'True' else 'FAIL'
                c_time = row.get('ceiling_runtime', '')
                m_time = row.get('maxlpt_runtime', '')
                
                # Format floats
                if c_time:
                    try:
                        c_time = f"{float(c_time):.4f}"
                    except:
                        pass
                if m_time:
                    try:
                        m_time = f"{float(m_time):.4f}"
                    except:
                        pass
                
                f.write(f"  {inst:<35} {n_jobs:<3} {machines:<3} {budget:<5} {c_nodes:<8} {m_nodes:<8} {match:<8} {c_time:<8} {m_time:<8}\n")
            
            f.write("\n\n")
        
        f.write("=" * 120 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 120 + "\n")
    
    print(f"Generated: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
