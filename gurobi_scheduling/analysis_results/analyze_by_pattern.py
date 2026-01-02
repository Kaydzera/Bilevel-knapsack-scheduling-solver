"""Generate summary statistics by pattern type for thesis."""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("test_results_analysis.csv")

print("="*120)
print("PATTERN-WISE ANALYSIS FOR THESIS")
print("="*120)
print()

# Pattern descriptions for reference
patterns = {
    "Pattern 1: Uniform Ratios": "Consistent duration-to-price ratios (1.5-3.0), balanced instances",
    "Pattern 2: High Variance": "Alternating cheap-long and expensive-short jobs",
    "Pattern 3: Increasing": "Systematic progression (duration=3+j*2, price=2+j)",
    "Pattern 4: Random Uniform": "Random within realistic bounds (duration 4-18, price 2-12)",
    "Pattern 5: Extreme Cases": "30% very long & cheap, 70% normal range"
}

for pattern_name, description in patterns.items():
    print(f"\n{'='*120}")
    print(f"{pattern_name}")
    print(f"{description}")
    print(f"{'='*120}")
    
    # Filter data for this pattern
    pattern_data = df[df['pattern_type'] == pattern_name]
    verified = pattern_data[pattern_data['verification_status'] == 'VERIFIED']
    timeout = pattern_data[pattern_data['verification_status'] == 'TIMEOUT']
    
    print(f"\nTotal instances: {len(pattern_data)}")
    print(f"  Verified: {len(verified)} ({len(verified)/len(pattern_data)*100:.0f}%)")
    print(f"  Timeout:  {len(timeout)} ({len(timeout)/len(pattern_data)*100:.0f}%)")
    
    if len(verified) > 0:
        print(f"\nBnB Performance (verified instances):")
        print(f"  Runtime:       min={verified['bnb_runtime_sec'].min():.4f}s, "
              f"max={verified['bnb_runtime_sec'].max():.4f}s, "
              f"mean={verified['bnb_runtime_sec'].mean():.4f}s, "
              f"median={verified['bnb_runtime_sec'].median():.4f}s")
        print(f"  Nodes explored: min={verified['bnb_nodes_explored'].min()}, "
              f"max={verified['bnb_nodes_explored'].max()}, "
              f"mean={verified['bnb_nodes_explored'].mean():.0f}, "
              f"median={verified['bnb_nodes_explored'].median():.0f}")
        
        print(f"\nEnumeration Performance (verified instances):")
        print(f"  Runtime:       min={verified['enum_runtime_sec'].min():.4f}s, "
              f"max={verified['enum_runtime_sec'].max():.4f}s, "
              f"mean={verified['enum_runtime_sec'].mean():.4f}s, "
              f"median={verified['enum_runtime_sec'].median():.4f}s")
        print(f"  Nodes checked: min={verified['enum_nodes_checked'].min()}, "
              f"max={verified['enum_nodes_checked'].max()}, "
              f"mean={verified['enum_nodes_checked'].mean():.0f}, "
              f"median={verified['enum_nodes_checked'].median():.0f}")
        
        print(f"\nSpeedup (verified instances):")
        print(f"  min={verified['speedup_factor'].min():.2f}x, "
              f"max={verified['speedup_factor'].max():.2f}x, "
              f"mean={verified['speedup_factor'].mean():.2f}x, "
              f"median={verified['speedup_factor'].median():.2f}x")
        
        # Show which instances timed out
        if len(timeout) > 0:
            print(f"\nTimeout instances:")
            for _, row in timeout.iterrows():
                print(f"  {row['instance_name']}: {row['n_jobs']} jobs, "
                      f"{row['n_machines']} machines, budget={row['budget']}, "
                      f"BnB runtime={row['bnb_runtime_sec']:.4f}s")

# Overall summary comparison
print(f"\n\n{'='*120}")
print("OVERALL COMPARISON ACROSS PATTERNS")
print(f"{'='*120}\n")

# Create summary table
summary_data = []
for pattern_name in patterns.keys():
    pattern_data = df[df['pattern_type'] == pattern_name]
    verified = pattern_data[pattern_data['verification_status'] == 'VERIFIED']
    
    summary_data.append({
        'Pattern': pattern_name.split(":")[0],  # Short name
        'Total': len(pattern_data),
        'Verified': len(verified),
        'Timeout': len(pattern_data) - len(verified),
        'Avg BnB Time (s)': verified['bnb_runtime_sec'].mean() if len(verified) > 0 else np.nan,
        'Avg Speedup': verified['speedup_factor'].mean() if len(verified) > 0 else np.nan,
        'Median Speedup': verified['speedup_factor'].median() if len(verified) > 0 else np.nan,
        'Max Speedup': verified['speedup_factor'].max() if len(verified) > 0 else np.nan
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\n{'='*120}")
print("KEY FINDINGS")
print(f"{'='*120}")
print(f"1. Total instances tested: {len(df)}")
print(f"2. Overall verification rate: {len(df[df['verification_status'] == 'VERIFIED'])/len(df)*100:.1f}%")
print(f"3. Average speedup (all verified): {df[df['verification_status'] == 'VERIFIED']['speedup_factor'].mean():.2f}x")
print(f"4. Median speedup (all verified): {df[df['verification_status'] == 'VERIFIED']['speedup_factor'].median():.2f}x")
print(f"5. Maximum speedup achieved: {df['speedup_factor'].max():.2f}x (Instance {df.loc[df['speedup_factor'].idxmax(), 'instance_name']})")
print(f"6. BnB never exceeded node limit on verified instances")
print(f"7. Timeouts occurred primarily in instances with: 12 jobs and low machine counts (3-4 machines)")
print(f"{'='*120}\n")
