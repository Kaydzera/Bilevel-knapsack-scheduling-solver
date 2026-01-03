"""Analyze sensitivity analysis results for job types parameter."""

import pandas as pd
from pathlib import Path

# Find the most recent sensitivity_jobs CSV file
results_dir = Path("../results/sensitivity")
csv_files = list(results_dir.glob("sensitivity_jobs_*_fixed.csv"))
if not csv_files:
    # Fall back to original files if no fixed version exists
    csv_files = list(results_dir.glob("sensitivity_jobs_*.csv"))
if not csv_files:
    print("Error: No sensitivity_jobs CSV file found!")
    exit(1)

# Use the most recent file
csv_file = sorted(csv_files)[-1]
print(f"Analyzing: {csv_file}\n")

# Load data
df = pd.read_csv(csv_file)

# Group by number of jobs
grouped = df.groupby('value')

print("="*120)
print("SENSITIVITY ANALYSIS: JOB TYPES")
print("="*120)
print()

# Create summary table
summary_data = []

for n_jobs, group in grouped:
    n_jobs = int(n_jobs)
    
    summary_data.append({
        'n_jobs': n_jobs,
        'repetitions': len(group),
        'avg_runtime': group['runtime'].mean(),
        'min_runtime': group['runtime'].min(),
        'max_runtime': group['runtime'].max(),
        'avg_nodes_explored': group['nodes_explored'].mean(),
        'min_nodes_explored': group['nodes_explored'].min(),
        'max_nodes_explored': group['nodes_explored'].max(),
        'avg_pruned_bound': group['pruned_bound'].mean(),
        'avg_pruning_rate': group['pruning_rate'].mean(),
        'avg_makespan': group['best_makespan'].mean(),
        'min_makespan': group['best_makespan'].min(),
        'max_makespan': group['best_makespan'].max(),
        'successes': (group['status'] == 'success').sum(),
        'timeouts': (group['status'] == 'timeout').sum(),
    })

summary_df = pd.DataFrame(summary_data)

# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

print("SUMMARY BY NUMBER OF JOB TYPES")
print("="*120)
print()
print(summary_df.to_string(index=False))

print()
print("="*120)
print("KEY OBSERVATIONS")
print("="*120)
print()

# Find critical points
max_jobs_tested = summary_df['n_jobs'].max()
timeout_rows = summary_df[summary_df['timeouts'] > 0]

if len(timeout_rows) > 0:
    first_timeout_jobs = timeout_rows['n_jobs'].min()
    print(f"✓ First timeout occurred at: {first_timeout_jobs} job types")
else:
    print(f"✓ No timeouts encountered up to {max_jobs_tested} job types")

print(f"✓ Total job type values tested: {len(summary_df)}")
print(f"✓ Range: {summary_df['n_jobs'].min()} to {summary_df['n_jobs'].max()} jobs")
print(f"✓ Total test instances: {len(df)}")

# Runtime scaling
print(f"\n✓ Runtime scaling:")
for _, row in summary_df.iterrows():
    print(f"  {int(row['n_jobs'])} jobs: {row['avg_runtime']:.2f}s (min: {row['min_runtime']:.2f}s, max: {row['max_runtime']:.2f}s)")

# Node scaling
print(f"\n✓ Node exploration scaling:")
for _, row in summary_df.iterrows():
    print(f"  {int(row['n_jobs'])} jobs: {row['avg_nodes_explored']:.0f} nodes (min: {row['min_nodes_explored']:.0f}, max: {row['max_nodes_explored']:.0f})")

# Pruning effectiveness
print(f"\n✓ Pruning effectiveness:")
for _, row in summary_df.iterrows():
    print(f"  {int(row['n_jobs'])} jobs: {row['avg_pruning_rate']*100:.1f}% pruned")

print()
print("="*120)

# Export detailed CSV
output_file = Path("sensitivity_jobs_summary.csv")
summary_df.to_csv(output_file, index=False)
print(f"\nSummary table exported to: {output_file}")

# Also export full formatted view
output_txt = Path("sensitivity_jobs_full_view.txt")
with open(output_txt, 'w') as f:
    f.write("="*120 + "\n")
    f.write("SENSITIVITY ANALYSIS: JOB TYPES - FULL DATA\n")
    f.write("="*120 + "\n\n")
    f.write(df.to_string(index=False))
    f.write("\n\n" + "="*120 + "\n")
    f.write(f"Total instances: {len(df)}\n")
    f.write(f"Success rate: {(df['status'] == 'success').sum()}/{len(df)}\n")
    f.write("="*120 + "\n")

print(f"Full data view exported to: {output_txt}")
print("\nAnalysis complete!")
