"""Analyze sensitivity analysis results for machines parameter."""

import pandas as pd
from pathlib import Path

# Find the most recent sensitivity_machines CSV file
results_dir = Path("../results/sensitivity")
csv_files = list(results_dir.glob("sensitivity_machines_*.csv"))
if not csv_files:
    print("Error: No sensitivity_machines CSV file found!")
    exit(1)

# Use the most recent file
csv_file = sorted(csv_files)[-1]
print(f"Analyzing: {csv_file}\n")

# Load data
df = pd.read_csv(csv_file)

# Group by number of machines
grouped = df.groupby('value')

print("="*120)
print("SENSITIVITY ANALYSIS: NUMBER OF MACHINES")
print("="*120)
print()

# Create summary table
summary_data = []

for machines, group in grouped:
    summary_data.append({
        'machines': int(machines),
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

print("SUMMARY BY NUMBER OF MACHINES")
print("="*120)
print()
print(summary_df.to_string(index=False))

print()
print("="*120)
print("KEY OBSERVATIONS")
print("="*120)
print()

# Find critical points
max_machines_tested = summary_df['machines'].max()
timeout_rows = summary_df[summary_df['timeouts'] > 0]

if len(timeout_rows) > 0:
    first_timeout_machines = timeout_rows['machines'].min()
    print(f"✓ First timeout occurred at: {first_timeout_machines} machines")
else:
    print(f"✓ No timeouts encountered up to {max_machines_tested} machines")

print(f"✓ Total machine counts tested: {len(summary_df)}")
print(f"✓ Range: {summary_df['machines'].min()} to {summary_df['machines'].max()} machines")
print(f"✓ Total test instances: {len(df)}")

print()
print("✓ Runtime scaling:")
for idx, row in summary_df.iterrows():
    print(f"  {row['machines']} machines: {row['avg_runtime']:.2f}s (min: {row['min_runtime']:.2f}s, max: {row['max_runtime']:.2f}s)")

print()
print("✓ Node exploration scaling:")
for idx, row in summary_df.iterrows():
    print(f"  {row['machines']} machines: {int(row['avg_nodes_explored'])} nodes (min: {int(row['min_nodes_explored'])}, max: {int(row['max_nodes_explored'])})")

print()
print("✓ Pruning effectiveness:")
for idx, row in summary_df.iterrows():
    print(f"  {row['machines']} machines: {row['avg_pruning_rate']*100:.1f}% pruned")

print()
print("✓ Makespan trend:")
for idx, row in summary_df.iterrows():
    print(f"  {row['machines']} machines: avg makespan = {row['avg_makespan']:.1f} (min: {row['min_makespan']:.1f}, max: {row['max_makespan']:.1f})")

print()
print("="*120)

# Export summary
output_dir = Path("../Thesis_Bilevel_TUBS/sensitivity_analysis/machines")
output_dir.mkdir(parents=True, exist_ok=True)

summary_file = output_dir / "sensitivity_machines_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"\n✓ Summary exported to: {summary_file}")

# Export full view as text
full_view_file = output_dir / "sensitivity_machines_full_view.txt"
with open(full_view_file, 'w') as f:
    f.write("="*120 + "\n")
    f.write("SENSITIVITY ANALYSIS: NUMBER OF MACHINES - FULL VIEW\n")
    f.write("="*120 + "\n\n")
    f.write("SUMMARY BY NUMBER OF MACHINES\n")
    f.write("="*120 + "\n\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")
    f.write("="*120 + "\n")
    f.write("DETAILED RESULTS (ALL REPETITIONS)\n")
    f.write("="*120 + "\n\n")
    f.write(df.to_string(index=False))

print(f"✓ Full view exported to: {full_view_file}")
print()
