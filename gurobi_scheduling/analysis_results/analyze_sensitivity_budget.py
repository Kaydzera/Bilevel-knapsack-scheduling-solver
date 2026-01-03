"""Analyze sensitivity analysis results for budget parameter."""

import pandas as pd
from pathlib import Path

# Find the most recent sensitivity_budget CSV file
results_dir = Path("../results/sensitivity")
csv_files = list(results_dir.glob("sensitivity_budget_*.csv"))
if not csv_files:
    print("Error: No sensitivity_budget CSV file found!")
    exit(1)

# Use the most recent file
csv_file = sorted(csv_files)[-1]
print(f"Analyzing: {csv_file}\n")

# Load data
df = pd.read_csv(csv_file)

# Group by budget multiplier
grouped = df.groupby('value')

print("="*120)
print("SENSITIVITY ANALYSIS: BUDGET MULTIPLIER")
print("="*120)
print()

# Create summary table
summary_data = []

for budget_mult, group in grouped:
    summary_data.append({
        'budget_mult': budget_mult,
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

print("SUMMARY BY BUDGET MULTIPLIER")
print("="*120)
print()
print(summary_df.to_string(index=False))

print()
print("="*120)
print("KEY OBSERVATIONS")
print("="*120)
print()

# Find critical points
max_budget_tested = summary_df['budget_mult'].max()
timeout_rows = summary_df[summary_df['timeouts'] > 0]

if len(timeout_rows) > 0:
    first_timeout_budget = timeout_rows['budget_mult'].min()
    print(f"✓ First timeout occurred at: {first_timeout_budget}× budget multiplier")
else:
    print(f"✓ No timeouts encountered up to {max_budget_tested}× budget multiplier")

print(f"✓ Total budget multiplier values tested: {len(summary_df)}")
print(f"✓ Range: {summary_df['budget_mult'].min():.1f}× to {summary_df['budget_mult'].max():.1f}× minimum cost")
print(f"✓ Total test instances: {len(df)}")

print()
print("✓ Runtime scaling:")
for idx, row in summary_df.iterrows():
    print(f"  {row['budget_mult']:.1f}× budget: {row['avg_runtime']:.2f}s (min: {row['min_runtime']:.2f}s, max: {row['max_runtime']:.2f}s)")

print()
print("✓ Node exploration scaling:")
for idx, row in summary_df.iterrows():
    print(f"  {row['budget_mult']:.1f}× budget: {int(row['avg_nodes_explored'])} nodes (min: {int(row['min_nodes_explored'])}, max: {int(row['max_nodes_explored'])})")

print()
print("✓ Pruning effectiveness:")
for idx, row in summary_df.iterrows():
    print(f"  {row['budget_mult']:.1f}× budget: {row['avg_pruning_rate']*100:.1f}% pruned")

print()
print("="*120)

# Export summary
summary_output = Path("sensitivity_budget_summary.csv")
summary_df.to_csv(summary_output, index=False)
print(f"\nSummary table exported to: {summary_output}")

# Export full view
full_output = Path("sensitivity_budget_full_view.txt")
with open(full_output, 'w') as f:
    f.write("="*120 + "\n")
    f.write("SENSITIVITY ANALYSIS: BUDGET MULTIPLIER - FULL RESULTS\n")
    f.write("="*120 + "\n\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")
print(f"Full data view exported to: {full_output}")

print("\nAnalysis complete!")
