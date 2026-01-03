"""Fix sensitivity results by extracting pruning metrics from JSON files."""

import pandas as pd
import json
from pathlib import Path

# Load the original CSV
csv_file = Path("results/sensitivity/sensitivity_jobs_20260102_141304.csv")
df = pd.read_csv(csv_file)

print(f"Processing {len(df)} rows from {csv_file.name}")
print()

# Update each row with correct pruning metrics
fixed_rows = []
for idx, row in df.iterrows():
    # Construct the instance name from the row data
    n_jobs = int(row['n_jobs'])
    m_machines = int(row['m_machines'])
    budget_mult = row['budget_multiplier']
    seed = int(row['seed'])
    
    instance_name = f"sensitivity_{n_jobs}j_{m_machines}m_{budget_mult}b_s{seed}"
    
    # Find the corresponding metrics JSON file
    metrics_files = list(Path("logs/sensitivity").glob(f"{instance_name}_*_metrics.json"))
    
    if metrics_files:
        # Use the most recent one if multiple exist
        metrics_file = sorted(metrics_files)[-1]
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Extract only bound_dominated pruning (algorithmic contribution)
            pruning_reasons = metrics.get('pruning_reasons', {})
            pruned_bound = pruning_reasons.get('bound_dominated', 0)
            
            # Calculate pruning rate
            nodes_explored = row['nodes_explored']
            if nodes_explored > 0:
                pruning_rate = pruned_bound / nodes_explored
            else:
                pruning_rate = 0.0
            
            # Update the row (remove pruned_budget, update pruned_bound and pruning_rate)
            fixed_row = {
                'parameter': row['parameter'],
                'value': row['value'],
                'repetition': row['repetition'],
                'seed': row['seed'],
                'n_jobs': row['n_jobs'],
                'm_machines': row['m_machines'],
                'budget_multiplier': row['budget_multiplier'],
                'runtime': row['runtime'],
                'nodes_explored': row['nodes_explored'],
                'pruned_bound': pruned_bound,
                'pruning_rate': pruning_rate,
                'best_makespan': row['best_makespan'],
                'status': row['status']
            }
            
            fixed_rows.append(fixed_row)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} rows...")
            
        except Exception as e:
            print(f"Error processing {instance_name}: {e}")
            fixed_rows.append({k: v for k, v in row.items() if k != 'pruned_budget'})
    else:
        print(f"Warning: No metrics file found for {instance_name}")
        fixed_rows.append({k: v for k, v in row.items() if k != 'pruned_budget'})

# Create new dataframe
fixed_df = pd.DataFrame(fixed_rows)

# Save to new CSV file
output_file = csv_file.parent / f"{csv_file.stem}_fixed.csv"
fixed_df.to_csv(output_file, index=False)

print()
print(f"✓ Fixed results saved to: {output_file}")
print()
print("Sample of corrected data:")
print(fixed_df[['n_jobs', 'nodes_explored', 'pruned_bound', 'pruning_rate']].head(10).to_string(index=False))

