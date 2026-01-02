"""Quick viewer for test results with better formatting."""

import pandas as pd

# Load data
df = pd.read_csv("test_results_analysis.csv")

# Configure pandas for better display
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 200)  # Wider display
pd.set_option('display.max_colwidth', 35)  # Max column width

print("="*150)
print("COMPLETE TEST RESULTS - ALL 100 INSTANCES")
print("="*150)
print()

# Display full table
print(df.to_string(index=False))

print("\n" + "="*150)
print(f"Total instances: {len(df)}")
print(f"Verified: {(df['verification_status'] == 'VERIFIED').sum()}")
print(f"Timeout: {(df['verification_status'] == 'TIMEOUT').sum()}")
print(f"Average speedup (verified): {df[df['verification_status'] == 'VERIFIED']['speedup_factor'].mean():.2f}x")
print(f"Max speedup: {df['speedup_factor'].max():.2f}x")
print("="*150)
