"""
Test script to load an instance from CSV and generate MPS/AUX files.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from formulation.bilevel_model import BilevelInstance, load_instances_from_csv
from formulation.mps_generator import generate_mps_aux_files


def main():
    # Path to cleaned CSV
    csv_path = r"c:\Users\oleda\.vscode\Solving stuff with Gurobi\gurobi_scheduling\results\sensitivity_grid\grid_20260225_143058\sensitivity_grid_cleaned.csv"
    
    # Load first instance (2m_4j)
    print("Loading instances from CSV...")
    instances = load_instances_from_csv(csv_path, limit=1)
    
    if not instances:
        print("ERROR: No instances loaded!")
        return
    
    instance = instances[0]
    print(f"\nLoaded instance: {instance.get_instance_id()}")
    print(f"  n_job_types: {instance.n_job_types}")
    print(f"  n_machines: {instance.n_machines}")
    print(f"  durations: {instance.durations}")
    print(f"  prices: {instance.prices}")
    print(f"  budget: {instance.budget}")
    print(f"  seed: {instance.seed}")
    
    # Generate MPS/AUX files
    output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    print(f"\nGenerating MPS/AUX files to {output_dir}...")
    
    mps_path, aux_path = generate_mps_aux_files(instance, output_dir)
    
    print(f"\nGenerated files:")
    print(f"  MPS: {mps_path}")
    print(f"  AUX: {aux_path}")
    
    # Display file contents
    print("\n" + "="*60)
    print("MPS File Content:")
    print("="*60)
    with open(mps_path, 'r') as f:
        print(f.read())
    
    print("\n" + "="*60)
    print("AUX File Content:")
    print("="*60)
    with open(aux_path, 'r') as f:
        print(f.read())
    
    print("\nFiles ready for MibS!")
    return mps_path, aux_path


if __name__ == "__main__":
    main()
