"""
Test minimal instance: 1 machine, 1 job type
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from formulation.bilevel_model import BilevelInstance
from formulation.mps_generator import generate_mps_aux_files


def main():
    # Minimal instance: 1 machine, 1 job type
    instance = BilevelInstance(
        n_job_types=1,
        n_machines=1,
        durations=[10],
        prices=[50],
        budget=100,
        seed=12345,
        metadata={'test': 'minimal'}
    )
    
    print(f"Testing minimal instance: {instance.get_instance_id()}")
    print(f"  Duration: {instance.durations}")
    print(f"  Price: {instance.prices}")
    print(f"  Budget: {instance.budget}")
    
    output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    mps_path, aux_path = generate_mps_aux_files(instance, output_dir)
    
    print(f"\nGenerated: {mps_path}")
    print(f"Generated: {aux_path}")
    
    print("\n" + "="*60)
    print("MPS File:")
    print("="*60)
    with open(mps_path, 'r') as f:
        print(f.read())
    
    print("\n" + "="*60)
    print("AUX File:")
    print("="*60)
    with open(aux_path, 'r') as f:
        print(f.read())


if __name__ == "__main__":
    main()
