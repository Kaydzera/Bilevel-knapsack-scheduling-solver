"""
Bilevel problem instance representation.

This module defines the BilevelInstance class which represents
instances of the bilevel knapsack-scheduling problem in a format
suitable for both your BnB solver and MibS.

Mathematical Formulation (Linear Bilevel MILP):
    Upper Level (Leader - Maximize Makespan):
        max  y_hat
        s.t. Σᵢ pᵢ × xᵢ ≤ B
             xᵢ ∈ ℤ₊

    Lower Level (Follower - Minimize Makespan):
        min  y_hat
        s.t. Σₖ yᵢₖ = xᵢ  ∀i
             y_hat ≥ Σᵢ dᵢ × yᵢₖ  ∀k
             yᵢₖ ∈ ℤ₊

See FORMULATION.md for complete details and MPS translation guide.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json
import pandas as pd


@dataclass
class BilevelInstance:
    """Represents a bilevel knapsack-scheduling problem instance.
    
    Leader (upper level): Select job types to maximize makespan
    Follower (lower level): Schedule jobs to minimize makespan
    
    Attributes:
        n_job_types: Number of distinct job types available
        n_machines: Number of identical parallel machines
        durations: List of processing times for each job type
        prices: List of costs for each job type
        budget: Total budget available to leader
        seed: Random seed used to generate this instance
        metadata: Optional dict with additional info (e.g., from grid sensitivity)
    """
    n_job_types: int
    n_machines: int
    durations: List[int]
    prices: List[int]
    budget: float
    seed: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate instance data."""
        assert len(self.durations) == self.n_job_types
        assert len(self.prices) == self.n_job_types
        assert all(d > 0 for d in self.durations), "Durations must be positive"
        assert all(p > 0 for p in self.prices), "Prices must be positive"
        assert self.budget > 0, "Budget must be positive"
        assert self.n_machines > 0, "Must have at least one machine"
        
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            'n_job_types': self.n_job_types,
            'n_machines': self.n_machines,
            'durations': self.durations,
            'prices': self.prices,
            'budget': self.budget,
            'seed': self.seed,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BilevelInstance':
        """Deserialize from dictionary."""
        return cls(
            n_job_types=data['n_job_types'],
            n_machines=data['n_machines'],
            durations=data['durations'],
            prices=data['prices'],
            budget=data['budget'],
            seed=data['seed'],
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_grid_sensitivity_row(cls, row: pd.Series) -> 'BilevelInstance':
        """Create instance from test_grid_sensitivity.py CSV row.
        
        Args:
            row: pandas Series with columns from sensitivity_grid_cleaned.csv
            
        Note:
            The CSV doesn't store durations/prices arrays, only results.
            You'll need to regenerate the instance using the saved seed.
        """
        # Extract basic info
        n_machines = int(row['m_machines'])
        n_jobs = int(row['n_jobs'])
        budget = float(row['budget'])
        seed = int(row['seed'])
        multiplier = float(row['budget_multiplier'])
        rep = int(row['repetition'])
        
        # Need to regenerate instance using same logic as test_grid_sensitivity.py
        # This requires access to generate_random_instance()
        from .generate_instance import regenerate_from_seed
        
        durations, prices = regenerate_from_seed(n_jobs, seed)
        
        metadata = {
            'source': 'grid_sensitivity',
            'budget_multiplier': multiplier,
            'repetition': rep,
            'timestamp': row['timestamp'],
            'ceiling_status': row['ceiling_status'],
            'ceiling_makespan': float(row['ceiling_final']),
            'ceiling_time': float(row['ceiling_time']),
            'maxlpt_status': row['maxlpt_status'],
            'maxlpt_makespan': float(row['maxlpt_final']),
            'maxlpt_time': float(row['maxlpt_time'])
        }
        
        return cls(
            n_job_types=n_jobs,
            n_machines=n_machines,
            durations=durations,
            prices=prices,
            budget=budget,
            seed=seed,
            metadata=metadata
        )
    
    def save_json(self, filepath: str):
        """Save instance to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'BilevelInstance':
        """Load instance from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_instance_id(self) -> str:
        """Generate unique identifier for this instance."""
        return f"{self.n_machines}m_{self.n_job_types}j_rep{self.metadata.get('repetition', 0)}"
    
    def __repr__(self) -> str:
        return (f"BilevelInstance({self.n_machines}m, {self.n_job_types}j, "
                f"budget={self.budget:.0f}, seed={self.seed})")


def load_instances_from_csv(csv_path: str, limit: int = None) -> List[BilevelInstance]:
    """Load instances from sensitivity grid CSV.
    
    Args:
        csv_path: Path to sensitivity_grid_cleaned.csv
        limit: Optional limit on number of instances to load
        
    Returns:
        List of BilevelInstance objects
    """
    df = pd.read_csv(csv_path)
    
    if limit is not None:
        df = df.head(limit)
    
    instances = []
    for _, row in df.iterrows():
        try:
            instance = BilevelInstance.from_grid_sensitivity_row(row)
            instances.append(instance)
        except Exception as e:
            print(f"Warning: Failed to load instance from row: {e}")
            continue
    
    return instances


if __name__ == "__main__":
    # Example usage
    instance = BilevelInstance(
        n_job_types=4,
        n_machines=2,
        durations=[10, 20, 15, 25],
        prices=[5, 8, 6, 10],
        budget=20,
        seed=42,
        metadata={'source': 'manual_test'}
    )
    
    print(instance)
    print(f"ID: {instance.get_instance_id()}")
    
    # Save and load
    instance.save_json('test_instance.json')
    loaded = BilevelInstance.load_json('test_instance.json')
    print(f"Loaded: {loaded}")
