"""
Helper function to regenerate instances from seeds.

This replicates the instance generation logic from test_grid_sensitivity.py
so we can recreate instances from just the seed.
"""

import random
from typing import Tuple, List


def regenerate_from_seed(n_jobs: int, seed: int, 
                        duration_range: Tuple[int, int] = (5, 50),
                        price_range: Tuple[int, int] = (10, 100)) -> Tuple[List[int], List[int]]:
    """Regenerate job durations and prices from seed.
    
    This matches the logic in test_grid_sensitivity.py:
    - Uses random.seed() for reproducibility
    - Generates PRICES FIRST in [10, 100]
    - Then generates durations in [5, 50]
    
    Args:
        n_jobs: Number of job types
        seed: Random seed
        duration_range: (min, max) for durations
        price_range: (min, max) for prices
        
    Returns:
        Tuple of (durations, prices) lists
    """
    random.seed(seed)
    
    # IMPORTANT: Must generate prices first, then durations (matches test_grid_sensitivity.py)
    prices = [random.randint(*price_range) for _ in range(n_jobs)]
    durations = [random.randint(*duration_range) for _ in range(n_jobs)]
    
    return durations, prices


def calculate_budget(prices: List[int], n_machines: int, multiplier: float) -> float:
    """Calculate budget using the formula from test_grid_sensitivity.py.
    
    Budget = average_price × n_machines × multiplier
    
    Args:
        prices: List of job type prices
        n_machines: Number of machines
        multiplier: Budget multiplier (e.g., 1.3, 2.5, 5.0)
        
    Returns:
        Calculated budget
    """
    avg_price = sum(prices) / len(prices)
    return avg_price * n_machines * multiplier


if __name__ == "__main__":
    # Test that regeneration matches expected values
    seed = 2334587927  # From first test in grid sensitivity
    n_jobs = 4
    
    durations, prices = regenerate_from_seed(n_jobs, seed)
    print(f"Seed {seed}:")
    print(f"  Durations: {durations}")
    print(f"  Prices: {prices}")
    
    # Calculate budget for 2 machines with 1.3 multiplier
    n_machines = 2
    multiplier = 1.3
    budget = calculate_budget(prices, n_machines, multiplier)
    print(f"  Budget ({n_machines}m × {multiplier}): {budget:.2f}")
    
    # This should match the first entry in sensitivity_grid_cleaned.csv:
    # 2m, 4j, budget=91.00, seed=2334587927
