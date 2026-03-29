# MibS Comparison Project

This project compares the custom branch-and-bound solver for the bilevel knapsack-scheduling problem with MibS (Mixed Integer Bilevel Solver).

## Quick Start

### 1. Installation

**Option A: Conda (Recommended)**
```bash
conda create -n mibs_comparison python=3.11
conda activate mibs_comparison
conda install -c conda-forge coinbrew
pip install -r requirements.txt
```

**Option B: Docker**
```bash
docker pull coinor/coin-or-optimization-suite
```

See `MIBS_SETUP_GUIDE.md` for detailed installation instructions.

### 2. Verify Installation

```bash
python tests/test_mibs_installation.py
```

### 3. Run Small Test

```bash
python experiments/run_small_tests.py
```

## Project Structure

- `formulation/`: Problem representation and MPS generation
- `solvers/`: Wrappers for MibS and existing BnB solver
- `instances/`: Test instances and MPS files
- `experiments/`: Comparison scripts
- `results/`: Output data and analysis
- `tests/`: Unit tests

## Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)**: Step-by-step getting started guide
- **[FORMULATION.md](FORMULATION.md)**: Complete linear bilevel MILP formulation with MPS translation
- **[MIBS_SETUP_GUIDE.md](../MIBS_SETUP_GUIDE.md)**: Detailed setup and MibS explanation
- **[MIBS_PROJECT_STRUCTURE.md](../MIBS_PROJECT_STRUCTURE.md)**: Project organization and workflow
- **[STATUS.md](STATUS.md)**: Current project status

## Status

- [ ] Phase 1: MibS installation
- [ ] Phase 2: Small instance testing (5 instances)
- [ ] Phase 3: Automation scripts
- [ ] Phase 4: Performance estimation
- [ ] Phase 5: Full comparison (180 instances)

## Results

Results will be saved in `results/` directory:
- Small test comparison: `results/small_test_comparison.csv`
- Full comparison: `results/full_comparison.csv`

## Contact

For questions about MibS setup, see GitHub issues or COIN-OR forums.
