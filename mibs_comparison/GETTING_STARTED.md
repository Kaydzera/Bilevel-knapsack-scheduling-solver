# Getting Started with MibS Comparison Project

This guide walks you through the first steps to set up and test the MibS comparison project.

## What We've Created So Far

The project foundation is ready:

```
mibs_comparison/
├── README.md                    # Project overview
├── requirements.txt             # Python dependencies
├── test_installation.py         # Installation verification script
├── MIBS_SETUP_GUIDE.md         # Detailed MibS setup guide (200+ lines)
├── MIBS_PROJECT_STRUCTURE.md   # Project organization (250+ lines)
└── formulation/
    ├── __init__.py              # Package initialization
    ├── bilevel_model.py         # BilevelInstance dataclass
    └── generate_instance.py     # Instance regeneration from seeds
```

## Step 1: Review Documentation

**Start here** to understand what we're building:

1. Open `MIBS_SETUP_GUIDE.md` - This explains:
   - What MibS is and how it works
   - The **linear bilevel formulation** (already MibS-compatible!)
   - Installation options (conda, source, Docker)
   - Mathematical formulation details
   - Phase-by-phase plan

2. Open `MIBS_PROJECT_STRUCTURE.md` - This shows:
   - Complete project architecture
   - What each component does
   - 5-phase implementation plan
   - Expected outputs

**Questions?** Ask about anything unclear in these docs!

## Step 2: Test Current Setup

Before installing MibS, verify the Python foundation works:

```powershell
cd mibs_comparison
python test_installation.py
```

**Expected output:**
- ✓ Python packages (numpy, pandas, matplotlib)
- ✓ Formulation code (BilevelInstance, regenerate_from_seed)
- ✗ MibS executable (not installed yet - expected!)

If you see errors on the first two, run:
```powershell
pip install -r requirements.txt
```

## Step 3: Install MibS (3 Options)

### **Option A: Conda (Recommended for Windows)**

```powershell
# Install coinbrew
conda install -c conda-forge coinbrew

# Fetch MibS
coinbrew fetch MibS@stable/2.1

# Build MibS
coinbrew build MibS

# Add to PATH (Windows)
# The executable will be in: coin-or-mibs/build/bin/mibs.exe
# Add this to your PATH or note the location
```

### **Option B: Docker (Easier, No Local Install)**

```powershell
# Pull MibS Docker image
docker pull tkralphs/mibs

# Test it runs
docker run -it tkralphs/mibs mibs --version

# Note: Will need to mount volumes to share files
```

### **Option C: Source Build (Advanced, Most Control)**

See `MIBS_SETUP_GUIDE.md` Section 2.3 for detailed instructions.

## Step 4: Verify MibS Installation

After installing, run the test again:

```powershell
python test_installation.py
```

**Expected output:**
- ✓ Python packages
- ✓ Formulation code  
- ✓ MibS executable

If MibS is not found, you may need to add it to PATH or use the full path.

## Step 5: Test Instance Loading

Let's verify we can load instances from your grid sensitivity results:

```powershell
cd formulation
python bilevel_model.py
```

This should:
- Load the first test instance from cleaned CSV
- Print instance details (durations, prices, budget, seed)
- Show instance ID

**Expected output:**
```
Loading first instance from cleaned grid sensitivity results...
Instance ID: 2m_4j_b91.00_s2334587927
...
```

If this fails, check the CSV path in bilevel_model.py (around line 140).

## Step 6: Next Steps (After Verification)

Once Steps 1-5 are complete, we'll proceed to:

**Phase 2: Small Instance Testing (Manual)**
- Select 5 test instances
- Generate MPS files
- Run MibS manually
- Understand MibS behavior

**Phase 3: MPS Generator**
- Implement mps_generator.py
- Translate bilevel formulation to MPS/AUX format
- Test on small instances


**Phase 4: Comparison Scripts**
- Run mibs on the same instances that were run with bnb
- Compare: makespan quality, runtime, nodes explored
- Analyze results

## Important Notes

### Path Adjustments Needed

The CSV path in `bilevel_model.py` assumes:
```
../gurobi_scheduling/results/sensitivity_grid/grid_20260225_143058/sensitivity_grid_cleaned.csv
```

If your paths differ, update line ~140 in bilevel_model.py.

### Instance Reproducibility

The `generate_instance.py` file uses the exact same random generation logic as `test_grid_sensitivity.py`:
- Duration range: [5, 50]
- Price range: [10, 100]  
- Budget: avg_price × n_machines × multiplier

This ensures we can recreate exact instances from just the seed.

### Expected Challenges

1. **MibS Installation**: May take 1-2 hours, especially source build
2. **MPS Format**: Learning curve for MPS/AUX file format
3. **Performance**: MibS likely much slower than your custom BnB (general vs specialized solver)

## Questions to Consider

As you review the documentation and start installation:

1. **Does the linear bilevel formulation make sense?** (See MIBS_SETUP_GUIDE.md)
2. **Which installation method fits your environment?** (Conda vs Docker vs Source)
3. **Are the project phases reasonable?** (5 phases in MIBS_PROJECT_STRUCTURE.md)
4. **Should we test on all 180 instances or a subset?** (Depends on MibS speed)

## Getting Help

- **MibS Issues**: GitHub Issues at https://github.com/coin-or/MibS
- **Our Code**: Ask me questions about any part!
- **Formulation**: See MIBS_SETUP_GUIDE.md for complete mathematical formulation

---

**Ready to start?** 

1. Read the two guide files
2. Run test_installation.py
3. Choose MibS installation method
4. Report back on progress/issues!
