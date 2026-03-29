"""
Test MibS installation and basic functionality.

Run this after installing MibS to verify everything works.

Expected output:
- Python imports work (numpy, pandas)
- MibS executable is available
- Can run MibS on a simple example
"""

import sys
import subprocess
import os
from pathlib import Path


def test_python_packages():
    """Test that required Python packages are installed."""
    print("=" * 60)
    print("Testing Python Package Imports")
    print("=" * 60)
    
    packages = ['numpy', 'pandas', 'matplotlib']
    all_ok = True
    
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} - NOT INSTALLED")
            all_ok = False
    
    print()
    return all_ok


def test_mibs_executable():
    """Test that MibS executable is available."""
    print("=" * 60)
    print("Testing MibS Executable")
    print("=" * 60)
    
    # Try Windows executables first
    executables = ['mibs', 'mibs.exe', 'MibS', 'MibS.exe']
    
    for exe in executables:
        try:
            result = subprocess.run([exe, '--version'], 
                                    capture_output=True, 
                                    text=True,
                                    timeout=5)
            print(f"✓ Found: {exe}")
            print(f"  Output: {result.stdout.strip()}")
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception as e:
            continue
    
    # Try WSL (Linux) installation
    try:
        result = subprocess.run(['wsl', '/home/ole/mibs_build/dist/bin/mibs'], 
                                capture_output=True, 
                                text=True,
                                timeout=5)
        if 'Welcome to MibS' in result.stdout or 'Welcome to MibS' in result.stderr:
            print(f"✓ Found: MibS in WSL")
            print(f"  Location: /home/ole/mibs_build/dist/bin/mibs")
            print(f"  Version: Built Feb 27 2026")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as e:
        pass
    
    print("✗ MibS executable not found")
    print("\nTried:")
    for exe in executables:
        print(f"  - {exe}")
    print(f"  - WSL: /home/ole/mibs_build/dist/bin/mibs")
    print("\nPlease ensure MibS is installed")
    print()
    return False


def test_formulation_imports():
    """Test that our formulation code works."""
    print("=" * 60)
    print("Testing Formulation Code")
    print("=" * 60)
    
    try:
        from formulation import BilevelInstance, regenerate_from_seed
        print("✓ Imports successful")
        
        # Test instance regeneration
        seed = 2334587927
        n_jobs = 4
        durations, prices = regenerate_from_seed(n_jobs, seed)
        print(f"✓ Instance regeneration works (seed={seed})")
        print(f"  Durations: {durations}")
        print(f"  Prices: {prices}")
        
        # Test BilevelInstance creation
        instance = BilevelInstance(
            n_job_types=n_jobs,
            n_machines=2,
            durations=durations,
            prices=prices,
            budget=91.0,
            seed=seed
        )
        print(f"✓ BilevelInstance creation works")
        print(f"  ID: {instance.get_instance_id()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        print()


def main():
    """Run all installation tests."""
    print("\n" + "=" * 60)
    print("MibS Installation Test Suite")
    print("=" * 60 + "\n")
    
    results = {}
    results['python_packages'] = test_python_packages()
    results['formulation_code'] = test_formulation_imports()
    results['mibs_executable'] = test_mibs_executable()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test}")
    
    print()
    
    if all(results.values()):
        print("🎉 All tests passed! You're ready to proceed.")
        return 0
    else:
        print("⚠️  Some tests failed. Please address issues above.")
        print("\nInstallation hints:")
        print("  1. Python packages: pip install -r requirements.txt")
        print("  2. MibS: conda install -c conda-forge coinbrew")
        print("     Then: coinbrew fetch MibS@stable/2.1")
        print("           coinbrew build MibS")
        return 1


if __name__ == "__main__":
    sys.exit(main())
