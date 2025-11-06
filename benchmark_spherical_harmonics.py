#!/usr/bin/env python3
"""
Benchmark script comparing sympy vs NumPy implementations.

Compares:
1. Accuracy: Both methods should give identical results
2. Speed: NumPy should be significantly faster

Usage:
    python benchmark_spherical_harmonics.py
"""

import numpy as np
import time
from config import NMRConfig
from spherical_harmonics import SphericalHarmonicsCalculator


def benchmark_accuracy():
    """Test that both implementations give identical results."""
    print("\n" + "="*70)
    print("ACCURACY TEST: Sympy vs NumPy")
    print("="*70)
    
    config = NMRConfig()
    config.interaction_type = 'CSA'
    config.delta_sigma = 100.0
    config.eta = 0.3
    config.delta_iso = 50.0
    config.verbose = False
    
    # Create both calculators
    calc_sympy = SphericalHarmonicsCalculator(config, use_sympy=True)
    calc_numpy = SphericalHarmonicsCalculator(config, use_sympy=False)
    
    # Test with multiple orientations
    np.random.seed(42)
    n_test = 100
    euler_angles = np.random.uniform(0, 2*np.pi, (n_test, 3))
    
    print(f"\nTesting with {n_test} random orientations...")
    
    Y2m_sympy = calc_sympy._calculate_CSA(euler_angles)
    Y2m_numpy = calc_numpy._calculate_CSA(euler_angles)
    
    # Compare results
    diff = np.abs(Y2m_sympy - Y2m_numpy)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nAccuracy comparison:")
    print(f"  Maximum difference: {max_diff:.2e}")
    print(f"  Mean difference:    {mean_diff:.2e}")
    print(f"  Relative error:     {max_diff / np.max(np.abs(Y2m_sympy)):.2e}")
    
    if max_diff < 1e-10:
        print("\n✓ PASSED: Both methods give identical results (within numerical precision)")
        return True
    else:
        print(f"\n✗ FAILED: Difference too large!")
        return False


def benchmark_speed():
    """Benchmark computation speed of both implementations."""
    print("\n" + "="*70)
    print("SPEED BENCHMARK: Sympy vs NumPy")
    print("="*70)
    
    config = NMRConfig()
    config.interaction_type = 'CSA'
    config.delta_sigma = 100.0
    config.eta = 0.4
    config.delta_iso = 50.0
    config.verbose = False
    
    # Test different sizes
    sizes = [100, 1000, 10000]
    
    results = []
    
    for n_steps in sizes:
        print(f"\nTesting with {n_steps:,} time steps:")
        
        # Generate random Euler angles
        np.random.seed(42)
        euler_angles = np.random.uniform(0, 2*np.pi, (n_steps, 3))
        
        # Benchmark sympy
        calc_sympy = SphericalHarmonicsCalculator(config, use_sympy=True)
        start = time.time()
        Y2m_sympy = calc_sympy._calculate_CSA(euler_angles)
        time_sympy = time.time() - start
        
        # Benchmark numpy
        calc_numpy = SphericalHarmonicsCalculator(config, use_sympy=False)
        start = time.time()
        Y2m_numpy = calc_numpy._calculate_CSA(euler_angles)
        time_numpy = time.time() - start
        
        speedup = time_sympy / time_numpy
        
        print(f"  Sympy:  {time_sympy:.4f} s")
        print(f"  NumPy:  {time_numpy:.4f} s")
        print(f"  Speedup: {speedup:.1f}x faster with NumPy")
        
        results.append({
            'n_steps': n_steps,
            'time_sympy': time_sympy,
            'time_numpy': time_numpy,
            'speedup': speedup
        })
    
    print("\n" + "-"*70)
    print("Summary:")
    print("-"*70)
    print(f"{'N Steps':<15} {'Sympy (s)':<15} {'NumPy (s)':<15} {'Speedup':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['n_steps']:<15,} {r['time_sympy']:<15.4f} {r['time_numpy']:<15.4f} {r['speedup']:<15.1f}x")
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    print("-"*70)
    print(f"Average speedup: {avg_speedup:.1f}x")
    
    return results


def test_both_methods_comprehensive():
    """Comprehensive test for various CSA parameters."""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST: Various CSA Parameters")
    print("="*70)
    
    test_cases = [
        {'name': 'Axial (η=0)', 'delta_sigma': 100.0, 'eta': 0.0, 'delta_iso': 50.0},
        {'name': 'Slightly non-axial (η=0.1)', 'delta_sigma': 100.0, 'eta': 0.1, 'delta_iso': 50.0},
        {'name': 'Moderate asymmetry (η=0.5)', 'delta_sigma': 100.0, 'eta': 0.5, 'delta_iso': 50.0},
        {'name': 'Maximum asymmetry (η=1.0)', 'delta_sigma': 100.0, 'eta': 1.0, 'delta_iso': 50.0},
        {'name': 'Large anisotropy', 'delta_sigma': 200.0, 'eta': 0.3, 'delta_iso': 100.0},
    ]
    
    np.random.seed(42)
    euler_angles = np.random.uniform(0, 2*np.pi, (50, 3))
    
    all_passed = True
    
    for case in test_cases:
        config = NMRConfig()
        config.interaction_type = 'CSA'
        config.delta_sigma = case['delta_sigma']
        config.eta = case['eta']
        config.delta_iso = case['delta_iso']
        config.verbose = False
        
        calc_sympy = SphericalHarmonicsCalculator(config, use_sympy=True)
        calc_numpy = SphericalHarmonicsCalculator(config, use_sympy=False)
        
        Y2m_sympy = calc_sympy._calculate_CSA(euler_angles)
        Y2m_numpy = calc_numpy._calculate_CSA(euler_angles)
        
        max_diff = np.max(np.abs(Y2m_sympy - Y2m_numpy))
        
        status = "✓" if max_diff < 1e-10 else "✗"
        print(f"\n{status} {case['name']:<30} max_diff={max_diff:.2e}")
        
        if max_diff >= 1e-10:
            all_passed = False
    
    if all_passed:
        print("\n✓ All test cases passed!")
    else:
        print("\n✗ Some test cases failed!")
    
    return all_passed


def main():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("SPHERICAL HARMONICS IMPLEMENTATION BENCHMARK")
    print("="*70)
    print("\nComparing two implementations:")
    print("  1. Sympy-based: Symbolic Wigner D-matrix (mathematically explicit)")
    print("  2. NumPy-based: Optimized numerical calculation (faster)")
    
    try:
        # Test accuracy
        accuracy_passed = benchmark_accuracy()
        
        if not accuracy_passed:
            print("\n✗ Accuracy test failed! Aborting speed benchmark.")
            return 1
        
        # Test comprehensive cases
        comprehensive_passed = test_both_methods_comprehensive()
        
        if not comprehensive_passed:
            print("\n✗ Comprehensive test failed!")
            return 1
        
        # Benchmark speed
        benchmark_speed()
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70)
        print("\nConclusions:")
        print("  • Both implementations give identical results")
        print("  • NumPy version is significantly faster (recommended for production)")
        print("  • Sympy version is useful for mathematical verification")
        print("\nRecommendation:")
        print("  Use use_sympy=False (default) for speed")
        print("  Use use_sympy=True for debugging/verification")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
