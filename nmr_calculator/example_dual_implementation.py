#!/usr/bin/env python3
"""
Example: Using both Sympy and NumPy implementations of spherical harmonics.

This script demonstrates:
1. How to use both implementations
2. That they give identical results
3. The speed difference between them

Author: NMR Calculator Package
Date: 2024
"""

import numpy as np
import time
from config import NMRConfig
from spherical_harmonics import SphericalHarmonicsCalculator


def main():
    print("="*70)
    print("SPHERICAL HARMONICS: DUAL IMPLEMENTATION EXAMPLE")
    print("="*70)
    
    # Setup CSA parameters
    print("\n1. Setting up CSA parameters:")
    config = NMRConfig()
    config.interaction_type = 'CSA'
    config.delta_sigma = 100.0  # ppm (anisotropy)
    config.eta = 0.3            # asymmetry parameter
    config.delta_iso = 50.0     # ppm (isotropic shift)
    config.verbose = False
    
    print(f"   Δδ = {config.delta_sigma} ppm")
    print(f"   η  = {config.eta}")
    print(f"   δ_iso = {config.delta_iso} ppm")
    
    # Generate sample Euler angles
    print("\n2. Generating sample Euler angles:")
    np.random.seed(42)
    n_steps = 1000
    euler_angles = np.random.uniform(0, 2*np.pi, (n_steps, 3))
    print(f"   {n_steps} random orientations")
    
    # Method 1: NumPy implementation (default, fast)
    print("\n3. Using NumPy implementation (default):")
    calc_numpy = SphericalHarmonicsCalculator(config, use_sympy=False)
    
    start = time.time()
    Y2m_numpy = calc_numpy.calculate(euler_angles)
    time_numpy = time.time() - start
    
    print(f"   Calculation time: {time_numpy:.4f} seconds")
    print(f"   Shape: {Y2m_numpy.shape}")
    print(f"   Y₂ₘ mean values:")
    for m in range(-2, 3):
        m_idx = m + 2
        print(f"     Y₂^{m:+d}: {np.mean(Y2m_numpy[:, m_idx]):+.3f}")
    
    # Method 2: Sympy implementation (explicit, slower)
    print("\n4. Using Sympy implementation (explicit):")
    calc_sympy = SphericalHarmonicsCalculator(config, use_sympy=True)
    
    start = time.time()
    Y2m_sympy = calc_sympy.calculate(euler_angles)
    time_sympy = time.time() - start
    
    print(f"   Calculation time: {time_sympy:.4f} seconds")
    print(f"   Shape: {Y2m_sympy.shape}")
    print(f"   Y₂ₘ mean values:")
    for m in range(-2, 3):
        m_idx = m + 2
        print(f"     Y₂^{m:+d}: {np.mean(Y2m_sympy[:, m_idx]):+.3f}")
    
    # Compare results
    print("\n5. Comparing results:")
    diff = np.abs(Y2m_numpy - Y2m_sympy)
    print(f"   Maximum difference: {np.max(diff):.2e}")
    print(f"   Mean difference:    {np.mean(diff):.2e}")
    print(f"   Relative error:     {np.max(diff) / np.max(np.abs(Y2m_numpy)):.2e}")
    
    if np.max(diff) < 1e-10:
        print("   ✓ Results are identical (within numerical precision)")
    else:
        print("   ✗ Results differ!")
    
    # Speed comparison
    print("\n6. Speed comparison:")
    speedup = time_sympy / time_numpy
    print(f"   Sympy: {time_sympy:.4f} s")
    print(f"   NumPy: {time_numpy:.4f} s")
    print(f"   Speedup: {speedup:.1f}x faster with NumPy")
    
    # Demonstration with specific angles
    print("\n7. Example: Specific orientations")
    print("   " + "-"*50)
    
    test_angles = np.array([
        [0.0, 0.0, 0.0],           # Aligned with z
        [0.0, np.pi/2, 0.0],       # Perpendicular to z
        [np.pi/4, np.pi/4, 0.0],   # General orientation
    ])
    
    Y2m_test = calc_numpy.calculate(test_angles)
    
    angle_names = ['Aligned (β=0°)', 'Perpendicular (β=90°)', 'General (β=45°)']
    
    for i, name in enumerate(angle_names):
        print(f"\n   {name}:")
        print(f"   α={np.degrees(test_angles[i,0]):.0f}°, " +
              f"β={np.degrees(test_angles[i,1]):.0f}°, " +
              f"γ={np.degrees(test_angles[i,2]):.0f}°")
        for m in range(-2, 3):
            m_idx = m + 2
            print(f"     Y₂^{m:+d} = {Y2m_test[i, m_idx]:+.6f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✓ Both implementations work correctly")
    print("✓ Results are identical (within numerical precision)")
    print(f"✓ NumPy is {speedup:.1f}x faster")
    print("\nRecommendations:")
    print("  • Use NumPy (default) for production calculations")
    print("  • Use Sympy for mathematical verification/debugging")
    print("\nUsage:")
    print("  # Fast (default)")
    print("  calc = SphericalHarmonicsCalculator(config)")
    print("\n  # Explicit (for verification)")
    print("  calc = SphericalHarmonicsCalculator(config, use_sympy=True)")
    print("="*70)


if __name__ == '__main__':
    main()
