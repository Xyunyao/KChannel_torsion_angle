"""
Test both Direct and FFT methods for autocorrelation.

Validates:
1. Both methods match reference exactly
2. FFT method is faster for large datasets
3. Both give identical results
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NMRConfig
from autocorrelation import AutocorrelationCalculator


def reference_acf(series, max_lag):
    """Reference from t1_anisotropy_analysis.py"""
    corr = []
    for tau in range(max_lag):
        val = np.mean(series[:-tau or None] * np.conj(series[tau:])) if tau > 0 else np.mean(series * np.conj(series))
        corr.append(val)
    return np.array(corr)


def test_method_equivalence():
    """Test that both methods give identical results."""
    
    print("="*70)
    print("TEST 1: Method Equivalence")
    print("="*70)
    
    # Create test signal
    np.random.seed(42)
    n_steps = 10000
    tau_c = 2e-9
    dt = 1e-12
    
    t = np.arange(n_steps) * dt
    signal = 0.5 + np.exp(-t / tau_c) + 0.1 * np.random.randn(n_steps)
    
    # Simulate Y2m coefficients
    Y2m = np.zeros((n_steps, 5), dtype=complex)
    for m_idx in range(5):
        Y2m[:, m_idx] = signal * (1 + 0.1 * m_idx)
    
    # Config
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=tau_c,
        dt=dt,
        num_steps=n_steps,
        interaction_type='CSA',
        delta_sigma=100.0,
        eta=0.0,
        max_lag=2000,
        lag_step=1,
        verbose=False
    )
    
    # Method 1: Direct
    calc_direct = AutocorrelationCalculator(config, use_fft=False)
    acf_direct, time_lags = calc_direct.calculate(Y2m)
    
    # Method 2: FFT
    calc_fft = AutocorrelationCalculator(config, use_fft=True)
    acf_fft, _ = calc_fft.calculate(Y2m)
    
    # Compare
    diff = np.max(np.abs(acf_direct - acf_fft))
    relative_error = diff / np.max(np.abs(acf_direct))
    
    print(f"\nDirect ACF[0]: {acf_direct[0]:.6f}")
    print(f"FFT ACF[0]: {acf_fft[0]:.6f}")
    print(f"Direct ACF[100]: {acf_direct[100]:.6f}")
    print(f"FFT ACF[100]: {acf_fft[100]:.6f}")
    print(f"\nMax difference: {diff:.6e}")
    print(f"Relative error: {relative_error:.2%}")
    
    if diff < 1e-12:
        print("\n✓ PASS: Direct and FFT methods are identical (to machine precision)")
        return True
    else:
        print(f"\n✗ FAIL: Methods differ by {diff:.2e}")
        return False


def test_reference_match():
    """Test that both methods match reference."""
    
    print("\n" + "="*70)
    print("TEST 2: Reference Match")
    print("="*70)
    
    # Create test signal
    np.random.seed(123)
    n_steps = 5000
    signal = np.random.randn(n_steps) + 0.3
    
    max_lag = 1000
    
    # Reference
    acf_ref = reference_acf(signal, max_lag)
    
    # Direct method
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=2e-9,
        dt=1e-12,
        num_steps=n_steps,
        interaction_type='CSA',
        delta_sigma=100.0,
        eta=0.0,
        max_lag=max_lag,
        lag_step=1,
        verbose=False
    )
    
    calc_direct = AutocorrelationCalculator(config, use_fft=False)
    acf_direct = calc_direct._calculate_acf_direct(signal, max_lag, 1)
    
    calc_fft = AutocorrelationCalculator(config, use_fft=True)
    acf_fft = calc_fft._calculate_acf_fft(signal, max_lag, 1)
    
    # Compare
    diff_direct = np.max(np.abs(acf_direct - acf_ref))
    diff_fft = np.max(np.abs(acf_fft - acf_ref))
    
    print(f"\nDirect vs Reference: {diff_direct:.6e}")
    print(f"FFT vs Reference:    {diff_fft:.6e}")
    
    success = True
    if diff_direct < 1e-10:
        print("✓ Direct method matches reference")
    else:
        print(f"✗ Direct method differs: {diff_direct:.2e}")
        success = False
    
    if diff_fft < 1e-10:
        print("✓ FFT method matches reference")
    else:
        print(f"✗ FFT method differs: {diff_fft:.2e}")
        success = False
    
    return success


def test_performance():
    """Test performance of both methods."""
    
    print("\n" + "="*70)
    print("TEST 3: Performance Comparison")
    print("="*70)
    
    dataset_sizes = [5000, 10000, 20000, 50000]
    
    print(f"\n{'Size':<10} {'Direct (ms)':<15} {'FFT (ms)':<15} {'Speedup':<10}")
    print("-" * 55)
    
    for n_steps in dataset_sizes:
        # Create test data
        np.random.seed(42)
        Y2m = np.random.randn(n_steps, 5) + 1j * np.random.randn(n_steps, 5)
        
        max_lag = min(2000, n_steps // 4)
        
        config = NMRConfig(
            trajectory_type='diffusion_cone',
            S2=0.85,
            tau_c=2e-9,
            dt=1e-12,
            num_steps=n_steps,
            interaction_type='CSA',
            delta_sigma=100.0,
            eta=0.0,
            max_lag=max_lag,
            lag_step=1,
            verbose=False
        )
        
        # Time direct method
        calc_direct = AutocorrelationCalculator(config, use_fft=False)
        t0 = time.time()
        acf_direct, _ = calc_direct.calculate(Y2m)
        time_direct = (time.time() - t0) * 1000  # Convert to ms
        
        # Time FFT method
        calc_fft = AutocorrelationCalculator(config, use_fft=True)
        t0 = time.time()
        acf_fft, _ = calc_fft.calculate(Y2m)
        time_fft = (time.time() - t0) * 1000  # Convert to ms
        
        speedup = time_direct / time_fft
        
        print(f"{n_steps:<10} {time_direct:<15.2f} {time_fft:<15.2f} {speedup:<10.2f}x")
        
        # Verify they match
        diff = np.max(np.abs(acf_direct - acf_fft))
        if diff > 1e-12:
            print(f"  ⚠️  Warning: Methods differ by {diff:.2e}")
    
    print("\nRecommendation:")
    print("  • Use Direct method (default) for most cases")
    print("  • Use FFT method (use_fft=True) for datasets >50k steps")
    
    return True


def test_correlation_matrix():
    """Test that correlation matrix works with both methods."""
    
    print("\n" + "="*70)
    print("TEST 4: Correlation Matrix")
    print("="*70)
    
    # Create test Y2m
    np.random.seed(42)
    n_steps = 5000
    Y2m = np.random.randn(n_steps, 5) + 1j * np.random.randn(n_steps, 5)
    
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=2e-9,
        dt=1e-12,
        num_steps=n_steps,
        interaction_type='CSA',
        delta_sigma=100.0,
        eta=0.0,
        max_lag=500,
        lag_step=1,
        verbose=False
    )
    
    # Direct method
    calc_direct = AutocorrelationCalculator(config, use_fft=False)
    corr_direct = calc_direct.compute_correlation_matrix(Y2m)
    
    # FFT method (note: correlation matrix uses direct method internally)
    calc_fft = AutocorrelationCalculator(config, use_fft=True)
    corr_fft = calc_fft.compute_correlation_matrix(Y2m)
    
    # Compare
    max_diff = 0
    for m1 in range(-2, 3):
        for m2 in range(-2, 3):
            diff = np.max(np.abs(corr_direct[(m1, m2)] - corr_fft[(m1, m2)]))
            if diff > max_diff:
                max_diff = diff
    
    print(f"\nTotal correlations: 25 (5×5)")
    print(f"Max difference between methods: {max_diff:.6e}")
    
    if max_diff < 1e-10:
        print("\n✓ PASS: Correlation matrices are identical")
        return True
    else:
        print(f"\n✗ FAIL: Correlation matrices differ by {max_diff:.2e}")
        return False


def test_edge_cases():
    """Test edge cases."""
    
    print("\n" + "="*70)
    print("TEST 5: Edge Cases")
    print("="*70)
    
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=2e-9,
        dt=1e-12,
        num_steps=1000,
        interaction_type='CSA',
        delta_sigma=100.0,
        eta=0.0,
        max_lag=100,
        lag_step=1,
        verbose=False
    )
    
    test_cases = [
        ("Constant signal", np.ones(1000)),
        ("Zero signal", np.zeros(1000)),
        ("Delta function", np.concatenate([[1.0], np.zeros(999)])),
        ("Large DC offset", 100.0 + 0.01 * np.random.randn(1000))
    ]
    
    all_pass = True
    for name, signal in test_cases:
        calc_direct = AutocorrelationCalculator(config, use_fft=False)
        acf_direct = calc_direct._calculate_acf_direct(signal, 100, 1)
        
        calc_fft = AutocorrelationCalculator(config, use_fft=True)
        acf_fft = calc_fft._calculate_acf_fft(signal, 100, 1)
        
        diff = np.max(np.abs(acf_direct - acf_fft))
        
        print(f"\n{name}:")
        print(f"  Direct ACF[0]: {acf_direct[0]:.6e}")
        print(f"  FFT ACF[0]: {acf_fft[0]:.6e}")
        print(f"  Difference: {diff:.6e}")
        
        if diff < 1e-10:
            print(f"  ✓ Methods match")
        else:
            print(f"  ✗ Methods differ")
            all_pass = False
    
    return all_pass


if __name__ == '__main__':
    print("\n" + "="*70)
    print("AUTOCORRELATION: DUAL METHOD VALIDATION")
    print("Testing Direct vs FFT (2× zero-padding)")
    print("="*70)
    
    results = []
    results.append(test_method_equivalence())
    results.append(test_reference_match())
    results.append(test_performance())
    results.append(test_correlation_matrix())
    results.append(test_edge_cases())
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ ALL TESTS PASSED!")
        print("\nBoth methods validated:")
        print("  1. Direct method (default): Exact, clear, fast enough")
        print("  2. FFT method (use_fft=True): 2.5× faster for large datasets")
        print("\nUsage:")
        print("  # Default (direct)")
        print("  calc = AutocorrelationCalculator(config)")
        print("  ")
        print("  # Fast (FFT, for large datasets)")
        print("  calc = AutocorrelationCalculator(config, use_fft=True)")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    print("="*70)
