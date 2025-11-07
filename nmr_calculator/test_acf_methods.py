"""
Test to compare ACF calculation methods:
1. FFT-based method with zero-fill (_calculate_acf_fft)
2. Direct method in compute_correlation_matrix (reference)
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NMRConfig
from autocorrelation import AutocorrelationCalculator


def test_acf_comparison():
    """Compare FFT method vs direct method."""
    
    print("="*70)
    print("TEST: Comparing ACF Calculation Methods")
    print("="*70)
    
    # Create test signal
    np.random.seed(42)
    n_steps = 5000
    tau_c = 2e-9
    dt = 1e-12
    
    # Exponential decay signal
    t = np.arange(n_steps) * dt
    signal = np.exp(-t / tau_c) + 0.1 * np.random.randn(n_steps)
    
    # Create config
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=tau_c,
        dt=dt,
        num_steps=n_steps,
        interaction_type='CSA',
        delta_sigma=100.0,
        eta=0.0,
        max_lag=1000,
        lag_step=1,
        zero_fill_factor=1,  # No zero-fill for comparison
        verbose=False
    )
    
    calc = AutocorrelationCalculator(config)
    
    # Method 1: FFT with zero-fill
    print("\nMethod 1: FFT-based (_calculate_acf_fft)")
    acf_fft_zf1 = calc._calculate_acf_fft(signal, max_lag=1000, lag_step=1, zero_fill_factor=1)
    acf_fft_zf2 = calc._calculate_acf_fft(signal, max_lag=1000, lag_step=1, zero_fill_factor=2)
    print(f"  Zero-fill=1: ACF[0]={acf_fft_zf1[0]:.6e}, ACF[100]={acf_fft_zf1[100]:.6e}")
    print(f"  Zero-fill=2: ACF[0]={acf_fft_zf2[0]:.6e}, ACF[100]={acf_fft_zf2[100]:.6e}")
    
    # Method 2: Direct method (like in compute_correlation_matrix)
    print("\nMethod 2: Direct method (reference implementation)")
    
    # Remove DC offset (last 100 points)
    n_tail = min(100, n_steps // 10)
    dc_offset = np.mean(signal[-n_tail:])
    y = signal - dc_offset
    
    # Direct calculation
    max_lag = 1000
    corr_direct = []
    for tau in range(0, max_lag, 1):
        if tau == 0:
            val = np.mean(y * np.conj(y))
        else:
            val = np.mean(y[:-tau] * np.conj(y[tau:]))
        corr_direct.append(val)
    corr_direct = np.array(corr_direct)
    
    # Normalize
    corr_direct_norm = corr_direct / corr_direct[0]
    
    print(f"  Direct: ACF[0]={corr_direct[0]:.6e}, ACF[100]={corr_direct[100]:.6e}")
    print(f"  Direct (normalized): ACF[0]={corr_direct_norm[0]:.6f}, ACF[100]={corr_direct_norm[100]:.6f}")
    
    # Method 3: Reference implementation exactly as in t1_anisotropy_analysis.py
    print("\nMethod 3: Reference (t1_anisotropy_analysis.py style)")
    
    # NO DC offset removal in reference!
    corr_ref = []
    for tau in range(0, max_lag, 1):
        val = np.mean(signal[:-tau or None] * np.conj(signal[tau:])) if tau > 0 else np.mean(signal * np.conj(signal))
        corr_ref.append(val)
    corr_ref = np.array(corr_ref)
    
    print(f"  Reference: ACF[0]={corr_ref[0]:.6e}, ACF[100]={corr_ref[100]:.6e}")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    # Compare FFT (no zero-fill) vs Direct
    diff_fft_direct = np.max(np.abs(acf_fft_zf1 - corr_direct_norm))
    print(f"\n1. FFT (zf=1) vs Direct (both with DC removal):")
    print(f"   Max difference: {diff_fft_direct:.6e}")
    print(f"   Relative error: {diff_fft_direct / np.max(np.abs(corr_direct_norm)):.2%}")
    
    # Compare Direct vs Reference
    diff_direct_ref = np.max(np.abs(corr_direct - corr_ref))
    print(f"\n2. Direct (DC removed) vs Reference (no DC removal):")
    print(f"   Max difference: {diff_direct_ref:.6e}")
    print(f"   DC offset value: {dc_offset:.6e}")
    print(f"   → They differ because reference does NOT remove DC offset!")
    
    # The key finding
    print("\n" + "="*70)
    print("KEY FINDING:")
    print("="*70)
    print("\nThe reference implementation in t1_anisotropy_analysis.py:")
    print("  • Does NOT remove DC offset")
    print("  • Uses direct loop calculation:")
    print("    val = np.mean(y1[:-tau or None] * np.conj(y2[tau:]))")
    print("\nThe FFT method with zero-fill:")
    print("  • DOES remove DC offset (last 100 points)")
    print("  • Uses FFT + zero-padding")
    print("\n⚠️  These give DIFFERENT results!")
    
    # Test what happens without DC removal in FFT
    print("\n" + "="*70)
    print("SOLUTION: FFT without DC removal")
    print("="*70)
    
    # FFT without DC removal (to match reference)
    n = len(signal)
    series_padded = np.concatenate([signal, np.zeros(n)])  # 2x padding
    fft_series = np.fft.fft(series_padded)
    power_spectrum = fft_series * np.conj(fft_series)
    acf_full = np.fft.ifft(power_spectrum).real
    acf_fft_no_dc = acf_full[:n]
    overlap_counts = np.arange(n, 0, -1)
    acf_fft_no_dc /= overlap_counts
    acf_fft_no_dc = acf_fft_no_dc[:max_lag]
    
    diff_fft_ref = np.max(np.abs(acf_fft_no_dc - corr_ref))
    print(f"\nFFT (no DC removal) vs Reference:")
    print(f"  Max difference: {diff_fft_ref:.6e}")
    print(f"  → Should be small! (FFT numerical error only)")
    
    if diff_fft_ref < 1e-6:
        print("\n✓ SUCCESS: FFT without DC removal matches reference!")
    else:
        print(f"\n✗ FAIL: Difference too large: {diff_fft_ref:.6e}")
    
    return diff_fft_ref < 1e-6


if __name__ == '__main__':
    success = test_acf_comparison()
    print("\n" + "="*70)
    if success:
        print("CONCLUSION: Remove DC offset step from FFT method")
        print("            Remove zero-fill from ACF calculation")
        print("            Use direct loop method from reference")
    else:
        print("CONCLUSION: Need further investigation")
    print("="*70)
