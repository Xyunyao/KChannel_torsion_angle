"""
Test: Does FFT method (without DC removal) match reference?

This tests whether the issue was:
1. DC offset removal (wrong)
2. FFT method itself (wrong)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def reference_acf(series, max_lag):
    """Reference from t1_anisotropy_analysis.py"""
    corr = []
    for tau in range(max_lag):
        val = np.mean(series[:-tau or None] * np.conj(series[tau:])) if tau > 0 else np.mean(series * np.conj(series))
        corr.append(val)
    return np.array(corr)


def fft_acf_no_dc_removal(series, max_lag):
    """FFT method WITHOUT DC removal"""
    n = len(series)
    
    # NO DC removal - use signal as-is
    
    # Power spectrum
    fft_series = np.fft.fft(series)
    power_spectrum = fft_series * np.conj(fft_series)
    
    # Inverse FFT gives autocorrelation
    acf_full = np.fft.ifft(power_spectrum).real
    
    # Normalize by number of overlapping points
    overlap_counts = np.arange(n, 0, -1)
    acf = acf_full / overlap_counts
    
    # Take first max_lag points
    return acf[:max_lag]


def fft_acf_with_zero_pad_no_dc(series, max_lag, zero_fill=2):
    """FFT method with zero-padding but NO DC removal"""
    n = len(series)
    
    # NO DC removal
    
    # Zero-pad
    n_padded = zero_fill * n
    series_padded = np.concatenate([series, np.zeros(n_padded - n)])
    
    # Power spectrum on padded signal
    fft_series = np.fft.fft(series_padded)
    power_spectrum = fft_series * np.conj(fft_series)
    
    # Inverse FFT
    acf_full = np.fft.ifft(power_spectrum).real
    
    # Take first n points (original length)
    acf = acf_full[:n]
    
    # Normalize by overlap counts
    overlap_counts = np.arange(n, 0, -1)
    acf /= overlap_counts
    
    return acf[:max_lag]


def direct_method(series, max_lag):
    """Direct loop method (current implementation)"""
    corr = []
    for tau in range(max_lag):
        if tau == 0:
            val = np.mean(series * np.conj(series))
        else:
            val = np.mean(series[:-tau] * np.conj(series[tau:]))
        corr.append(val)
    return np.array(corr)


def test_all_methods():
    """Compare all ACF methods"""
    
    print("="*70)
    print("TEST: FFT (no DC removal) vs Direct vs Reference")
    print("="*70)
    
    # Test signals
    np.random.seed(42)
    n_steps = 5000
    max_lag = 1000
    
    # Signal 1: Exponential decay with DC offset
    tau_c = 2e-9
    dt = 1e-12
    dc_offset = 0.5
    t = np.arange(n_steps) * dt
    signal1 = dc_offset + np.exp(-t / tau_c) + 0.1 * np.random.randn(n_steps)
    
    # Signal 2: Pure noise (no DC)
    signal2 = np.random.randn(n_steps)
    
    # Signal 3: Pure exponential (no DC, no noise)
    signal3 = np.exp(-t / tau_c)
    
    signals = [
        ("Exp decay + DC + noise", signal1),
        ("Pure noise", signal2),
        ("Pure exp decay", signal3)
    ]
    
    for name, signal in signals:
        print(f"\n{'='*70}")
        print(f"Signal: {name}")
        print(f"{'='*70}")
        print(f"  Mean: {np.mean(signal):.6f}")
        print(f"  Std: {np.std(signal):.6f}")
        
        # Calculate ACF with all methods
        acf_ref = reference_acf(signal, max_lag)
        acf_direct = direct_method(signal, max_lag)
        acf_fft_no_dc = fft_acf_no_dc_removal(signal, max_lag)
        acf_fft_zf_no_dc = fft_acf_with_zero_pad_no_dc(signal, max_lag, zero_fill=2)
        
        # Compare
        diff_direct = np.max(np.abs(acf_direct - acf_ref))
        diff_fft = np.max(np.abs(acf_fft_no_dc - acf_ref))
        diff_fft_zf = np.max(np.abs(acf_fft_zf_no_dc - acf_ref))
        
        print(f"\n  Method Comparison:")
        print(f"    Direct vs Reference:          {diff_direct:.6e}")
        print(f"    FFT (no DC) vs Reference:     {diff_fft:.6e}")
        print(f"    FFT+ZF (no DC) vs Reference:  {diff_fft_zf:.6e}")
        
        # Sample values
        print(f"\n  ACF[0] values:")
        print(f"    Reference:      {acf_ref[0]:.6e}")
        print(f"    Direct:         {acf_direct[0]:.6e}")
        print(f"    FFT (no DC):    {acf_fft_no_dc[0]:.6e}")
        print(f"    FFT+ZF (no DC): {acf_fft_zf_no_dc[0]:.6e}")
        
        print(f"\n  ACF[100] values:")
        print(f"    Reference:      {acf_ref[100]:.6e}")
        print(f"    Direct:         {acf_direct[100]:.6e}")
        print(f"    FFT (no DC):    {acf_fft_no_dc[100]:.6e}")
        print(f"    FFT+ZF (no DC): {acf_fft_zf_no_dc[100]:.6e}")
        
        # Pass/fail
        tol = 1e-10
        if diff_direct < tol:
            print(f"\n  ✓ Direct method matches reference")
        else:
            print(f"\n  ✗ Direct method differs: {diff_direct:.2e}")
        
        if diff_fft < tol:
            print(f"  ✓ FFT (no DC) matches reference")
        else:
            print(f"  ✗ FFT (no DC) differs: {diff_fft:.2e}")
        
        if diff_fft_zf < tol:
            print(f"  ✓ FFT+ZF (no DC) matches reference")
        else:
            print(f"  ✗ FFT+ZF (no DC) differs: {diff_fft_zf:.2e}")


def test_fft_edge_cases():
    """Test FFT method edge cases"""
    
    print("\n" + "="*70)
    print("EDGE CASES: FFT Normalization")
    print("="*70)
    
    # Simple test: constant signal
    print("\nTest 1: Constant signal")
    signal = np.ones(1000)
    acf_ref = reference_acf(signal, 100)
    acf_fft = fft_acf_no_dc_removal(signal, 100)
    
    print(f"  Reference ACF[0]: {acf_ref[0]:.6f}")
    print(f"  FFT ACF[0]: {acf_fft[0]:.6f}")
    print(f"  Difference: {np.max(np.abs(acf_ref - acf_fft)):.6e}")
    
    # Delta function
    print("\nTest 2: Delta function")
    signal = np.zeros(1000)
    signal[0] = 1.0
    acf_ref = reference_acf(signal, 100)
    acf_fft = fft_acf_no_dc_removal(signal, 100)
    
    print(f"  Reference ACF[0]: {acf_ref[0]:.6e}")
    print(f"  FFT ACF[0]: {acf_fft[0]:.6e}")
    print(f"  Difference: {np.max(np.abs(acf_ref - acf_fft)):.6e}")
    
    # Sine wave
    print("\nTest 3: Sine wave")
    t = np.linspace(0, 10*np.pi, 1000)
    signal = np.sin(t)
    acf_ref = reference_acf(signal, 100)
    acf_fft = fft_acf_no_dc_removal(signal, 100)
    
    print(f"  Reference ACF[0]: {acf_ref[0]:.6f}")
    print(f"  FFT ACF[0]: {acf_fft[0]:.6f}")
    print(f"  Difference: {np.max(np.abs(acf_ref - acf_fft)):.6e}")


if __name__ == '__main__':
    test_all_methods()
    test_fft_edge_cases()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The key question: Does FFT (without DC removal) match reference?

If YES → Problem was DC offset removal
If NO  → Problem is FFT method itself

Results will show which method(s) are correct.
    """)
    print("="*70)
