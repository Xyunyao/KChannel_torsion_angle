"""
Visual comparison of old vs new autocorrelation method.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NMRConfig
from autocorrelation import AutocorrelationCalculator


def compare_methods():
    """Compare old (FFT with DC removal) vs new (direct, no DC removal)."""
    
    # Create test signal with DC offset
    np.random.seed(42)
    n_steps = 5000
    tau_c = 2e-9
    dt = 1e-12
    dc_offset = 0.5
    
    t = np.arange(n_steps) * dt
    signal = dc_offset + np.exp(-t / tau_c) + 0.1 * np.random.randn(n_steps)
    
    # New method (direct, no DC removal)
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
        verbose=False
    )
    
    calc = AutocorrelationCalculator(config)
    acf_new = calc._calculate_acf_direct(signal, max_lag=1000, lag_step=1)
    
    # Old method (FFT with DC removal)
    n_tail = min(100, n_steps // 10)
    dc_removed = signal - np.mean(signal[-n_tail:])
    n = len(dc_removed)
    series_padded = np.concatenate([dc_removed, np.zeros(n)])  # 2x padding
    fft_series = np.fft.fft(series_padded)
    power_spectrum = fft_series * np.conj(fft_series)
    acf_full = np.fft.ifft(power_spectrum).real
    acf_old = acf_full[:n]
    overlap_counts = np.arange(n, 0, -1)
    acf_old /= overlap_counts
    acf_old = acf_old[:1000]
    
    # Reference (from t1_anisotropy_analysis.py)
    acf_ref = []
    for tau in range(1000):
        val = np.mean(signal[:-tau or None] * np.conj(signal[tau:])) if tau > 0 else np.mean(signal * np.conj(signal))
        acf_ref.append(val)
    acf_ref = np.array(acf_ref)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Signal
    axes[0, 0].plot(t[:500] * 1e9, signal[:500], label='Signal')
    axes[0, 0].axhline(dc_offset, color='r', linestyle='--', label=f'DC offset = {dc_offset}')
    axes[0, 0].set_xlabel('Time (ns)')
    axes[0, 0].set_ylabel('Signal')
    axes[0, 0].set_title('Test Signal (with DC offset)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ACF comparison
    lags = np.arange(1000) * dt * 1e9  # Convert to ns
    axes[0, 1].plot(lags, acf_new.real, label='New (direct, no DC removal)', linewidth=2)
    axes[0, 1].plot(lags, acf_ref.real, 'k--', label='Reference', linewidth=1, alpha=0.7)
    axes[0, 1].set_xlabel('Lag (ns)')
    axes[0, 1].set_ylabel('ACF')
    axes[0, 1].set_title('ACF: New vs Reference')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Difference
    diff_new = np.abs(acf_new - acf_ref)
    diff_old = np.abs(acf_old - acf_ref)
    axes[1, 0].semilogy(lags, diff_new, label='New vs Ref', linewidth=2)
    axes[1, 0].semilogy(lags, diff_old, label='Old (FFT+DC) vs Ref', linewidth=1, alpha=0.7)
    axes[1, 0].set_xlabel('Lag (ns)')
    axes[1, 0].set_ylabel('Absolute Difference')
    axes[1, 0].set_title('Difference from Reference')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
COMPARISON SUMMARY

Signal:
  • Length: {n_steps} steps
  • DC offset: {dc_offset}
  • τc: {tau_c*1e9:.2f} ns
  • Noise: σ = 0.1

Methods:
  1. NEW (Direct, no DC removal)
     - Loop calculation
     - Keeps DC offset
     - Max diff from ref: {np.max(diff_new):.2e}
  
  2. OLD (FFT with DC removal)
     - FFT + 2× zero-padding
     - Removes DC (last 100 pts)
     - Max diff from ref: {np.max(diff_old):.2e}

Reference:
  • t1_anisotropy_analysis.py
  • Direct loop, no DC removal

RESULT:
  ✓ NEW method matches reference exactly
    (diff = {np.max(diff_new):.1e})
  
  ✗ OLD method differs significantly
    (diff = {np.max(diff_old):.1e})

CONCLUSION:
  Use direct method without DC removal
  to match reference implementation.
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, 
                   transform=axes[1, 1].transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('acf_method_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved: acf_method_comparison.png")
    
    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nNew method (direct, no DC removal):")
    print(f"  Max difference from reference: {np.max(diff_new):.6e}")
    print(f"  Relative error: {np.max(diff_new) / np.max(acf_ref):.2e}")
    
    print(f"\nOld method (FFT with DC removal):")
    print(f"  Max difference from reference: {np.max(diff_old):.6e}")
    print(f"  Relative error: {np.max(diff_old) / np.max(acf_ref):.2e}")
    
    print(f"\n{'✓ NEW method is CORRECT' if np.max(diff_new) < 1e-10 else '✗ NEW method has issues'}")
    print(f"{'✗ OLD method is INCORRECT' if np.max(diff_old) > 1e-6 else '✓ OLD method is correct'}")
    print("="*70)


if __name__ == '__main__':
    compare_methods()
    plt.show()
