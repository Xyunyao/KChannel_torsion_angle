"""
Debug cross-correlation FFT implementation.
"""

import numpy as np

# Test simple case
np.random.seed(42)
n = 1000
y1 = np.random.randn(n) + 1j * np.random.randn(n)
y2 = np.random.randn(n) + 1j * np.random.randn(n)
max_lag = 100

# Direct method (reference)
print("Direct method:")
corr_direct = []
for tau in range(max_lag):
    if tau == 0:
        val = np.mean(y1 * np.conj(y2))
    else:
        val = np.mean(y1[:-tau] * np.conj(y2[tau:]))
    corr_direct.append(val)
corr_direct = np.array(corr_direct)

print(f"  C[0] = {corr_direct[0]:.6e}")
print(f"  C[10] = {corr_direct[10]:.6e}")
print(f"  C[50] = {corr_direct[50]:.6e}")

# FFT method (incorrect?)
print("\nFFT method (current implementation):")
n_padded = 2 * n
s1_padded = np.concatenate([y1, np.zeros(n_padded - n)])
s2_padded = np.concatenate([y2, np.zeros(n_padded - n)])

fft_s1 = np.fft.fft(s1_padded)
fft_s2 = np.fft.fft(s2_padded)
cross_spectrum = fft_s1 * np.conj(fft_s2)

cross_corr_full = np.fft.ifft(cross_spectrum)
cross_corr = cross_corr_full[:n]

# Normalize
overlap_counts = np.arange(n, 0, -1)
cross_corr_fft = cross_corr / overlap_counts
cross_corr_fft = cross_corr_fft[:max_lag]

print(f"  C[0] = {cross_corr_fft[0]:.6e}")
print(f"  C[10] = {cross_corr_fft[10]:.6e}")
print(f"  C[50] = {cross_corr_fft[50]:.6e}")

# Compare
diff = np.max(np.abs(corr_direct - cross_corr_fft))
print(f"\nDifference: {diff:.6e}")

if diff > 1e-10:
    print("✗ FFT method has error!")
    print("\nLet me check the formula...")
    
    # The issue: For cross-correlation C_{1,2}(τ) = ⟨x₁(t) × x₂*(t+τ)⟩
    # This is NOT the same as standard FFT cross-correlation!
    
    # Standard cross-correlation: C(τ) = ⟨x₁(t+τ) × x₂*(t)⟩
    # Our definition: C(τ) = ⟨x₁(t) × x₂*(t+τ)⟩
    
    # These are related by: our_C(τ) = standard_C(-τ)
    
    print("\nTrying reversed indices...")
    cross_corr_fft_rev = cross_corr_fft
    
    # Actually, let me recalculate with proper understanding
    # C(τ) = ⟨x₁(t) × x₂*(t+τ)⟩
    # = ⟨x₁(t-τ) × x₂*(t)⟩  (shift in x₁)
    # = IFFT(FFT(x₁) × conj(FFT(x₂)))  but with shift
    
    print("\nThe problem: FFT gives standard correlation, but we need time-reversed")
    print("Solution: Use np.correlate or manually reverse/conjugate")
else:
    print("✓ FFT method is correct!")
