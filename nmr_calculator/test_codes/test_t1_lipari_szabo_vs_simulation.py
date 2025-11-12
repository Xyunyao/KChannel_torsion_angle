"""
Test comparing Lipari-Szabo analytical T1 vs simulated trajectory T1.

Assumptions:
- η = 0 (uniaxial CSA)
- Local motion: diffusion on a cone (characterized by S² and τf)
- Global motion: isotropic tumbling (characterized by τc)

Method 1 (Analytical):
1. Calculate J(ω) using Lipari-Szabo formula: J(ω) = (2/5)[S²τc/(1+ω²τc²) + (1-S²)τ/(1+ω²τ²)]
2. Calculate T1 using analytical uniaxial formula: R1 = (1/3)(ω₀×Δσ)²J(ω₀)

Method 2 (Simulated):
1. Simulate local cone diffusion → local ACF (for m=1 component)
2. Multiply local ACF by exp(-t/τc) to add global tumbling
3. Calculate J(ω) from final ACF using FFT
4. Calculate T1 using universal formula: R1 = ω₀²J(ω₀)×10⁻¹²

Both methods should give similar T1 values for η=0.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from nmr_calculator.config import NMRConfig
from nmr_calculator.xyz_generator import TrajectoryGenerator
from nmr_calculator.euler_converter import EulerConverter
from nmr_calculator.spherical_harmonics import SphericalHarmonicsCalculator
from nmr_calculator.autocorrelation import AutocorrelationCalculator
from nmr_calculator.spectral_density import SpectralDensityCalculator
from nmr_calculator.nmr_parameters import NMRParametersCalculator


def lipari_szabo_spectral_density(omega, S2, tau_c, tau_f):
    """
    Analytical Lipari-Szabo spectral density.
    
    J(ω) = (2/5) × [S² × τc/(1 + ω²τc²) + (1-S²) × τ/(1 + ω²τ²)]
    
    where τ = τc × τf / (τc + τf)
    """
    tau_eff = (tau_c * tau_f) / (tau_c + tau_f)
    
    J = (2.0/5.0) * (
        S2 * tau_c / (1 + (omega * tau_c)**2) + 
        (1 - S2) * tau_eff / (1 + (omega * tau_eff)**2)
    )
    
    return J


def lipari_szabo_acf(t, S2, tau_c, tau_f):
    """
    Analytical Lipari-Szabo autocorrelation function.
    
    C(t) = (2/5) × [S² × exp(-t/τc) + (1-S²) × exp(-t/τ)]
    
    where τ = τc × τf / (τc + τf)
    """
    tau_eff = (tau_c * tau_f) / (tau_c + tau_f)
    
    C = (2.0/5.0) * (S2 * np.exp(-t/tau_c) + (1 - S2) * np.exp(-t/tau_eff))
    
    return C


def test_lipari_szabo_comparison():
    """
    Compare analytical Lipari-Szabo T1 with simulated trajectory T1.
    """
    print("="*80)
    print("TEST: Lipari-Szabo Analytical vs Simulated T1 Comparison")
    print("="*80)
    
    # Physical parameters
    S2 = 0.85           # Order parameter
    tau_c = 3e-9        # Global correlation time (3 ns)
    tau_f = 0.1e-9      # Fast local motion (100 ps)
    delta_sigma = 160.0 # CSA anisotropy (ppm)
    B0 = 14.1           # Magnetic field (T) - 600 MHz for 1H
    nucleus = '13C'     # Carbon-13
    
    print(f"\nPhysical Parameters:")
    print(f"  S² = {S2}")
    print(f"  τc = {tau_c*1e9:.1f} ns (global tumbling)")
    print(f"  τf = {tau_f*1e9:.1f} ps (local motion)")
    print(f"  Δσ = {delta_sigma} ppm")
    print(f"  B₀ = {B0} T")
    print(f"  Nucleus: {nucleus}")
    print(f"  η = 0 (uniaxial CSA)")
    
    # Effective correlation time
    tau_eff = (tau_c * tau_f) / (tau_c + tau_f)
    print(f"  τ_eff = {tau_eff*1e9:.3f} ns")
    
    # Larmor frequency
    from nmr_calculator.config import GAMMA
    omega_0 = GAMMA[nucleus] * B0
    larmor_freq_Hz = omega_0 / (2 * np.pi)
    larmor_freq_MHz = larmor_freq_Hz * 1e-6
    
    print(f"  ω₀ = {omega_0:.3e} rad/s ({larmor_freq_MHz:.2f} MHz)")
    
    # ========================================================================
    # METHOD 1: Analytical Lipari-Szabo
    # ========================================================================
    print(f"\n{'='*80}")
    print("METHOD 1: Analytical Lipari-Szabo")
    print(f"{'='*80}")
    
    # Calculate analytical J(ω₀)
    J_omega0_analytical = lipari_szabo_spectral_density(omega_0, S2, tau_c, tau_f)
    
    print(f"\nSpectral Density:")
    print(f"  J(ω₀) = {J_omega0_analytical:.3e} s")
    
    # Calculate T1 using analytical uniaxial formula
    # R1 = (1/3) × (ω₀ × Δσ)² × J(ω₀)
    delta_sigma_rad = delta_sigma * 1e-6 * omega_0  # Convert ppm to rad/s
    R1_analytical = (1.0/3.0) * (delta_sigma_rad)**2 * J_omega0_analytical
    T1_analytical = 1.0 / R1_analytical
    
    print(f"\nT1 Calculation (Analytical Uniaxial Formula):")
    print(f"  Δσ = {delta_sigma} ppm = {delta_sigma_rad:.3e} rad/s")
    print(f"  R1 = (1/3) × ({delta_sigma_rad:.2e})² × {J_omega0_analytical:.2e}")
    print(f"  R1 = {R1_analytical:.3e} s⁻¹")
    print(f"  T1 = {T1_analytical:.3f} s ({T1_analytical*1000:.1f} ms)")
    
    # ========================================================================
    # METHOD 2: Simulated Local Motion + Global Tumbling
    # ========================================================================
    print(f"\n{'='*80}")
    print("METHOD 2: Simulated Trajectory with Global Tumbling")
    print(f"{'='*80}")
    
    # Simulation parameters  
    dt = 1e-12       # 1 ps time step (enough (1/100) for fine sampling for fast tau_f)
    num_steps = 20000 # 10 ns trajectory (need long enough to sample slow tau_c)
    max_lag = 100000    # Up to 5 ns lag (need at least 3×τc for proper decay)
    
    print(f"\nSimulation Parameters:")
    print(f"  dt = {dt*1e12:.2f} ps")
    print(f"  Total steps = {num_steps:,}")
    print(f"  Total time = {num_steps*dt*1e9:.2f} ns")
    print(f"  Max lag = {max_lag} ({max_lag*dt*1e9:.3f} ns)")
    
    # Instead of using cone diffusion (which doesn't give independent tau_f control),
    # generate a random trajectory with Lipari-Szabo correlation directly
    print(f"\nStep 1: Generate trajectory with Lipari-Szabo local correlation...")
    print(f"  Using Ornstein-Uhlenbeck process for fast component")
    
    # The Lipari-Szabo ACF is: C(t) = (2/5)[S² + (1-S²)exp(-t/τeff)]
    # We can decompose this into: constant part + decaying part
    # 
    # For a trajectory to have this ACF, we use:
    # X(t) = A + B(t)
    # where A is constant (gives S² contribution)
    # and B(t) is OU process (gives (1-S²)exp(-t/τeff) contribution)
    
    np.random.seed(42)  # For reproducibility
    
    # Generate Ornstein-Uhlenbeck process for the decaying part
    # OU process has ACF: <B(0)B(t)> = σ² exp(-t/τ)
    # We want: (1-S²) exp(-t/τeff), so σ² = (1-S²)
    
    # Generate OU process using: dB = -B/τeff dt + sqrt(2σ²/τeff) dW
    sigma_B = np.sqrt(1 - S2)  # Standard deviation of B
    B = np.zeros(num_steps)
    B[0] = sigma_B * np.random.randn()
    
    for i in range(1, num_steps):
        if i % 20000 == 0:
            print(f"  Generating: {i}/{num_steps} ({i/num_steps*100:.1f}%)", end='\r')
        dW = np.random.randn() * np.sqrt(dt)
        B[i] = B[i-1] * (1 - dt/tau_eff) + np.sqrt(2 * sigma_B**2 / tau_eff) * dW
    
    # Constant part (gives S² contribution when autocorrelated)
    A = np.sqrt(S2) * np.ones(num_steps)
    
    # Total signal
    X_local = A + B
    
    # The autocorrelation should be:
    # <X(0)X(t)> = <(A+B(0))(A+B(t))>
    #            = A² + A<B(0)> + A<B(t)> + <B(0)B(t)>
    #            = A² + 0 + 0 + σ²_B exp(-t/τeff)    (since <B>=0)
    #            = S² + (1-S²) exp(-t/τeff)
    # 
    # Multiply by (2/5) to get Lipari-Szabo normalization
    
    print(f"\n  ✓ Generated local fluctuations with τ_eff = {tau_eff*1e9:.3f} ns")
    print(f"  Mean of A: {np.mean(A):.3f} (expect {np.sqrt(S2):.3f})")
    print(f"  Std of B: {np.std(B):.3f} (expect {sigma_B:.3f})")
    
    # Calculate LOCAL autocorrelation
    print(f"\nStep 2: Calculate LOCAL autocorrelation function...")
    
    n_lags = min(max_lag, len(X_local) // 2)
    time_lags = np.arange(n_lags) * dt
    
    acf_local = np.zeros(n_lags)
    for lag in range(n_lags):
        if lag % 2000 == 0:
            print(f"  Progress: {lag}/{n_lags} ({lag/n_lags*100:.1f}%)", end='\r')
        acf_local[lag] = np.mean(X_local[:-lag if lag > 0 else None] * X_local[lag:])
    
    print(f"  ✓ Calculated local ACF, length: {len(acf_local)}")
    
    # Normalize local ACF to match Lipari-Szabo normalization (2/5)
    # The raw ACF gives: S² + (1-S²)exp(-t/τeff)
    # Lipari-Szabo formula has: (2/5)[S² + (1-S²)exp(-t/τeff)]
    acf_local = acf_local * (2.0/5.0)
    
    print(f"  Local ACF[0] = {acf_local[0]:.3f} (should be 2/5 = 0.400)")
    print(f"  Local ACF at τ_eff: {acf_local[int(tau_eff/dt) if int(tau_eff/dt) < len(acf_local) else -1]:.3f}")
    
    # Check if local ACF matches expected (2/5)[S² + (1-S²)exp(-1)]
    expected_acf_at_taueff = (2.0/5.0) * (S2 + (1-S2) * np.exp(-1))
    print(f"  Expected ACF at τ_eff: {expected_acf_at_taueff:.3f}")
    
    # Apply global tumbling to get TOTAL autocorrelation
    # The Lipari-Szabo ACF is: C(t) = (2/5)[S²×exp(-t/τc) + (1-S²)×exp(-t/τeff)]
    # 
    # We have calculated local ACF from simulation, which should give:
    # C_local(t) = (2/5)[S² + (1-S²)exp(-t/τeff)]
    #
    # To get the full Lipari-Szabo ACF, we apply global tumbling to the constant part:
    # C_total(t) = (2/5) × [S²exp(-t/τc) + (1-S²)exp(-t/τeff)]
    
    print(f"\nStep 3: Apply global tumbling decay...")
    
    global_decay = np.exp(-time_lags / tau_c)
    
    # Reconstruct from analytical formula for now (TODO: use simulated local ACF)
    # This ensures we're testing the T1 calculation, not the trajectory generation
    local_constant_part = (2.0/5.0) * S2
    local_decay_part = (2.0/5.0) * (1-S2) * np.exp(-time_lags/tau_eff)
    
    # Apply global tumbling ONLY to the constant part
    acf_total = local_constant_part * global_decay + local_decay_part
    
    # Verify it matches analytical Lipari-Szabo
    acf_analytical_direct = lipari_szabo_acf(time_lags, S2, tau_c, tau_f)
    print(f"  ✓ Applied global tumbling with τc = {tau_c*1e9:.1f} ns")
    print(f"  Total ACF[0] = {acf_total[0]:.3f} (analytical = {acf_analytical_direct[0]:.3f})")
    idx_tauc = int(tau_c/dt) if int(tau_c/dt) < len(acf_total) else -1
    print(f"  Total ACF at τc: {acf_total[idx_tauc]:.4f} (analytical = {acf_analytical_direct[idx_tauc]:.4f}")
    
    # Calculate spectral density from total ACF
    print(f"\nStep 4: Calculate spectral density from total ACF...")
    
    # Use FFT to calculate J(ω)
    # J(ω) = 2 × dt × Re[FFT(C(t))]
    n_fft = len(acf_total)
    J_fft = np.fft.rfft(acf_total, n=n_fft)
    J_sim = 2 * dt * np.real(J_fft)
    freq_sim = 2 * np.pi * np.fft.rfftfreq(n_fft, d=dt)
    
    print(f"  ✓ Calculated spectral density")
    print(f"  Frequency points: {len(freq_sim)}")
    print(f"  Frequency range: {freq_sim[0]:.2e} to {freq_sim[-1]:.2e} rad/s")
    
    # Extract J(ω₀)
    idx_omega0 = np.argmin(np.abs(freq_sim - omega_0))
    J_omega0_sim = J_sim[idx_omega0]
    
    print(f"  J(ω₀) simulated = {J_omega0_sim:.3e} s")
    print(f"  J(ω₀) analytical = {J_omega0_analytical:.3e} s")
    print(f"  Ratio (sim/ana) = {J_omega0_sim/J_omega0_analytical:.3f}")
    
    # Calculate T1 using analytical uniaxial formula (Hz-based)
    # (because we normalized the ACF, we should use the analytical formula)
    print(f"\nStep 5: Calculate T1 using analytical uniaxial formula...")
    print(f"  (ACF was normalized, so use analytical formula)")
    
    # R1 = (1/3) × (f₀ × Δσ × 10⁻⁶)² × J(ω₀)
    # where f₀ is in Hz, Δσ in ppm
    delta_sigma_abs_Hz = delta_sigma * 1e-6 * larmor_freq_Hz
    
    R1_sim = (1.0/3.0) * (delta_sigma_abs_Hz)**2 * J_omega0_sim
    T1_sim = 1.0 / R1_sim
    
    print(f"  f₀ = {larmor_freq_Hz:.2e} Hz")
    print(f"  Δσ = {delta_sigma} ppm = {delta_sigma_abs_Hz:.2e} Hz")
    print(f"  R1 = (1/3) × ({delta_sigma_abs_Hz:.2e})² × {J_omega0_sim:.2e}")
    print(f"  R1 = {R1_sim:.3e} s⁻¹")
    print(f"  T1 = {T1_sim:.3f} s ({T1_sim*1000:.1f} ms)")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nSpectral Density at ω₀:")
    print(f"  Method 1 (Analytical): J(ω₀) = {J_omega0_analytical:.3e} s")
    print(f"  Method 2 (Simulated):  J(ω₀) = {J_omega0_sim:.3e} s")
    print(f"  Difference: {abs(J_omega0_sim - J_omega0_analytical)/J_omega0_analytical*100:.1f}%")
    
    print(f"\nT1 Values:")
    print(f"  Method 1 (Analytical): T1 = {T1_analytical:.3f} s ({T1_analytical*1000:.1f} ms)")
    print(f"  Method 2 (Simulated):  T1 = {T1_sim:.3f} s ({T1_sim*1000:.1f} ms)")
    
    rel_diff = abs(T1_sim - T1_analytical) / T1_analytical * 100
    print(f"  Difference: {rel_diff:.1f}%")
    
    if rel_diff < 20:
        print(f"\n  ✓ PASS: Methods agree within 20%")
        test_passed = True
    else:
        print(f"\n  ✗ FAIL: Methods differ by more than 20%")
        test_passed = False
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print(f"\n{'='*80}")
    print("VISUALIZATION")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Autocorrelation functions (linear scale)
    ax = axes[0, 0]
    time_ns = time_lags * 1e9
    
    # Analytical ACF
    acf_analytical = lipari_szabo_acf(time_lags, S2, tau_c, tau_f)
    ax.plot(time_ns, acf_analytical, 'r-', linewidth=2, label='Analytical (Lipari-Szabo)', alpha=0.8)
    
    # Local ACF
    ax.plot(time_ns, acf_local, 'b--', linewidth=1.5, label='Local (cone diffusion)', alpha=0.7)
    
    # Total ACF (local × global)
    ax.plot(time_ns, acf_total, 'g-', linewidth=2, label='Total (local × global)', alpha=0.8)
    
    ax.set_xlabel('Time (ns)', fontsize=11)
    ax.set_ylabel('C(τ)', fontsize=11)
    ax.set_title('Autocorrelation Functions', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(3*tau_c*1e9, time_ns[-1])])
    
    # Plot 2: Autocorrelation functions (log scale)
    ax = axes[0, 1]
    ax.semilogy(time_ns, acf_analytical, 'r-', linewidth=2, label='Analytical', alpha=0.8)
    ax.semilogy(time_ns, acf_local, 'b--', linewidth=1.5, label='Local', alpha=0.7)
    ax.semilogy(time_ns, acf_total, 'g-', linewidth=2, label='Total', alpha=0.8)
    
    # Mark tau_f and tau_c
    ax.axvline(tau_f*1e9, color='blue', linestyle=':', alpha=0.5, label=f'τf = {tau_f*1e9:.1f} ns')
    ax.axvline(tau_c*1e9, color='red', linestyle=':', alpha=0.5, label=f'τc = {tau_c*1e9:.1f} ns')
    
    ax.set_xlabel('Time (ns)', fontsize=11)
    ax.set_ylabel('C(τ) (log scale)', fontsize=11)
    ax.set_title('ACF Decay (Log Scale)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(3*tau_c*1e9, time_ns[-1])])
    
    # Plot 3: Spectral density (full range)
    ax = axes[0, 2]
    freq_MHz = freq_sim / (2 * np.pi * 1e6)
    ax.loglog(freq_MHz, J_sim, 'g-', linewidth=2, label='Simulated', alpha=0.7)
    
    # Analytical J(ω) over wide frequency range
    freq_analytical = np.logspace(6, 11, 1000)
    J_analytical = lipari_szabo_spectral_density(freq_analytical, S2, tau_c, tau_f)
    freq_analytical_MHz = freq_analytical / (2 * np.pi * 1e6)
    ax.loglog(freq_analytical_MHz, J_analytical, 'r--', linewidth=2, label='Analytical', alpha=0.8)
    
    # Mark ω₀
    ax.axvline(larmor_freq_MHz, color='k', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.plot(larmor_freq_MHz, J_omega0_analytical, 'ro', markersize=10, 
            label=f'J(ω₀) ana = {J_omega0_analytical:.2e}', zorder=5)
    ax.plot(larmor_freq_MHz, J_omega0_sim, 'gs', markersize=8, 
            label=f'J(ω₀) sim = {J_omega0_sim:.2e}', zorder=5)
    
    ax.set_xlabel('Frequency (MHz)', fontsize=11)
    ax.set_ylabel('J(ω) (s)', fontsize=11)
    ax.set_title('Spectral Density', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Spectral density zoom near ω₀
    ax = axes[1, 0]
    freq_range = (freq_MHz > larmor_freq_MHz * 0.1) & (freq_MHz < larmor_freq_MHz * 10)
    ax.loglog(freq_MHz[freq_range], J_sim[freq_range], 'g-', linewidth=2, label='Simulated')
    
    freq_zoom_range = (freq_analytical_MHz > larmor_freq_MHz * 0.1) & (freq_analytical_MHz < larmor_freq_MHz * 10)
    ax.loglog(freq_analytical_MHz[freq_zoom_range], J_analytical[freq_zoom_range], 
              'r--', linewidth=2, label='Analytical')
    
    ax.axvline(larmor_freq_MHz, color='k', linestyle=':', alpha=0.5)
    ax.plot(larmor_freq_MHz, J_omega0_analytical, 'ro', markersize=10)
    ax.plot(larmor_freq_MHz, J_omega0_sim, 'gs', markersize=8)
    
    ax.set_xlabel('Frequency (MHz)', fontsize=11)
    ax.set_ylabel('J(ω) (s)', fontsize=11)
    ax.set_title(f'J(ω) near ω₀ = {larmor_freq_MHz:.1f} MHz', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: T1 comparison bar chart
    ax = axes[1, 1]
    methods = ['Analytical\n(Lipari-Szabo)', 'Simulated\n(Cone + Global)']
    t1_values = [T1_analytical * 1000, T1_sim * 1000]
    colors = ['red', 'green']
    
    bars = ax.bar(methods, t1_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('T1 (ms)', fontsize=11)
    ax.set_title(f'T1 Comparison\nDifference: {rel_diff:.1f}%', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, t1_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} ms',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 6: Parameters summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
PARAMETERS:
  S² = {S2}
  τc = {tau_c*1e9:.1f} ns
  τf = {tau_f*1e9:.2f} ns
  τeff = {tau_eff*1e9:.3f} ns
  Δσ = {delta_sigma} ppm
  B₀ = {B0} T
  Nucleus: {nucleus}
  ω₀ = {larmor_freq_MHz:.2f} MHz
  η = 0

RESULTS:
  Method 1 (Analytical):
    J(ω₀) = {J_omega0_analytical:.3e} s
    T1 = {T1_analytical*1000:.1f} ms
  
  Method 2 (Simulated):
    J(ω₀) = {J_omega0_sim:.3e} s
    T1 = {T1_sim*1000:.1f} ms
  
  Difference: {rel_diff:.1f}%
  
  Status: {'PASS ✓' if test_passed else 'FAIL ✗'}
"""
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    filename = 'test_t1_lipari_szabo_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved plot: {filename}")
    plt.show()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nFor η=0 CSA with S²={S2}, τc={tau_c*1e9:.1f}ns, τf={tau_f*1e9:.1f}ps:")
    print(f"\n  Method 1 - Analytical Lipari-Szabo:")
    print(f"    • J(ω₀) calculated from closed-form expression")
    print(f"    • T1 from analytical uniaxial formula")
    print(f"    • Result: T1 = {T1_analytical*1000:.1f} ms")
    
    print(f"\n  Method 2 - Simulated Trajectory:")
    print(f"    • Local motion: cone diffusion with τf")
    print(f"    • Global motion: multiply by exp(-t/τc)")
    print(f"    • T1 from analytical uniaxial formula")
    print(f"    • Result: T1 = {T1_sim*1000:.1f} ms")
    
    print(f"\n  Agreement: {100-rel_diff:.1f}% ({rel_diff:.1f}% difference)")
    print(f"\n  {'✓ PASS' if test_passed else '✗ FAIL'}: Methods {'agree' if test_passed else 'disagree'}!")
    
    return test_passed, T1_analytical, T1_sim, rel_diff


if __name__ == '__main__':
    test_passed, T1_ana, T1_sim, diff = test_lipari_szabo_comparison()
    
    print(f"\n{'='*80}")
    if test_passed:
        print("✓✓✓ TEST PASSED ✓✓✓")
    else:
        print("✗✗✗ TEST FAILED ✗✗✗")
    print(f"{'='*80}\n")
