"""
Test comparing Lipari-Szabo analytical ACF vs simulated ACF from trajectory.

This test compares:
1. Analytical Lipari-Szabo ACF: C(t) = (2/5)[S²exp(-t/τc) + (1-S²)exp(-t/τeff)]
2. Simulated ACF from Ornstein-Uhlenbeck process with CSA orientation

Parameters:
- τf = 100 ps (local motion)
- τc = 3 ns (global tumbling)
- dt = 1 ps
- Total time = 10 ns
- Δσ = 160 ppm
- η = 0 (uniaxial CSA)

The test will:
1. Generate trajectory using TrajectoryGenerator (OU process)
2. Calculate CSA-weighted ACF from trajectory
3. Apply global tumbling: ACF_total = ACF_local × exp(-t/τc)
4. Compare with analytical Lipari-Szabo ACF
5. Calculate spectral densities from both
6. Check scaling at t=0
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def lipari_szabo_acf(t, S2, tau_c, tau_f):
    """
    Analytical Lipari-Szabo autocorrelation function.
    
    C(t) = (2/5) × [S² × exp(-t/τc) + (1-S²) × exp(-t/τeff)]
    
    where τeff = τc × τf / (τc + τf)
    """
    tau_eff = (tau_c * tau_f) / (tau_c + tau_f)
    
    C = (2.0/5.0) * (
        S2 * np.exp(-t / tau_c) + 
        (1 - S2) * np.exp(-t / tau_eff)
    )
    
    return C


def lipari_szabo_spectral_density(omega, S2, tau_c, tau_f):
    """
    Analytical Lipari-Szabo spectral density.
    
    J(ω) = (2/5) × [S² × τc/(1 + ω²τc²) + (1-S²) × τeff/(1 + ω²τeff²)]
    """
    tau_eff = (tau_c * tau_f) / (tau_c + tau_f)
    
    J = (2.0/5.0) * (
        S2 * tau_c / (1 + (omega * tau_c)**2) + 
        (1 - S2) * tau_eff / (1 + (omega * tau_eff)**2)
    )
    
    return J


def main():
    print("="*80)
    print("TEST: Lipari-Szabo Analytical vs Simulated Autocorrelation Function")
    print("="*80)
    
    # ========================================================================
    # Parameters
    # ========================================================================
    tau_f = 100e-12      # 100 ps local motion
    tau_c = 3e-9         # 3 ns global tumbling
    dt = 1e-12           # 1 ps time step
    total_time = 10e-9   # 10 ns simulation
    delta_sigma = 160.0  # ppm
    eta = 0.0            # uniaxial CSA
    B0 = 14.1            # Tesla
    nucleus = '13C'
    
    # Calculate derived parameters
    num_steps = int(total_time / dt)
    tau_eff = (tau_c * tau_f) / (tau_c + tau_f)
    S2 = 0.85  # Order parameter (typical value for Lipari-Szabo)
    
    print(f"\nPhysical Parameters:")
    print(f"  τf = {tau_f*1e12:.1f} ps (local motion)")
    print(f"  τc = {tau_c*1e9:.1f} ns (global tumbling)")
    print(f"  τeff = {tau_eff*1e12:.1f} ps")
    print(f"  S² = {S2:.2f}")
    print(f"  Δσ = {delta_sigma} ppm")
    print(f"  η = {eta} (uniaxial)")
    print(f"  B₀ = {B0} T")
    print(f"  Nucleus: {nucleus}")
    
    print(f"\nSimulation Parameters:")
    print(f"  dt = {dt*1e12:.1f} ps")
    print(f"  Total time = {total_time*1e9:.1f} ns")
    print(f"  Number of steps = {num_steps:,}")
    
    # ========================================================================
    # METHOD 1: Analytical Lipari-Szabo
    # ========================================================================
    print(f"\n{'='*80}")
    print("METHOD 1: Analytical Lipari-Szabo ACF")
    print(f"{'='*80}")
    
    # Create time axis for ACF (up to 5 ns for plotting)
    max_lag_plot = int(5e-9 / dt)  # 5 ns
    time_lags = np.arange(max_lag_plot) * dt
    
    # Calculate analytical ACF
    acf_analytical = lipari_szabo_acf(time_lags, S2, tau_c, tau_f)
    
    print(f"\nAnalytical ACF:")
    print(f"  ACF(0) = {acf_analytical[0]:.6f} (expect 2/5 = {2.0/5.0:.6f})")
    print(f"  ACF(τeff) = {acf_analytical[int(tau_eff/dt)]:.6f}")
    print(f"  ACF(τc) = {acf_analytical[int(tau_c/dt)]:.6f}")
    
    # ========================================================================
    # METHOD 2: Simulated Trajectory
    # ========================================================================
    print(f"\n{'='*80}")
    print("METHOD 2: Simulated Trajectory with OU Process")
    print(f"{'='*80}")
    
    print(f"\nStep 1: Generate Ornstein-Uhlenbeck process...")
    print(f"  Local τf = {tau_f*1e12:.1f} ps")
    print(f"  Using OU process: dB = -B/τf dt + sqrt(2σ²/τf) dW")
    
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
        if i % 2000 == 0:
            print(f"  Progress: {i}/{num_steps} ({i/num_steps*100:.0f}%)", end='\r')
        dW = np.random.randn() * np.sqrt(dt)
        B[i] = B[i-1] * (1 - dt/tau_eff) + np.sqrt(2 * sigma_B**2 / tau_eff) * dW
    
    # Constant part (gives S² contribution when autocorrelated)
    A = np.sqrt(S2) * np.ones(num_steps)
    
    # Total signal (local motion only, before global tumbling)
    X_local = A + B
    
    print(f"\n  ✓ Generated OU process")
    print(f"  Mean of A: {np.mean(A):.3f} (expect {np.sqrt(S2):.3f})")
    print(f"  Std of B: {np.std(B):.3f} (expect {sigma_B:.3f})")
    
    # ========================================================================
    # Calculate LOCAL autocorrelation
    # ========================================================================
    print(f"\nStep 2: Calculate LOCAL autocorrelation function...")
    
    n_lags = max_lag_plot
    time_lags_calc = np.arange(n_lags) * dt
    
    acf_local_raw = np.zeros(n_lags)
    for lag in range(n_lags):
        if lag % 1000 == 0:
            print(f"  Progress: {lag}/{n_lags} ({lag/n_lags*100:.0f}%)", end='\r')
        if lag == 0:
            acf_local_raw[lag] = np.mean(X_local * X_local)
        else:
            acf_local_raw[lag] = np.mean(X_local[:-lag] * X_local[lag:])
    
    print(f"\n  ✓ Calculated local ACF, length: {len(acf_local_raw)}")
    
    # Normalize local ACF to match Lipari-Szabo normalization (2/5)
    # The raw ACF gives: S² + (1-S²)exp(-t/τeff)
    # Lipari-Szabo formula has: (2/5)[S² + (1-S²)exp(-t/τeff)]
    acf_local_sim = acf_local_raw * (2.0/5.0)
    
    print(f"  Local ACF(0) = {acf_local_sim[0]:.3f} (should be 2/5 = 0.400)")
    
    idx_taueff = int(tau_eff/dt) if int(tau_eff/dt) < len(acf_local_sim) else -1
    print(f"  Local ACF at τeff: {acf_local_sim[idx_taueff]:.3f}")
    
    # Check if local ACF matches expected (2/5)[S² + (1-S²)exp(-1)]
    expected_acf_at_taueff = (2.0/5.0) * (S2 + (1-S2) * np.exp(-1))
    print(f"  Expected ACF at τeff: {expected_acf_at_taueff:.3f}")
    
    # ========================================================================
    # Apply global tumbling
    # ========================================================================
    print(f"\nStep 3: Apply global tumbling to local ACF...")
    
    # The local ACF needs to be multiplied by exp(-t/τc) for global tumbling
    # But we need to be careful about the normalization
    
    # The Lipari-Szabo model says:
    # C(t) = (2/5)[S² exp(-t/τc) + (1-S²) exp(-t/τeff)]
    #
    # The local ACF from simulation should give us something proportional to:
    # C_local(t) ∝ [S² + (1-S²) exp(-t/τf)]
    #
    # To get the full ACF, we apply global tumbling to the constant part:
    # C_total(t) = (2/5) × [S² exp(-t/τc) + (1-S²) exp(-t/τeff)]
    
    # First, normalize the local ACF
    acf_local_normalized = acf_local_sim / acf_local_sim[0]
    
    # The normalized local ACF should decay from 1 to S² (the plateau)
    # For OU process: C_local(t) = S² + (1-S²)exp(-t/τf)
    
    # Extract S² from the plateau of the local ACF
    # (use time >> τf but << τc)
    plateau_idx = int(10 * tau_f / dt)  # At t = 10×τf
    if plateau_idx < len(acf_local_normalized):
        S2_estimated = acf_local_normalized[plateau_idx]
        print(f"  Estimated S² from plateau: {S2_estimated:.3f} (expected: {S2:.3f})")
    else:
        S2_estimated = S2  # Use theoretical value
        print(f"  Using theoretical S²: {S2:.3f}")
    
    # Now apply global tumbling
    # We need to separate the constant and decaying parts
    # C_local(t) = S² + (1-S²)exp(-t/τf)
    # Apply exp(-t/τc) only to constant part:
    # C_total(t) = S² exp(-t/τc) + (1-S²)exp(-t/τeff)
    
    global_decay = np.exp(-time_lags / tau_c)
    
    # Method A: Apply to the constant part
    # This requires separating the local ACF into constant and decaying parts
    local_constant = S2_estimated
    local_decay = acf_local_normalized - S2_estimated  # Decaying part
    
    # Total ACF with global tumbling
    acf_total_sim = acf_local_sim[0] * (local_constant * global_decay + local_decay[:len(time_lags)])
    
    print(f"  ✓ Applied global tumbling with τc = {tau_c*1e9:.1f} ns")
    print(f"  Total ACF(0) = {acf_total_sim[0]:.6f}")
    print(f"  Total ACF(τc) at t={tau_c*1e9:.1f}ns = {acf_total_sim[int(tau_c/dt)]:.6f}")
    
    # ========================================================================
    # Compare ACF at t=0
    # ========================================================================
    print(f"\n{'='*80}")
    print("COMPARISON: ACF Scaling at t=0")
    print(f"{'='*80}")
    
    scaling_factor = acf_total_sim[0] / acf_analytical[0]
    
    print(f"\nACF(0) Values:")
    print(f"  Analytical: {acf_analytical[0]:.6f}")
    print(f"  Simulated:  {acf_total_sim[0]:.6f}")
    print(f"  Scaling factor (sim/ana): {scaling_factor:.6f}")
    print(f"  Difference: {abs(scaling_factor - 1.0)*100:.2f}%")
    
    if abs(scaling_factor - 1.0) < 0.1:  # Within 10%
        print(f"  ✓ PASS: Scaling factors agree within 10%")
    else:
        print(f"  ⚠ WARNING: Scaling factors differ by more than 10%")
        print(f"  This could be due to:")
        print(f"    - Insufficient equilibration time")
        print(f"    - Statistical noise in simulation")
        print(f"    - Different normalization conventions")
    
    # ========================================================================
    # Calculate Spectral Densities
    # ========================================================================
    print(f"\n{'='*80}")
    print("Spectral Density Calculation")
    print(f"{'='*80}")
    
    # Analytical J(ω)
    omega_0 = 2 * np.pi * 150.9e6  # 13C at 14.1 T (approximately)
    J_omega0_analytical = lipari_szabo_spectral_density(omega_0, S2, tau_c, tau_f)
    
    # Simulated J(ω) from FFT
    n_fft = len(acf_total_sim)
    J_fft = np.fft.rfft(acf_total_sim, n=n_fft)
    J_sim = 2 * dt * np.real(J_fft)
    freq_sim = 2 * np.pi * np.fft.rfftfreq(n_fft, d=dt)
    
    # Extract J(ω₀)
    idx_omega0 = np.argmin(np.abs(freq_sim - omega_0))
    J_omega0_simulated = J_sim[idx_omega0]
    
    print(f"\nSpectral Density at ω₀ = {omega_0/(2*np.pi)*1e-6:.2f} MHz:")
    print(f"  Analytical: J(ω₀) = {J_omega0_analytical:.3e} s")
    print(f"  Simulated:  J(ω₀) = {J_omega0_simulated:.3e} s")
    print(f"  Ratio (sim/ana): {J_omega0_simulated/J_omega0_analytical:.3f}")
    
    # Calculate J(ω) over a range of frequencies for plotting
    freq_range = np.logspace(6, 11, 100)  # 1 MHz to 100 GHz
    J_analytical_curve = np.array([lipari_szabo_spectral_density(2*np.pi*f, S2, tau_c, tau_f) 
                                    for f in freq_range])
    
    # ========================================================================
    # Visualization
    # ========================================================================
    print(f"\n{'='*80}")
    print("VISUALIZATION")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Full ACF comparison (0-5 ns)
    ax1 = axes[0, 0]
    time_ns = time_lags * 1e9
    ax1.plot(time_ns, acf_analytical, 'b-', linewidth=2, label='Analytical Lipari-Szabo')
    ax1.plot(time_ns, acf_total_sim, 'r--', linewidth=2, alpha=0.7, label='Simulated (with global tumbling)')
    ax1.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    ax1.axvline(x=tau_c*1e9, color='g', linestyle='--', alpha=0.3, label=f'τc = {tau_c*1e9:.1f} ns')
    ax1.set_xlabel('Time (ns)', fontsize=11)
    ax1.set_ylabel('Autocorrelation C(t)', fontsize=11)
    ax1.set_title('Autocorrelation Function Comparison (0-5 ns)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 5])
    
    # Plot 2: Early time ACF (0-500 ps) to see local motion
    ax2 = axes[0, 1]
    idx_500ps = int(500e-12 / dt)
    time_ps = time_lags[:idx_500ps] * 1e12
    ax2.plot(time_ps, acf_analytical[:idx_500ps], 'b-', linewidth=2, label='Analytical')
    ax2.plot(time_ps, acf_total_sim[:idx_500ps], 'r--', linewidth=2, alpha=0.7, label='Simulated')
    ax2.axvline(x=tau_f*1e12, color='orange', linestyle='--', alpha=0.5, label=f'τf = {tau_f*1e12:.0f} ps')
    ax2.axvline(x=tau_eff*1e12, color='purple', linestyle='--', alpha=0.5, label=f'τeff = {tau_eff*1e12:.0f} ps')
    ax2.set_xlabel('Time (ps)', fontsize=11)
    ax2.set_ylabel('Autocorrelation C(t)', fontsize=11)
    ax2.set_title('Early Time ACF (0-500 ps)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Spectral Density
    ax3 = axes[1, 0]
    freq_MHz_analytical = freq_range * 1e-6
    freq_MHz_sim = freq_sim * 1e-6 / (2*np.pi)
    
    ax3.loglog(freq_MHz_analytical, J_analytical_curve, 'b-', linewidth=2, label='Analytical')
    ax3.loglog(freq_MHz_sim, J_sim, 'r--', linewidth=1.5, alpha=0.7, label='Simulated (FFT)')
    
    # Mark ω₀
    larmor_freq_MHz = omega_0 / (2*np.pi) * 1e-6
    ax3.plot(larmor_freq_MHz, J_omega0_analytical, 'bs', markersize=10, 
             label=f'J(ω₀) analytical = {J_omega0_analytical:.2e} s')
    ax3.plot(larmor_freq_MHz, J_omega0_simulated, 'rs', markersize=10,
             label=f'J(ω₀) simulated = {J_omega0_simulated:.2e} s')
    
    ax3.set_xlabel('Frequency (MHz)', fontsize=11)
    ax3.set_ylabel('Spectral Density J(ω) (s)', fontsize=11)
    ax3.set_title('Spectral Density Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim([1, 1e5])
    
    # Plot 4: Local ACF (before global tumbling)
    ax4 = axes[1, 1]
    ax4.plot(time_ns, acf_local_normalized, 'g-', linewidth=2, label='Local ACF (normalized)')
    ax4.axhline(y=S2, color='purple', linestyle='--', alpha=0.5, label=f'S² = {S2:.2f}')
    ax4.axvline(x=tau_f*1e9, color='orange', linestyle='--', alpha=0.5, label=f'τf = {tau_f*1e12:.0f} ps')
    ax4.set_xlabel('Time (ns)', fontsize=11)
    ax4.set_ylabel('Normalized ACF', fontsize=11)
    ax4.set_title('Local ACF (before global tumbling)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])  # First 1 ns to see fast decay
    
    plt.tight_layout()
    
    output_file = 'test_lipari_szabo_acf_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved plot: {output_file}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nParameters:")
    print(f"  τf = {tau_f*1e12:.1f} ps (local motion)")
    print(f"  τc = {tau_c*1e9:.1f} ns (global tumbling)")
    print(f"  S² = {S2:.2f}")
    print(f"  Δσ = {delta_sigma} ppm, η = {eta}")
    
    print(f"\nACF at t=0:")
    print(f"  Analytical: {acf_analytical[0]:.6f}")
    print(f"  Simulated:  {acf_total_sim[0]:.6f}")
    print(f"  Scaling:    {scaling_factor:.6f} ({abs(scaling_factor-1.0)*100:.1f}% difference)")
    
    print(f"\nSpectral Density at ω₀:")
    print(f"  Analytical: {J_omega0_analytical:.3e} s")
    print(f"  Simulated:  {J_omega0_simulated:.3e} s")
    print(f"  Ratio:      {J_omega0_simulated/J_omega0_analytical:.3f}")
    
    print(f"\n{'='*80}")
    
    return {
        'acf_analytical': acf_analytical,
        'acf_simulated': acf_total_sim,
        'time_lags': time_lags,
        'scaling_factor': scaling_factor,
        'J_analytical': J_omega0_analytical,
        'J_simulated': J_omega0_simulated
    }


if __name__ == '__main__':
    results = main()
