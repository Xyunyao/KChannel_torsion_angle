"""
Example: Complete NMR T1 Calculation

This example demonstrates the full NMR calculation pipeline:
1. Generate diffusion-on-cone trajectory
2. Calculate spectral density
3. Calculate T1 relaxation time

Compare with analytical Lipari-Szabo model.
"""

import numpy as np
import matplotlib.pyplot as plt
from nmr_calculator import NMRConfig, NMRPipeline


def example_csa_t1():
    """
    Calculate T1 for 15N CSA relaxation.
    
    Typical parameters for backbone amide 15N:
    - S² = 0.85 (moderately restricted)
    - τc = 5 ns (small protein)
    - Δσ = 160 ppm (typical for amide 15N)
    - B₀ = 14.1 T (600 MHz 1H frequency)
    """
    print("="*70)
    print("EXAMPLE: 15N CSA T1 Calculation")
    print("="*70)
    
    # Configure calculation
    config = NMRConfig(
        # Trajectory parameters
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=5e-9,  # 5 ns
        dt=1e-12,    # 1 ps
        num_steps=20000,
        
        # NMR parameters
        B0=14.1,     # Tesla (600 MHz)
        nucleus='15N',
        interaction_type='CSA',
        delta_sigma=160.0,  # ppm
        eta=0.0,     # Axially symmetric
        
        # Calculation options
        max_lag=10000,
        lag_step=10,
        zero_fill_factor=4,
        frequency_markers=True,
        
        # Output options
        calculate_T1=True,
        calculate_T2=True,
        output_dir='example_output',
        save_intermediate=True,
        verbose=True
    )
    
    # Run pipeline
    pipeline = NMRPipeline(config)
    results = pipeline.run()
    
    # Save complete results
    pipeline.save_all_results('example_output/csa_t1_results.npz')
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"T1 = {results['T1']:.4f} s = {results['T1']*1000:.2f} ms")
    print(f"T2 = {results['T2']:.4f} s = {results['T2']*1000:.2f} ms")
    print(f"T1/T2 ratio = {results['T1']/results['T2']:.2f}")
    print("="*70)
    
    return results


def example_dipolar_t1():
    """
    Calculate T1 for 15N-1H dipolar relaxation.
    
    Typical parameters for backbone amide:
    - S² = 0.85
    - τc = 5 ns
    - r(NH) = 1.02 Å
    """
    print("\n" + "="*70)
    print("EXAMPLE: 15N-1H Dipolar T1 Calculation")
    print("="*70)
    
    config = NMRConfig(
        # Trajectory
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=5e-9,
        dt=1e-12,
        num_steps=20000,
        
        # NMR parameters
        B0=14.1,
        nucleus='15N',
        interaction_type='dipolar',
        
        # Calculation
        max_lag=10000,
        lag_step=10,
        zero_fill_factor=4,
        
        # Output
        calculate_T1=True,
        calculate_T2=True,
        output_dir='example_output',
        save_intermediate=False,
        verbose=True
    )
    
    pipeline = NMRPipeline(config)
    results = pipeline.run()
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"T1 = {results['T1']:.4f} s = {results['T1']*1000:.2f} ms")
    print(f"T2 = {results['T2']:.4f} s = {results['T2']*1000:.2f} ms")
    print(f"NOE = {results['NOE']:.3f}")
    print("="*70)
    
    return results


def example_plot_spectral_density(results):
    """
    Plot spectral density function.
    
    Parameters
    ----------
    results : Dict
        Results from pipeline.run()
    """
    import matplotlib.pyplot as plt
    
    freq = results['frequencies']
    J = results['spectral_density']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot on log-log scale
    ax.loglog(freq / (2*np.pi), J, 'b-', linewidth=2, label='FFT calculation')
    
    # Mark specific frequencies
    if 'frequency_markers' in results and results['frequency_markers']:
        for name, (omega, J_val) in results['frequency_markers'].items():
            freq_hz = omega / (2*np.pi)
            ax.plot(freq_hz, J_val, 'ro', markersize=8)
            ax.text(freq_hz, J_val*1.5, name, ha='center', fontsize=10)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('J(ω) (s)', fontsize=14)
    ax.set_title('Spectral Density Function', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('example_output/spectral_density.png', dpi=300)
    print("\n✓ Saved spectral density plot to: example_output/spectral_density.png")
    
    plt.show()


def example_plot_acf(results):
    """
    Plot autocorrelation function.
    
    Parameters
    ----------
    results : Dict
        Results from pipeline.run()
    """
    time_lags = results['time_lags']
    acf = results['acf']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(time_lags * 1e9, acf, 'b-', linewidth=2, label='ACF')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Time lag (ns)', fontsize=14)
    ax.set_ylabel('C(τ) (normalized)', fontsize=14)
    ax.set_title('Autocorrelation Function', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('example_output/autocorrelation.png', dpi=300)
    print("✓ Saved ACF plot to: example_output/autocorrelation.png")
    
    plt.show()


def example_parameter_scan():
    """
    Scan T1 as function of correlation time.
    
    Demonstrates how T1 changes with molecular tumbling.
    """
    print("\n" + "="*70)
    print("EXAMPLE: T1 vs Correlation Time")
    print("="*70)
    
    # Scan tau_c from 0.1 ns to 100 ns
    tau_c_values = np.logspace(-10, -8, 20)  # 0.1 ns to 100 ns
    T1_values = []
    
    for tau_c in tau_c_values:
        config = NMRConfig(
            trajectory_type='diffusion_cone',
            S2=0.85,
            tau_c=tau_c,
            dt=1e-12,
            num_steps=10000,
            B0=14.1,
            nucleus='15N',
            interaction_type='CSA',
            delta_sigma=160.0,
            max_lag=5000,
            lag_step=10,
            calculate_T1=True,
            calculate_T2=False,
            verbose=False
        )
        
        pipeline = NMRPipeline(config)
        results = pipeline.run()
        T1_values.append(results['T1'])
        
        print(f"τc = {tau_c*1e9:.2f} ns → T1 = {results['T1']*1000:.1f} ms")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(np.array(tau_c_values)*1e9, np.array(T1_values)*1000, 'bo-', 
                linewidth=2, markersize=8)
    ax.set_xlabel('Correlation time τc (ns)', fontsize=14)
    ax.set_ylabel('T1 (ms)', fontsize=14)
    ax.set_title('T1 vs Correlation Time', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_output/t1_vs_tau_c.png', dpi=300)
    print("\n✓ Saved parameter scan to: example_output/t1_vs_tau_c.png")
    
    plt.show()


def example_compare_csa_vs_dipolar():
    """
    Compare CSA and dipolar contributions to T1.
    """
    print("\n" + "="*70)
    print("EXAMPLE: CSA vs Dipolar Relaxation")
    print("="*70)
    
    # CSA
    config_csa = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=5e-9,
        dt=1e-12,
        num_steps=20000,
        B0=14.1,
        nucleus='15N',
        interaction_type='CSA',
        delta_sigma=160.0,
        calculate_T1=True,
        verbose=False
    )
    
    pipeline_csa = NMRPipeline(config_csa)
    results_csa = pipeline_csa.run()
    
    # Dipolar
    config_dipolar = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=5e-9,
        dt=1e-12,
        num_steps=20000,
        B0=14.1,
        nucleus='15N',
        interaction_type='dipolar',
        calculate_T1=True,
        verbose=False
    )
    
    pipeline_dipolar = NMRPipeline(config_dipolar)
    results_dipolar = pipeline_dipolar.run()
    
    print("\nComparison:")
    print(f"  CSA:     T1 = {results_csa['T1']*1000:.2f} ms")
    print(f"  Dipolar: T1 = {results_dipolar['T1']*1000:.2f} ms")
    print(f"  Ratio (dipolar/CSA): {results_dipolar['T1']/results_csa['T1']:.2f}")


if __name__ == '__main__':
    # Run examples
    
    # 1. Basic CSA T1 calculation
    results = example_csa_t1()
    
    # 2. Plot results
    try:
        example_plot_spectral_density(results)
        example_plot_acf(results)
    except ImportError:
        print("\nNote: Install matplotlib to generate plots")
    
    # 3. Dipolar calculation
    example_dipolar_t1()
    
    # 4. Parameter scan
    try:
        example_parameter_scan()
    except:
        print("\nSkipping parameter scan")
    
    # 5. Compare mechanisms
    example_compare_csa_vs_dipolar()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
