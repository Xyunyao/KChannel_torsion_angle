"""
Compare Diffusion Within Cone vs On Cone Edge

This script demonstrates the difference between:
1. Diffusion within a cone (variable β: 0 ≤ β ≤ θ_cone)
2. Diffusion on cone edge (fixed β: β = θ_cone)

Both models with same S² will have different cone angles and dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/yunyao_1/Dropbox/KcsA/analysis')

from nmr_calculator import NMRConfig, TrajectoryGenerator


def compare_cone_models(S2=0.85, num_steps=5000):
    """
    Compare within-cone and edge-cone diffusion models.
    
    Parameters
    ----------
    S2 : float
        Order parameter
    num_steps : int
        Number of trajectory steps
    """
    print("="*70)
    print("COMPARISON: Diffusion Within Cone vs On Cone Edge")
    print("="*70)
    print(f"Order parameter S² = {S2:.4f}")
    print(f"Number of steps = {num_steps}")
    
    # Model 1: Within cone
    print("\n" + "-"*70)
    print("Model 1: Diffusion WITHIN Cone")
    print("-"*70)
    config_within = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=S2,
        tau_c=5e-9,
        dt=1e-12,
        num_steps=num_steps,
        verbose=False
    )
    
    gen_within = TrajectoryGenerator(config_within)
    rotations_within, _ = gen_within.generate()
    
    # Extract angles
    eulers_within = np.array([r.as_euler('ZYZ') for r in rotations_within])
    alpha_within = eulers_within[:, 0]
    beta_within = eulers_within[:, 1]
    gamma_within = eulers_within[:, 2]
    
    # Model 2: On cone edge
    print("\n" + "-"*70)
    print("Model 2: Diffusion ON Cone Edge")
    print("-"*70)
    config_edge = NMRConfig(
        trajectory_type='diffusion_cone_edge',
        S2=S2,
        tau_c=5e-9,
        dt=1e-12,
        num_steps=num_steps,
        verbose=False
    )
    
    gen_edge = TrajectoryGenerator(config_edge)
    rotations_edge, _ = gen_edge.generate()
    
    # Extract angles
    eulers_edge = np.array([r.as_euler('ZYZ') for r in rotations_edge])
    alpha_edge = eulers_edge[:, 0]
    beta_edge = eulers_edge[:, 1]
    gamma_edge = eulers_edge[:, 2]
    
    # Calculate cone angles
    # Within cone: S² = ((1 + cos(θ))/2)²
    cos_theta_within = 2*np.sqrt(S2) - 1
    theta_within = np.arccos(np.clip(cos_theta_within, -1, 1))
    
    # On edge: S² = ((1 + cos(θ)) × cos(θ) / 2)²
    sqrt_S2 = np.sqrt(S2)
    cos_theta_edge = (-1 + np.sqrt(1 + 8*sqrt_S2)) / 2
    theta_edge = np.arccos(np.clip(cos_theta_edge, -1, 1))
    
    # Print statistics
    print("\n" + "="*70)
    print("CONE ANGLES")
    print("="*70)
    print(f"Within cone: θ = {np.degrees(theta_within):.2f}°")
    print(f"On edge:     θ = {np.degrees(theta_edge):.2f}°")
    
    print("\n" + "="*70)
    print("BETA ANGLE STATISTICS")
    print("="*70)
    print(f"Within cone:")
    print(f"  Mean:  {np.degrees(beta_within.mean()):.2f}°")
    print(f"  Std:   {np.degrees(beta_within.std()):.2f}°")
    print(f"  Range: {np.degrees(beta_within.min()):.2f}° to {np.degrees(beta_within.max()):.2f}°")
    
    print(f"\nOn edge:")
    print(f"  Mean:  {np.degrees(beta_edge.mean()):.2f}°")
    print(f"  Std:   {np.degrees(beta_edge.std()):.6f}°  (should be ~0)")
    print(f"  Range: {np.degrees(beta_edge.min()):.2f}° to {np.degrees(beta_edge.max()):.2f}°")
    
    # Calculate S² from trajectories
    def calc_S2(betas):
        """Calculate S² from <P2(cos(β))>"""
        cos_beta = np.cos(betas)
        P2 = (3*cos_beta**2 - 1) / 2
        return np.mean(P2)
    
    S2_within_calc = calc_S2(beta_within)
    S2_edge_calc = calc_S2(beta_edge)
    
    print("\n" + "="*70)
    print("S² VALIDATION")
    print("="*70)
    print(f"Target S² = {S2:.4f}")
    print(f"\nWithin cone:")
    print(f"  Calculated S² = {S2_within_calc:.4f}")
    print(f"  Error = {abs(S2_within_calc - S2):.4f} ({abs(S2_within_calc - S2)/S2*100:.2f}%)")
    
    print(f"\nOn edge:")
    print(f"  Calculated S² = {S2_edge_calc:.4f}")
    print(f"  Error = {abs(S2_edge_calc - S2):.4f} ({abs(S2_edge_calc - S2)/S2*100:.2f}%)")
    
    return (rotations_within, alpha_within, beta_within, gamma_within,
            rotations_edge, alpha_edge, beta_edge, gamma_edge,
            theta_within, theta_edge)


def plot_comparison(alpha_within, beta_within, alpha_edge, beta_edge, 
                   theta_within, theta_edge, S2):
    """
    Create visualization comparing both models.
    
    Parameters
    ----------
    alpha_within, beta_within : np.ndarray
        Angles for within-cone model
    alpha_edge, beta_edge : np.ndarray
        Angles for edge-cone model
    theta_within, theta_edge : float
        Cone angles (radians)
    S2 : float
        Order parameter
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Beta angle time series
    ax1 = plt.subplot(2, 3, 1)
    time_ns = np.arange(len(beta_within)) * 1e-12 * 1e9  # Convert to ns
    ax1.plot(time_ns[:1000], np.degrees(beta_within[:1000]), 'b-', alpha=0.7, label='Within cone')
    ax1.plot(time_ns[:1000], np.degrees(beta_edge[:1000]), 'r-', alpha=0.7, label='On edge')
    ax1.axhline(np.degrees(theta_within), color='b', linestyle='--', alpha=0.5, label=f'θ_within = {np.degrees(theta_within):.1f}°')
    ax1.axhline(np.degrees(theta_edge), color='r', linestyle='--', alpha=0.5, label=f'θ_edge = {np.degrees(theta_edge):.1f}°')
    ax1.set_xlabel('Time (ns)', fontsize=12)
    ax1.set_ylabel('β angle (degrees)', fontsize=12)
    ax1.set_title('β Angle Time Series (first 1000 steps)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Beta angle histogram
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(np.degrees(beta_within), bins=50, alpha=0.6, color='blue', label='Within cone', density=True)
    ax2.hist(np.degrees(beta_edge), bins=50, alpha=0.6, color='red', label='On edge', density=True)
    ax2.axvline(np.degrees(theta_within), color='b', linestyle='--', linewidth=2, label=f'θ_within = {np.degrees(theta_within):.1f}°')
    ax2.axvline(np.degrees(theta_edge), color='r', linestyle='--', linewidth=2, label=f'θ_edge = {np.degrees(theta_edge):.1f}°')
    ax2.set_xlabel('β angle (degrees)', fontsize=12)
    ax2.set_ylabel('Probability density', fontsize=12)
    ax2.set_title('β Angle Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 3D cone visualization
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    
    # Plot cone surface
    phi = np.linspace(0, 2*np.pi, 50)
    z_cone = np.linspace(0, 1, 30)
    phi_grid, z_grid = np.meshgrid(phi, z_cone)
    
    # Within cone
    x_within = z_grid * np.sin(theta_within) * np.cos(phi_grid)
    y_within = z_grid * np.sin(theta_within) * np.sin(phi_grid)
    z_within = z_grid * np.cos(theta_within)
    ax3.plot_surface(x_within, y_within, z_within, alpha=0.2, color='blue')
    
    # Edge cone
    x_edge = z_grid * np.sin(theta_edge) * np.cos(phi_grid)
    y_edge = z_grid * np.sin(theta_edge) * np.sin(phi_grid)
    z_edge = z_grid * np.cos(theta_edge)
    ax3.plot_surface(x_edge, y_edge, z_edge, alpha=0.2, color='red')
    
    # Plot sample trajectory points
    sample_within = beta_within[::10][:100]
    alpha_sample_within = alpha_within[::10][:100]
    x_traj_within = np.sin(sample_within) * np.cos(alpha_sample_within)
    y_traj_within = np.sin(sample_within) * np.sin(alpha_sample_within)
    z_traj_within = np.cos(sample_within)
    ax3.scatter(x_traj_within, y_traj_within, z_traj_within, c='blue', s=10, alpha=0.6)
    
    sample_edge = beta_edge[::10][:100]
    alpha_sample_edge = alpha_edge[::10][:100]
    x_traj_edge = np.sin(sample_edge) * np.cos(alpha_sample_edge)
    y_traj_edge = np.sin(sample_edge) * np.sin(alpha_sample_edge)
    z_traj_edge = np.cos(sample_edge)
    ax3.scatter(x_traj_edge, y_traj_edge, z_traj_edge, c='red', s=10, alpha=0.6)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D Cone Geometry', fontsize=14, fontweight='bold')
    ax3.set_box_aspect([1,1,1])
    
    # 4. Alpha angle time series
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time_ns[:1000], np.degrees(alpha_within[:1000]), 'b-', alpha=0.7, label='Within cone')
    ax4.plot(time_ns[:1000], np.degrees(alpha_edge[:1000]), 'r-', alpha=0.7, label='On edge')
    ax4.set_xlabel('Time (ns)', fontsize=12)
    ax4.set_ylabel('α angle (degrees)', fontsize=12)
    ax4.set_title('α Angle Time Series (azimuthal)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. 2D projection (α vs β)
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(np.degrees(alpha_within[::10]), np.degrees(beta_within[::10]), 
               c='blue', s=5, alpha=0.3, label='Within cone')
    ax5.scatter(np.degrees(alpha_edge[::10]), np.degrees(beta_edge[::10]), 
               c='red', s=5, alpha=0.3, label='On edge')
    ax5.axhline(np.degrees(theta_within), color='b', linestyle='--', alpha=0.5)
    ax5.axhline(np.degrees(theta_edge), color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('α angle (degrees)', fontsize=12)
    ax5.set_ylabel('β angle (degrees)', fontsize=12)
    ax5.set_title('Angle Correlation (α vs β)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. P2(cos(β)) time series
    ax6 = plt.subplot(2, 3, 6)
    P2_within = (3*np.cos(beta_within)**2 - 1) / 2
    P2_edge = (3*np.cos(beta_edge)**2 - 1) / 2
    
    # Running average
    window = 100
    P2_within_smooth = np.convolve(P2_within, np.ones(window)/window, mode='valid')
    P2_edge_smooth = np.convolve(P2_edge, np.ones(window)/window, mode='valid')
    time_smooth = time_ns[:len(P2_within_smooth)]
    
    ax6.plot(time_smooth, P2_within_smooth, 'b-', alpha=0.7, label='Within cone')
    ax6.plot(time_smooth, P2_edge_smooth, 'r-', alpha=0.7, label='On edge')
    ax6.axhline(S2, color='k', linestyle='--', linewidth=2, label=f'S² = {S2:.4f}')
    ax6.set_xlabel('Time (ns)', fontsize=12)
    ax6.set_ylabel('P₂(cos β)', fontsize=12)
    ax6.set_title('Order Parameter P₂ (running avg)', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Cone Diffusion Models Comparison (S² = {S2:.4f})', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    filename = f'cone_comparison_S2_{S2:.2f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {filename}")
    
    plt.show()


if __name__ == '__main__':
    # Compare models
    results = compare_cone_models(S2=0.85, num_steps=5000)
    
    # Unpack results
    (rotations_within, alpha_within, beta_within, gamma_within,
     rotations_edge, alpha_edge, beta_edge, gamma_edge,
     theta_within, theta_edge) = results
    
    # Create visualization
    try:
        plot_comparison(alpha_within, beta_within, alpha_edge, beta_edge,
                       theta_within, theta_edge, S2=0.85)
    except ImportError:
        print("\nNote: Install matplotlib to generate plots")
    
    print("\n" + "="*70)
    print("KEY DIFFERENCES")
    print("="*70)
    print("1. Within Cone:")
    print("   - β varies from 0 to θ_cone")
    print("   - Explores full cone volume")
    print("   - S² = ((1 + cos(θ))/2)²")
    print("   - Smaller cone angle for same S²")
    
    print("\n2. On Edge:")
    print("   - β fixed at θ_cone")
    print("   - Diffusion only in α (azimuthal)")
    print("   - S² = ((1 + cos(θ)) × cos(θ) / 2)²")
    print("   - Larger cone angle for same S²")
    
    print("\n" + "="*70)
    print("COMPLETED")
    print("="*70)
