#!/usr/bin/env python3
"""
Example: Using Pre-computed Euler Angles (Bypass Module 1)

This script demonstrates how to use Module 2 to load Euler angles directly
from a file, completely bypassing Module 1 (trajectory generation).

Use case: You have pre-computed orientations from MD simulations or other
sources and want to analyze them with the NMR calculator.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import NMRConfig
from euler_converter import EulerConverter


def create_example_euler_file(filename='example_euler.npz', n_steps=1000):
    """
    Create an example Euler angle file for demonstration.
    
    In practice, you would have these from your MD simulation or
    other trajectory analysis.
    """
    print(f"\n{'='*70}")
    print("Creating Example Euler Angle File")
    print(f"{'='*70}")
    
    # Simulate some example Euler angles
    # In reality, these would come from your MD trajectory analysis
    
    # Example: Diffusion on a cone (similar to Module 1, but pre-computed)
    np.random.seed(42)
    
    # Alpha: azimuthal angle (0 to 2π)
    alpha = np.random.uniform(-np.pi, np.pi, n_steps)
    
    # Beta: polar angle (restricted cone, 0 to ~30°)
    beta = np.abs(np.random.normal(0, np.radians(10), n_steps))
    beta = np.clip(beta, 0, np.radians(30))
    
    # Gamma: intrinsic rotation (0 to 2π)
    gamma = np.random.uniform(0, 2*np.pi, n_steps)
    
    # Combine into (n_steps, 3) array
    euler_angles = np.column_stack([alpha, beta, gamma])
    
    # Save with metadata (recommended format)
    np.savez(filename,
             euler_angles=euler_angles,
             convention='ZYZ',
             units='radians',
             source='Example MD trajectory',
             timestep=2e-12,  # 2 ps
             n_steps=n_steps,
             description='Simulated protein backbone orientations')
    
    print(f"  ✓ Created example file: {filename}")
    print(f"  ✓ Number of frames: {n_steps}")
    print(f"  ✓ Convention: ZYZ")
    print(f"  ✓ Units: radians")
    print(f"\n  Angular ranges:")
    print(f"    α: {np.degrees(alpha.min()):.1f}° to {np.degrees(alpha.max()):.1f}°")
    print(f"    β: {np.degrees(beta.min()):.1f}° to {np.degrees(beta.max()):.1f}°")
    print(f"    γ: {np.degrees(gamma.min()):.1f}° to {np.degrees(gamma.max()):.1f}°")
    
    return filename


def load_and_analyze(filename):
    """
    Load Euler angles from file and perform basic analysis.
    
    This bypasses Module 1 entirely!
    """
    print(f"\n{'='*70}")
    print("Loading Euler Angles (BYPASSING MODULE 1)")
    print(f"{'='*70}")
    
    # Create config - no trajectory generation parameters needed!
    config = NMRConfig(
        # Only need NMR parameters, not trajectory parameters
        B0=14.1,  # Magnetic field (Tesla)
        verbose=True
    )
    
    # Create converter
    converter = EulerConverter(config)
    
    # Load from file - MODULE 1 BYPASSED!
    euler_angles = converter.convert(from_file=filename)
    
    return euler_angles


def analyze_euler_angles(euler_angles):
    """Perform basic analysis on loaded Euler angles."""
    print(f"\n{'='*70}")
    print("Analysis of Loaded Euler Angles")
    print(f"{'='*70}")
    
    n_steps = len(euler_angles)
    
    print(f"\n  Dataset:")
    print(f"    Total frames: {n_steps}")
    print(f"    Shape: {euler_angles.shape}")
    print(f"    Memory: {euler_angles.nbytes / 1024:.1f} KB")
    
    # Calculate statistics for each angle
    for i, name in enumerate(['α (alpha)', 'β (beta)', 'γ (gamma)']):
        angles_deg = np.degrees(euler_angles[:, i])
        print(f"\n  {name}:")
        print(f"    Mean:   {angles_deg.mean():7.2f}°")
        print(f"    Std:    {angles_deg.std():7.2f}°")
        print(f"    Min:    {angles_deg.min():7.2f}°")
        print(f"    Max:    {angles_deg.max():7.2f}°")
        print(f"    Range:  {angles_deg.max() - angles_deg.min():7.2f}°")
    
    # Calculate order parameter from beta angle
    beta = euler_angles[:, 1]
    P2 = (3 * np.cos(beta)**2 - 1) / 2
    S2 = np.mean(P2)
    
    print(f"\n  Order Parameter:")
    print(f"    S² = ⟨P₂(cos β)⟩ = {S2:.4f}")
    print(f"    Indicates molecular order/restriction")
    
    # Correlation analysis
    print(f"\n  Temporal correlation:")
    
    # Simple lag-1 autocorrelation for beta angle
    beta_norm = (beta - beta.mean()) / beta.std()
    lag1_corr = np.corrcoef(beta_norm[:-1], beta_norm[1:])[0, 1]
    
    print(f"    β lag-1 autocorr: {lag1_corr:.4f}")
    print(f"    (1.0 = perfectly correlated, 0.0 = uncorrelated)")


def main():
    """Main demonstration."""
    print("="*70)
    print("EXAMPLE: Load Pre-computed Euler Angles (Bypass Module 1)")
    print("="*70)
    print("\nThis example shows how to:")
    print("  1. Create/save Euler angles in proper format")
    print("  2. Load them directly into Module 2")
    print("  3. Bypass Module 1 completely")
    print("  4. Continue with rest of NMR analysis")
    
    # Step 1: Create example file
    # (In practice, you'd have this from your MD simulation)
    filename = create_example_euler_file(n_steps=5000)
    
    # Step 2: Load from file (bypassing Module 1)
    euler_angles = load_and_analyze(filename)
    
    # Step 3: Analyze
    analyze_euler_angles(euler_angles)
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print("\n✅ Successfully loaded Euler angles from file")
    print("✅ Module 1 was completely bypassed")
    print("✅ Ready to continue with Modules 3-9:")
    print("   - Module 3: Calculate Y_l^m spherical harmonics")
    print("   - Module 4: Autocorrelation function")
    print("   - Module 5: Spectral density")
    print("   - Module 6: T1, T2, NOE calculations")
    print("   - ... etc.")
    
    print(f"\n{'='*70}")
    print("Workflow Comparison")
    print(f"{'='*70}")
    
    print("\nStandard workflow:")
    print("  Module 1 (Generate) → Module 2 (Convert) → Modules 3-9")
    
    print("\nDirect file input (THIS EXAMPLE):")
    print("  [Pre-computed file] → Module 2 (Load) → Modules 3-9")
    print("                          ↑")
    print("                    Start here!")
    
    print("\n" + "="*70)
    print("Use Case: When to use file input?")
    print("="*70)
    print("""
  ✓ You have MD trajectories analyzed elsewhere
  ✓ You want to analyze same trajectory multiple times
  ✓ You're using external orientation extraction tools
  ✓ You want to test with known orientation data
  ✓ You want to skip expensive trajectory generation
    """)
    
    # Cleanup
    import os
    try:
        os.remove(filename)
        print(f"  (Cleaned up example file: {filename})")
    except:
        pass


if __name__ == '__main__':
    main()
