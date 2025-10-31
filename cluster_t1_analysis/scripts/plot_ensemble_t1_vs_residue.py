#!/usr/bin/env python3
"""
Plot Ensemble T1 vs Residue Number

This script reads a CSV file containing ensemble-averaged T1 values
for multiple residues and creates a publication-quality plot.

Usage:
    python plot_ensemble_t1_vs_residue.py <csv_file> [output_file]

Example:
    python plot_ensemble_t1_vs_residue.py ./t1_results/ensemble_t1_summary.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_t1_vs_residue(csv_file, output_file='ensemble_t1_vs_residue.png'):
    """
    Plot ensemble T1 values vs residue number.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV file with columns: residue_number, residue_idx, chain, T1_ensemble_s
    output_file : str
        Output filename for the plot
    """
    
    # Read data
    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Remove NA values
    df = df[df['T1_ensemble_s'] != 'NA']
    df['T1_ensemble_s'] = pd.to_numeric(df['T1_ensemble_s'])
    
    print(f"Loaded {len(df)} residues with valid T1 values")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot data - convert pandas Series to numpy arrays
    ax.plot(df['residue_number'].values, df['T1_ensemble_s'].values, 
            marker='o', linestyle='-', linewidth=1.5, 
            markersize=4, color='steelblue', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Residue Number', fontsize=12)
    ax.set_ylabel('Ensemble T₁ (s)', fontsize=12)
    ax.set_title('Ensemble-Averaged T₁ Relaxation vs Residue', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics
    mean_t1 = df['T1_ensemble_s'].mean()
    std_t1 = df['T1_ensemble_s'].std()
    ax.axhline(mean_t1, color='red', linestyle='--', linewidth=1.5, 
               label=f'Mean: {mean_t1:.3f} ± {std_t1:.3f} s', alpha=0.7)
    
    # Add ±1 std deviation band
    ax.fill_between(df['residue_number'].values, 
                     mean_t1 - std_t1, 
                     mean_t1 + std_t1,
                     color='red', alpha=0.1, label='±1 σ')
    
    ax.legend(loc='best', fontsize=10)
    
    # Set reasonable y-axis limits
    y_min = max(0, df['T1_ensemble_s'].min() * 0.9)
    y_max = min(df['T1_ensemble_s'].max() * 1.1, mean_t1 + 5 * std_t1)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"{'='*60}")
    print(f"  Number of residues: {len(df)}")
    print(f"  Mean T₁: {mean_t1:.4f} s")
    print(f"  Std T₁: {std_t1:.4f} s")
    
    # Get min and max residues
    min_idx = df['T1_ensemble_s'].idxmin()
    max_idx = df['T1_ensemble_s'].idxmax()
    min_res = df.loc[min_idx, 'residue_number']
    max_res = df.loc[max_idx, 'residue_number']
    
    print(f"  Min T₁: {df['T1_ensemble_s'].min():.4f} s (residue {min_res:.0f})")
    print(f"  Max T₁: {df['T1_ensemble_s'].max():.4f} s (residue {max_res:.0f})")
    print(f"{'='*60}")
    
    # Identify outliers (> 2 std from mean)
    outliers = df[np.abs(df['T1_ensemble_s'] - mean_t1) > 2 * std_t1]
    if len(outliers) > 0:
        print(f"\nOutliers (> 2σ from mean):")
        for idx, row in outliers.iterrows():
            print(f"  Residue {row['residue_number']:.0f}: T₁ = {row['T1_ensemble_s']:.4f} s")
    
    plt.show()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: CSV file path required")
        print("Usage: python plot_ensemble_t1_vs_residue.py <csv_file> [output_file]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'ensemble_t1_vs_residue.png'
    
    try:
        plot_t1_vs_residue(csv_file, output_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
