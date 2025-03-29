import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran
import os
from math import ceil 
def plot_torsion_vs_time(residue, chain, analyzer, output_dir='plots'):
    """
    Plot the torsion angles of a specific residue over time.
    
    Parameters:
        residue (int): Residue index.
        chain (str): Chain identifier.
        analyzer: Data analyzer object containing torsion angle data.
        output_dir (str): Directory to save the output plot.
    """
    # Check if the residue exists in the data
    phi_col = f"{chain}:{residue}-phi"
    psi_col = f"{chain}:{residue}-psi"
    if phi_col not in analyzer.data.columns or psi_col not in analyzer.data.columns:
        print(f"Residue {residue} in chain {chain} not found in the data.")
        return

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot φ and ψ angles against time
    ax.plot(analyzer.data['time_ps'].to_numpy(), analyzer.data[phi_col].to_numpy(), label=f'φ (phi)')
    ax.plot(analyzer.data['time_ps'].to_numpy(), analyzer.data[psi_col].to_numpy(), label=f'ψ (psi)')

    # Set labels and title
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title(f'Torsion Angles vs Time: Residue {residue}, Chain {chain}')
    ax.legend()

    # Save and display the plot
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f"{output_dir}/Torsion_vs_Time_Residue_{residue}_Chain_{chain}.png"
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")
    plt.show()

def plot_inter_torsion_vs_time(residue, chain, analyzer, output_dir='plots'):
    """
    Plot the torsion angles of a specific residue over time.
    
    Parameters:
        residue (int): Residue index.
        chain (str): Chain identifier.
        analyzer: Data analyzer object containing torsion angle data.
        output_dir (str): Directory to save the output plot.
    """
    # Check if the residue exists in the data
    phi_col = f"{chain}:{residue+1}-phi"
    psi_col = f"{chain}:{residue}-psi"
    if phi_col not in analyzer.data.columns or psi_col not in analyzer.data.columns:
        print(f"Residue {residue} in chain {chain} not found in the data.")
        return

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot φ and ψ angles against time
    ax.plot(analyzer.data['time_ps'].to_numpy(), analyzer.data[phi_col].to_numpy(), label=f'i+1 φ (phi)')
    ax.plot(analyzer.data['time_ps'].to_numpy(), analyzer.data[psi_col].to_numpy(), label=f'ψ (psi)')

    # Set labels and title
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title(f'Inter Torsion Angles vs Time: Residue {residue}, Chain {chain}')
    ax.legend()

    # Save and display the plot
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f"{output_dir}/Inter_Torsion_vs_Time_Residue_{residue}_Chain_{chain}.png"
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")
    plt.show()