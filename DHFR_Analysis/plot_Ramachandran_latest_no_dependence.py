
# functions can be used independently
# according fucntions have been incoporated into TorsionAnaylzer class
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran
import os
from math import ceil


def plot_ramachandran(residue_index, PDB_code, chain, analyzer, ax):
    """
    Plots the Ramachandran plot for a given residue and chain on the provided axis.
    
    Parameters:
        residue_index (int): Residue index.
        PDB_code (str): PDB code.
        chain (str): Chain identifier.
        analyzer: Data analyzer object containing torsion angle data.
        ax (matplotlib.axes.Axes): Axis on which to plot.
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis.dihedrals import Ramachandran

    # Download and load the PDB file if not present
    pdb_file = f"{PDB_code}.pdb"
    if not os.path.exists(pdb_file):
        url = f"https://files.rcsb.org/download/{PDB_code}.pdb"
        urllib.request.urlretrieve(url, pdb_file)

    # Load the structure
    u = mda.Universe(pdb_file)

    # Select the residue
    selection = f"resid {residue_index} and segid {chain}"
    atoms = u.select_atoms(selection)

    if atoms.n_atoms == 0:
        print(f"Residue {residue_index} not found in chain {chain}")
        return

    # Perform Ramachandran analysis
    R = Ramachandran(atoms).run()
    R.plot(ax=ax, color='black', ref=True)

    # Scatter plot φ and ψ angles for each chain
    psi_col = f"{chain}:{residue_index}-psi"
    phi_col = f"{chain}:{residue_index}-phi"
    if psi_col in analyzer.data.columns and phi_col in analyzer.data.columns:
        ax.scatter(
            analyzer.data[phi_col], analyzer.data[psi_col],
            color='red', alpha=0.6, linewidth=2, marker='o', label=f'Chain {chain}'
        )
    ax.set_title(f"Ramachandran Plot: Residue {residue_index}, Chain {chain}")
    ax.legend()


def plot_all_ramachandran(residue_indices, PDB_code, chains, analyzer=None, output_dir='plots'):
    """
    Generate and arrange Ramachandran plots for all specified residues and chains in a single figure.
    
    Parameters:
        residue_indices (list): List of residue indices to plot.
        PDB_code (str): PDB code of the structure.
        chains (list): List of chain identifiers.
        analyzer: Data analyzer object containing torsion angle data (optional).
        output_dir (str): Directory to save the output plot.
    """
    # Determine number of subplots and layout
    num_plots = len(residue_indices) * len(chains)
    cols = 3  # Define number of columns
    rows = ceil(num_plots / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()  # Flatten to iterate easily if multiple rows
    
    plot_idx = 0
    for chain in chains:
        for residue_index in residue_indices:
            if plot_idx < len(axs):
                ax = axs[plot_idx]
                try:
                    # Call the Ramachandran plot function with the current axis
                    plot_ramachandran(
                        residue_index=residue_index,
                        PDB_code=PDB_code,
                        chain=chain,
                        analyzer=analyzer,
                        ax=ax
                    )
                except Exception as e:
                    print(f"Error plotting residue {residue_index} in chain {chain}: {e}")
                plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axs)):
        fig.delaxes(axs[i])

    # Save and display the combined figure
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f"{output_dir}/Combined_Ramachandran_{PDB_code}.png"
    plt.savefig(output_file)
    print(f"Combined plot saved as {output_file}")
    plt.show()
