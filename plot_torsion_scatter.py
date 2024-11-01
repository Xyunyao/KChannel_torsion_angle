import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys

def torsion_scatter_plot(file_name, residue_index):
    try:
        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
        return
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    intra_torsion = {}
    inter_torsion = {}

    # Define residue number adjustments
    residue_adjustments = {
        'A': 21,
        'B': -82,
        'C': -185,
        'D': -288
    }

    # Scatter plots for phi and psi angles
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    chains = ['A', 'B', 'C', 'D']

    for i, chain in enumerate(chains):
        adj_index = residue_index - residue_adjustments[chain]
        psi_col = f"{chain}:{adj_index}-psi"
        phi_col = f"{chain}:{adj_index}-phi"
        next_phi_col = f"{chain}:{adj_index + 1}-phi"

        if psi_col in data.columns and phi_col in data.columns:
            axs[0, i].scatter(data[phi_col], data[psi_col], color='blue', alpha=0.6)
            axs[0, i].set_title(f"{chain} Chain: φ({residue_index}) vs ψ({residue_index})")
            axs[0, i].set_xlabel("φ (phi)")
            axs[0, i].set_ylabel("ψ (psi)")

        if psi_col in data.columns and next_phi_col in data.columns:
            axs[1, i].scatter(data[next_phi_col], data[psi_col], color='green', alpha=0.6)
            axs[1, i].set_title(f"{chain} Chain: φ({residue_index}+1) vs ψ({residue_index})")
            axs[1, i].set_xlabel(f"φ(i +1) (phi)")
            axs[1, i].set_ylabel("ψ (psi)")

    plt.tight_layout()
    plot_file = os.path.join(os.path.dirname(file_name), f"{os.path.basename(file_name).replace('.csv', '')}_scatter.png")
    plt.savefig(plot_file)
    print(f"Scatter plot saved as {plot_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <file_name> <residue_index>")
        sys.exit(1)

    file_name = sys.argv[1]
    residue_index = int(sys.argv[2])
    torsion_scatter_plot(file_name, residue_index)
