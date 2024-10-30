import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

class TorsionAnalyzer:
    def __init__(self, file_name):
        # Load the CSV file
        try:
            self.data = pd.read_csv(file_name)
            self.file_name = file_name
            self.residue_adjustments = {'A': 21, 'B': -82, 'C': -185, 'D': -288}
            self._adjust_residue_numbers()
        except FileNotFoundError:
            print(f"Error: The file {file_name} was not found.")
        except Exception as e:
            print(f"Error reading the file: {e}")

    def _adjust_residue_numbers(self):
        # Filter to only phi and psi columns and adjust residue numbers
        valid_columns = [col for col in self.data.columns if '-psi' in col or '-phi' in col]
        self.data = self.data[valid_columns]

    def calculate_intra_residue_correlation(self):
        intra_torsion = {}
        for col in self.data.columns:
            if '-psi' in col:
                residue = col.split('-')[0]
                psi_col = col
                phi_col = psi_col.replace('-psi', '-phi')
                if phi_col in self.data.columns:
                    correlation = np.corrcoef(self.data[phi_col], self.data[psi_col])[0, 1]
                    intra_torsion[residue] = correlation
                else:
                    print(f"Warning: Missing phi column for residue {residue}.")
        return intra_torsion

    def calculate_inter_residue_correlation(self):
        inter_torsion = {}
        for col in self.data.columns:
            if '-psi' in col:
                residue = col.split('-')[0]
                psi_col = col
                next_residue = f"{residue[0]}:{int(residue.split(':')[1]) + 1}"
                next_phi_col = f"{next_residue}-phi"
                if next_phi_col in self.data.columns:
                    correlation = np.corrcoef(self.data[psi_col], self.data[next_phi_col])[0, 1]
                    inter_torsion[residue] = correlation
                else:
                    print(f"Warning: Missing phi column for next residue {next_residue}.")
        return inter_torsion

    def plot_scatter(self, residue_index):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        chains = ['A', 'B', 'C', 'D']

        for i, chain in enumerate(chains):
            adjusted_index = residue_index + self.residue_adjustments[chain]
            psi_col = f"{chain}:{residue_index}-psi"
            phi_col = f"{chain}:{residue_index}-phi"
            next_phi_col = f"{chain}:{residue_index + 1}-phi"

            if psi_col in self.data.columns and phi_col in self.data.columns:
                axs[0, i].scatter(self.data[phi_col], self.data[psi_col], color='blue', alpha=0.6)
                axs[0, i].set_title(f"{chain} Chain: φ({residue_index}) vs ψ({residue_index})")
                axs[0, i].set_xlabel(f"φ({residue_index}) (phi)")
                axs[0, i].set_ylabel(f"ψ({residue_index}) (psi)")

            if psi_col in self.data.columns and next_phi_col in self.data.columns:
                axs[1, i].scatter(self.data[next_phi_col], self.data[psi_col], color='green', alpha=0.6)
                axs[1, i].set_title(f"{chain} Chain: φ({residue_index+1}) vs ψ({residue_index})")
                axs[1, i].set_xlabel(f"φ({residue_index+1}) (phi)")
                axs[1, i].set_ylabel(f"ψ({residue_index}) (psi)")

        plt.tight_layout()
        plot_file = os.path.join(
    os.path.dirname(self.file_name), 
    f"{os.path.basename(self.file_name).replace('.csv', '')}_residue_{residue_index}_scatter.png"
)

        plt.savefig(plot_file)
        print(f"Scatter plot saved as {plot_file}")
        plt.show()
