import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran
import os
from math import ceil
import urllib

class TorsionAnalyzer2:
    def __init__(self, file_name, adjusted_residue_index=True, res_adjust_dic=None):
        """
        Initialize the TorsionAnalyzer by loading a CSV file and optionally adjusting residue numbers.

        Parameters:
            file_name (str): Path to the CSV file to load.
            adjusted_residue_index (bool): Whether to apply residue number adjustments. Default is True.
            res_adjust_dic (dict): Dictionary for residue number adjustments. Required if adjusted_residue_index is True.
        """
        try:
            # Load the CSV file
            self.data = pd.read_csv(file_name)
            self.file_name = file_name

            # Conditionally adjust residue numbers
            if adjusted_residue_index:
                if res_adjust_dic is None:
                    raise ValueError("res_adjust_dic must be provided when adjusted_residue_index is True.")
                self.residue_adjustments = res_adjust_dic
                self._adjust_residue_numbers()
            else:
                self.residue_adjustments = None
        except FileNotFoundError:
            print(f"Error: The file {file_name} was not found.")
        except Exception as e:
            print(f"Error reading the file: {e}")

    def _adjust_residue_numbers(self):
        # Filter to only phi and psi columns and adjust residue numbers
        #valid_columns = [col for col in self.data.columns if '-psi' in col or '-phi' in col]
        #self.data = self.data[valid_columns]
        # Create an empty dictionary to store adjusted data
        adjusted_data = {}
        
        for col in self.data.columns:
            if '-psi' in col or '-phi' in col:
                # Extract chain and residue index
                chain, residue_num = col.split('-')[0].split(':')
                residue_num = int(residue_num)

                # Apply residue adjustment for the given chain
                if chain in self.residue_adjustments:
                    adjusted_residue_num = residue_num + self.residue_adjustments[chain]
                    adjusted_col = f"{chain}:{adjusted_residue_num}-{col.split('-')[1]}"
                    adjusted_data[adjusted_col] = self.data[col]
                else:
                    # Keep the original data when no adjustment is applied
                    adjusted_data[col] = self.data[col]
            else:
                # Include columns that do not match the condition
                adjusted_data[col] = self.data[col]
                

        # Convert the adjusted data back into a DataFrame
        self.data = pd.DataFrame(adjusted_data)

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

    # general purpose function to plot scatter plot for a given residue index
    def plot_scatter(self, residue_index, chains, output_dir='plots'):
        # Adjust the subplot layout for inter and intra plots
        n_rows = len(chains)
        fig, axs = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))  # Two columns: one for intra and one for inter

        # Ensure axs is 2D for consistent indexing
        if len(chains) == 1:
            axs = [axs]

        for i, chain in enumerate(chains):
            psi_col = f"{chain}:{residue_index}-psi"
            phi_col = f"{chain}:{residue_index}-phi"
            next_phi_col = f"{chain}:{residue_index + 1}-phi"

            # Plot intra-residue scatter plot (φ vs ψ)
            if psi_col in self.data.columns and phi_col in self.data.columns:
                axs[i][0].scatter(self.data[phi_col], self.data[psi_col], color='blue', alpha=0.6)
                axs[i][0].set_title(f"{chain} Chain: φ({residue_index}) vs ψ({residue_index})")
                axs[i][0].set_xlabel(f"φ({residue_index}) (phi)")
                axs[i][0].set_ylabel(f"ψ({residue_index}) (psi)")

            # Plot inter-residue scatter plot (next φ vs ψ)
            if psi_col in self.data.columns and next_phi_col in self.data.columns:
                axs[i][1].scatter(self.data[next_phi_col], self.data[psi_col], color='green', alpha=0.6)
                axs[i][1].set_title(f"{chain} Chain: φ({residue_index+1}) vs ψ({residue_index})")
                axs[i][1].set_xlabel(f"φ({residue_index+1}) (phi)")
                axs[i][1].set_ylabel(f"ψ({residue_index}) (psi)")

        # Adjust layout and save the plot
        plt.tight_layout()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = f"{output_dir}/inter_intra_torsion_scatter_{residue_index}.png"
        plt.savefig(output_file)
        print(f"Combined plot saved as {output_file}")
        plt.show()


   # general purpose function to plot scatter plot for a given residue index
    def plot_ramachandran(self, residue_index, PDB_code,PDB_chain, chain,  ax):
        """
        Plots the Ramachandran plot for a given residue and chain on the provided axis.
        
        Parameters:
            residue_index (int): Residue index.
            PDB_code (str): PDB code.
            chain (str): Chain identifier.
            #analyzer: Data analyzer object containing torsion angle data.
            ax (matplotlib.axes.Axes): Axis on which to plot.
            PDB_chain (str): Chain identifier for the PDB file 
        """
        # import MDAnalysis as mda
        # from MDAnalysis.analysis.dihedrals import Ramachandran
        #analyzer =
        # Download and load the PDB file if not present
        pdb_file = f"{PDB_code}.pdb"
        if not os.path.exists(pdb_file):
            url = f"https://files.rcsb.org/download/{PDB_code}.pdb"
            urllib.request.urlretrieve(url, pdb_file)

        # Load the structure
        u = mda.Universe(pdb_file)

        # Select the residue
        selection = f"resid {residue_index} and segid {PDB_chain}"
        atoms = u.select_atoms(selection)

        if atoms.n_atoms == 0:
            print(f"Residue {residue_index} not found in chain {PDB_chain} in PDB {PDB_code}.")
            return

        # Perform Ramachandran analysis
        R = Ramachandran(atoms).run()
        R.plot(ax=ax, color='black', ref=True)

        # Scatter plot φ and ψ angles for each chain
        psi_col = f"{chain}:{residue_index}-psi"
        phi_col = f"{chain}:{residue_index}-phi"
        if psi_col in self.data.columns and phi_col in self.data.columns:
            ax.scatter(
                self.data[phi_col], self.data[psi_col],
                color='red', alpha=0.6, linewidth=2, marker='o', label=f'Chain {chain}'
            )
        ax.set_title(f"Ramachandran Plot: Residue {residue_index}, Chain {chain}")
        ax.legend()

    def plot_all_ramachandran(self, residue_indices, PDB_code, PDB_chain, chains, output_dir='plots'):
        """
        Generate and arrange Ramachandran plots for all specified residues and chains in a single figure.
        
        Parameters:
            residue_indices (list): List of residue indices to plot.
            PDB_code (str): PDB code of the structure.
            chains (list): List of chain identifiers.
            #analyzer: Data analyzer object containing torsion angle data (optional).
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
                        self.plot_ramachandran(
                            residue_index=residue_index,
                            PDB_code=PDB_code,
                            PDB_chain=PDB_chain,
                            chain=chain,
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
