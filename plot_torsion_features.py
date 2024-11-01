import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def calculate_and_plot_phi_minus_2psi(file_name):
    try:
        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
        return
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Define residue number adjustments
    residue_adjustments = {
        'A': 21,
        'B': -82,
        'C': -185,
        'D': -288
    }

    # Collect data for phi - 2 * psi
    residues = []
    phi_minus_2psi_values = []

    try:
        # Filter out any non-psi and non-phi columns
        valid_columns = [col for col in data.columns if '-psi' in col or '-phi' in col]
        data = data[valid_columns]

        # Extract residue identifiers (like A:1, B:2, etc.)
        residue_ids = sorted(set(col.split('-')[0] for col in data.columns if col[0] in ['A', 'B', 'C', 'D']))

        for residue in residue_ids:
            psi_col = f"{residue}-psi"
            phi_col = f"{residue}-phi"

            # Check if both phi and psi columns exist for this residue
            if psi_col in data.columns and phi_col in data.columns:
                # Calculate phi - 2 * psi for each entry in these columns
                combination_values = data[phi_col] - 2 * data[psi_col]
                
                # Adjust residue number and store the calculated values
                adjustment = residue_adjustments.get(residue[0], 0)
                adjusted_residue_index = int(residue.split(':')[1]) + adjustment
                residues.extend([adjusted_residue_index] * len(combination_values))
                phi_minus_2psi_values.extend(combination_values)

    except Exception as e:
        print(f"Error during processing: {e}")
        return

    # Plotting phi - 2 * psi against adjusted residue number
    plt.figure(figsize=(10, 6))
    plt.scatter(residues, phi_minus_2psi_values, alpha=0.5, color='teal', edgecolors='k')
    plt.title(r'Plot of $\phi - 2 \psi$ vs Adjusted Residue Number')
    plt.xlabel('Residue Number (Adjusted)')
    plt.ylabel(r'$\phi - 2 \psi$')
    plt.grid(True)

    # Create the plot file name based on the input file name
    base_name = os.path.basename(file_name).replace('.csv', '')
    plot_file = os.path.join(os.path.dirname(file_name), f"{base_name}_phi_minus_2psi.png")

    # Save and show the plot
    plt.savefig(plot_file)
    print(f"Plot saved as {plot_file}")
    plt.show()

if __name__ == "__main__":
    # Check if the file name is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_name>")
        sys.exit(1)

    file_name = sys.argv[1]
    calculate_and_plot_phi_minus_2psi(file_name)
