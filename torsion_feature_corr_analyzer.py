import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import re

def analyze_features(analyzer, metric_type='intra', res_sep=1, phi_coeff=1, psi_coeff=1):
    '''
    Calculate a metric (type: intra - phi and psi from the same residue; 
    type: inter - phi from (i+res_sep), psi from (i)).
    The metric formula is phi_coeff * phi + psi_coeff * psi.
    Then calculate the correlation matrix between residues based on this metric.
    plot correlation matrix

    Parameters:
    - analyzer: data source with psi and phi angle columns
    - metric_type: 'intra' for same residue phi-psi, 'inter' for phi (i+res_sep) and psi(i)
    - res_sep: separation distance for inter-residue calculations
    - phi_coeff: coefficient applied to phi angles in the metric
    - psi_coeff: coefficient applied to psi angles in the metric
    '''

    # Initialize a dictionary to store calculated values
    residue_values = {}

    # Get all unique residues from the column names
    residues = sorted(set(col.split('-')[0] for col in analyzer.data.columns if '-psi' in col or '-phi' in col))

    # Calculate metric values for each residue
    for residue in residues:
        psi_col = f"{residue}-psi"
        
        if metric_type == 'intra':
            phi_col = f"{residue}-phi"
            
            if psi_col in analyzer.data.columns and phi_col in analyzer.data.columns:
                try:
                    # Calculate intra-residue metric
                    residue_values[residue] = phi_coeff * analyzer.data[phi_col] + psi_coeff * analyzer.data[psi_col]
                except Exception as e:
                    print(f"Error calculating value for residue {residue}: {e}")
            else:
                print(f"Warning: Missing psi or phi column for residue {residue}.")

        elif metric_type == 'inter':
            # Define the other residue with separation of res_sep
            other_residue = f"{residue.split(':')[0]}:{int(residue.split(':')[1]) + res_sep}"
            other_phi_col = f"{other_residue}-phi"
            
            if psi_col in analyzer.data.columns and other_phi_col in analyzer.data.columns:
                try:
                    # Calculate inter-residue metric
                    residue_values[residue] = phi_coeff * analyzer.data[other_phi_col] + psi_coeff * analyzer.data[psi_col]
                except Exception as e:
                    print(f"Error calculating value for residue {residue}: {e}")
            else:
                print(f"Warning: Missing psi or phi column for residue {residue}.")
    
    # Convert the dictionary to a DataFrame
    values_df = pd.DataFrame(residue_values)

    # Initialize an empty matrix for Pearson correlations
    n = len(residues)
    correlation_matrix = np.zeros((n, n))

    # Calculate pairwise Pearson correlations
    for i in range(n):
        for j in range(n):
            try:
                # Calculate the Pearson correlation for each pair of residues
                correlation_matrix[i, j] = np.corrcoef(values_df[residues[i]], values_df[residues[j]])[0, 1]
            except KeyError as e:
                print(f"KeyError for residues {residues[i]} and {residues[j]}: {e}")
                correlation_matrix[i, j] = np.nan  # Assign NaN if there is a missing key
            except Exception as e:
                print(f"Error calculating correlation for residues {residues[i]} and {residues[j]}: {e}")
                correlation_matrix[i, j] = np.nan  # Assign NaN if another error occurs

    # Convert the correlation matrix to a DataFrame
    correlation_df = pd.DataFrame(correlation_matrix, index=residues, columns=residues)

    # Adjust column index names
    adjusted_residues = [int(res.split(':')[1]) for res in residues]
    correlation_df.index = adjusted_residues
    correlation_df.columns = adjusted_residues

    # Rerank and reorder the correlation matrix by adjusted residue numbers
    sorted_residues = sorted(adjusted_residues)
    correlation_df = correlation_df.loc[sorted_residues, sorted_residues]

    # Save the correlation matrix to a CSV file
    output_file = os.path.join(
        os.path.dirname(analyzer.file_name), 
        f"{os.path.basename(analyzer.file_name).rstrip('.csv')}_corr_mtx_{metric_type}_{phi_coeff}phi_{psi_coeff}psi.csv"
    )
    try:
        correlation_df.to_csv(output_file)
        print(f"Correlation matrix with adjusted and sorted residue numbers saved as {output_file}")
    except Exception as e:
        print(f"Error saving correlation matrix to file: {e}")

    # plot the correlation_matrix
    correlation_matrix = correlation_df
    
    # Plot the heatmap with fixed value range [-1, 1] and custom labels
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix,  cmap='coolwarm', vmin=-1, vmax=1, 
                square=True, cbar_kws={'shrink': .8})
    
    # Set x and y axis labels starting at 22
    num_features = correlation_matrix.shape[0]
    ticks = range(num_features)
    labels = [22 + i for i in ticks]
    plt.xticks(ticks=[i for i in range(0, num_features, 4)], 
               labels=[labels[i] for i in range(0, num_features, 4)])
    plt.yticks(ticks=[i for i in range(0, num_features, 4)], 
               labels=[labels[i] for i in range(0, num_features, 4)])

    plt.xlabel("Residues")
    plt.ylabel("Residues")
    
    # Set title and labels
    if metric_type == 'intra':
        plt.title(f"{os.path.basename(analyzer.file_name).rstrip('.csv')}_corr_mtx_{metric_type}_{phi_coeff}phi_{psi_coeff}psi")
        output_file = os.path.join(os.path.dirname(analyzer.file_name), f"{os.path.basename(analyzer.file_name).rstrip('.csv')}_corr_mtx_{metric_type}_{phi_coeff}phi_{psi_coeff}psi.pdf")
        plt.savefig(output_file)
    elif metric_type == 'inter':
        plt.title(f"{os.path.basename(analyzer.file_name).rstrip('.csv')}_corr_mtx_{metric_type}{res_sep}_{phi_coeff}phi_{psi_coeff}psi")
        output_file = os.path.join(os.path.dirname(analyzer.file_name), f"{os.path.basename(analyzer.file_name).rstrip('.csv')}_corr_mtx_{metric_type}{res_sep}_{phi_coeff}phi_{psi_coeff}psi.pdf")
        plt.savefig(output_file)
    
    # Show plot
    plt.show()
    print(f"Plot saved as {output_file}")

    # After calculating correlations, plot and save the correlation graphs
    #plot_correlation_i_i1_i2(correlation_df, file_name)
    return values_df



def analyze_features_in_windows(analyzer, metric_type='intra', res_sep=1, phi_coeff=1, psi_coeff=1, row_range=None):
    '''
    Calculate a metric based on phi and psi angles from specified rows within each column range.
    Then calculate and plot the correlation matrix based on this metric.

    Parameters:
    - analyzer: data source with psi and phi angle columns
    - metric_type: 'intra' for same-residue phi-psi, 'inter' for phi (i+res_sep) and psi(i)
    - res_sep: separation distance for inter-residue calculations
    - phi_coeff: coefficient applied to phi angles in the metric
    - psi_coeff: coefficient applied to psi angles in the metric
    - row_range: tuple (start, end) specifying row indices to include in calculations
    '''

    # Initialize a dictionary to store calculated values
    residue_values = {}

    # Get all unique residues from the column names
    residues = sorted(set(col.split('-')[0] for col in analyzer.data.columns if '-psi' in col or '-phi' in col))

    # Define row selection based on row_range
    start, end = row_range if row_range else (0, len(analyzer.data))

    # Calculate metric values for each residue
    for residue in residues:
        psi_col = f"{residue}-psi"
        
        if metric_type == 'intra':
            phi_col = f"{residue}-phi"
            
            if psi_col in analyzer.data.columns and phi_col in analyzer.data.columns:
                try:
                    # Calculate intra-residue metric using specified row range
                    psi_data = analyzer.data[psi_col].iloc[start:end]
                    phi_data = analyzer.data[phi_col].iloc[start:end]
                    residue_values[residue] = phi_coeff * phi_data + psi_coeff * psi_data
                except Exception as e:
                    print(f"Error calculating value for residue {residue}: {e}")
            else:
                print(f"Warning: Missing psi or phi column for residue {residue}.")

        elif metric_type == 'inter':
            other_residue = f"{residue.split(':')[0]}:{int(residue.split(':')[1]) + res_sep}"
            other_phi_col = f"{other_residue}-phi"
            
            if psi_col in analyzer.data.columns and other_phi_col in analyzer.data.columns:
                try:
                    psi_data = analyzer.data[psi_col].iloc[start:end]
                    phi_data = analyzer.data[other_phi_col].iloc[start:end]
                    residue_values[residue] = phi_coeff * phi_data + psi_coeff * psi_data
                except Exception as e:
                    print(f"Error calculating value for residue {residue}: {e}")
            else:
                print(f"Warning: Missing psi or phi column for residue {residue}.")

    # Convert the dictionary to a DataFrame
    values_df = pd.DataFrame(residue_values)

    # Calculate the correlation matrix
    correlation_matrix = values_df.corr()

    # Sort residues by residue number
    def extract_residue_number(residue):
        return int(re.search(r'\d+', residue).group())

    sorted_residues = sorted(residues, key=extract_residue_number)
    correlation_matrix = correlation_matrix.reindex(index=sorted_residues, columns=sorted_residues)

    # Plot the heatmap with fixed value range [-1, 1]
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, square=True, cbar_kws={'shrink': .8})

    plt.xticks(rotation=90)  # Rotate x-axis labels if needed
    plt.xlabel("Residues")
    plt.ylabel("Residues")

    # Define the output filename based on parameters
    plot_filename = f"{os.path.basename(analyzer.file_name).rstrip('.csv')}_corr_mtx_{metric_type}_{phi_coeff}phi_{psi_coeff}psi_range{start}-{end}"
    if metric_type == 'inter':
        plot_filename += f"_{res_sep}"
    
    # Save the plot
    output_file = os.path.join(os.path.dirname(analyzer.file_name), f"{plot_filename}.pdf")
    plt.title(plot_filename.replace('_', ' '))  # Format title for better readability
    plt.savefig(output_file)
    plt.show()
    print(f"Plot saved as {output_file}")

    return values_df

