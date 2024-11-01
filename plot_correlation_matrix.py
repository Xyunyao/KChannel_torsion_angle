import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

def plot_correlation_matrix(file_name):
    # Read the correlation matrix from the provided file
    correlation_matrix = pd.read_csv(file_name, header=None)
    
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
    
    # Set title and labels
    plt.title("Correlation Matrix Heatmap")
    plt.xlabel("Residues")
    plt.ylabel("Residues")

    # Save the plot in the same folder as the input file
    output_file = os.path.join(os.path.dirname(file_name), 'correlation_matrix_heatmap.png')
    plt.savefig(output_file)
    
    # Show plot
    plt.show()
    print(f"Plot saved as {output_file}")


# Example usage: file name passed as a command-line argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_correlation_matrix.py <file_name>")
    else:
        plot_correlation_matrix(sys.argv[1])
