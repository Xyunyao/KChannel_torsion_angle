#!/bin/bash

# scripts to run dihearal ananlysis a a batch fashion
# input file example:
#path to cms file/kcsa_100ns_0111-out.cms 310:410 D kcsa_100ns_0111_D_torsion.cvs
#path to cms file, residue, chain ID, output file name
#run as:./run_batch_dihedral.sh inputs_file.txt 

# Check if the data list file is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <data_list>"
    exit 1
fi

# Assign the data list file from the argument
data_list="$1"

# Check if the data list file exists
if [ ! -f "$data_list" ]; then
    echo "Error: Data list file '$data_list' does not exist."
    exit 1
fi

# Loop through each line in the data list
while IFS= read -r line || [[ -n "$line" ]]; do
    # Split the line into arguments: input_file, res, chain, and output_file_name
    input_file=$(echo "$line" | awk '{print $1}')
    res=$(echo "$line" | awk '{print $2}')
    chain=$(echo "$line" | awk '{print $3}')
    output_file_name=$(echo "$line" | awk '{print $4}')

    # Check if input file exists
    if [ ! -f "$input_file" ]; then
        echo "Error: Input file '$input_file' does not exist. Skipping..."
        continue
    fi

    # Get the directory of the input file and define output path
    input_dir=$(dirname "$input_file")
    output_file_path="$input_dir/$output_file_name"

    # Check if SCHRODINGER environment variable is set
    if [ -z "$SCHRODINGER" ]; then
        echo "Error: SCHRODINGER environment variable is not set. Exiting..."
        exit 1
    fi

    # Construct the SCHRODINGER command
    command="$SCHRODINGER/run trajectory_dihedral.py -c $input_file -r $res -output_csv $output_file_path -dihedrals backbone -chain $chain -f 10"
    
    # Execute the command
    echo "Executing: $command"
    eval "$command"
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to execute command for input file '$input_file'."
    else
        echo "Command executed successfully for input file '$input_file'."
    fi
done < "$data_list"
