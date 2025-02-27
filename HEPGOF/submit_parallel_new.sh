#!/bin/bash

# Ensure the correct shell is being used
# If you encounter execution issues, consider running as: bash submit_parallel.sh

# SLURM directives
#SBATCH --job-name=enjoycoding
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=4-1:10:00
#SBATCH --partition=standard
#SBATCH --account=xianglei0
#SBATCH --array=1-900  # Assuming a total of 2880 combinations

# Read the specific line from params.txt
param_line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" params_new.txt)


# Use eval and set to correctly split the line into variables
# This works by causing the shell to parse the quoted string correctly,
# setting fields to their respective values.
eval set -- $param_line

n=$1
B=$2
beta=$3
iterations=$4
strength=$5
strue=$6
snull=$7

# Remove double quotes from beta
beta="${beta//\"}"

# Debug: Print parsed variables
echo "n: $n"
echo "B: $B"
echo "beta: $beta"
echo "strength: $strength"
echo "iterations: $iterations"
echo "strue: $strue"
echo "snull: $snull"

# Prepare a unique job name or identifier
beta_safe=${beta// /_}
job_name="n${n}_B${B}_beta${beta_safe}_strength${strength}_iter${iterations}_strue${strue}_snull${snull}"

# Create output directory if it doesn't exist
mkdir -p job_outputs
module load python
# Execute the main script and write output to a specific file
output_file="job_outputs/${job_name}_${SLURM_ARRAY_TASK_ID}.out"
python main_submit.py --n "$n" --B "$B" --beta "$beta" --strength "$strength" --iterations "$iterations" --strue "$strue" --snull "$snull" > "$output_file"

