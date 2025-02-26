#!/bin/bash
  
# Parameter values
n_values=(50 100)
B_values=(500)
beta_vectors=("1 1" "0.5 1" "0.1 1")
alpha_values=(0.05 0.1 0.01)
iterations_values=(2000)
strue_values=('exp' 'powerlaw' 'constant' 'brokenpowerlaw' 'spectral_line')
snull_values=('exp' 'powerlaw')
# Remove existing params.txt if it exists
> params_test.txt
# Generate combinations and append to params.txt
for n in "${n_values[@]}"; do
    for B in "${B_values[@]}"; do
        for beta in "${beta_vectors[@]}"; do
            for alpha in "${alpha_values[@]}"; do
                for iterations in "${iterations_values[@]}"; do
                    for strue in "${strue_values[@]}"; do
                        for snull in "${snull_values[@]}"; do
                            echo "$n $B \"$beta\" $alpha $iterations $strue $snull" >> params_test.txt
                        done
                    done
                done
            done
        done
    done
done
echo "params_test.txt generated with all parameter combinations."

