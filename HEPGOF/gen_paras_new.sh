#!/bin/bash
  
# Parameter values
n_values=(10 25 50 100)
B_values=(100 1000)
iterations_values=(1000)
strength_values=(3)
snull_values=('powerlaw')

# Remove existing params.txt if it exists
> params_test.txt

# powerlaw
beta_vectors=("0.1 1" "1 1" "2.5 1" "5 1")
strue_values=('powerlaw')

# Generate combinations and append to params.txt
for n in "${n_values[@]}"; do
    for B in "${B_values[@]}"; do
        for beta in "${beta_vectors[@]}"; do
                for iterations in "${iterations_values[@]}"; do
                  for strength in "${strength_values[@]}"; do
                    for strue in "${strue_values[@]}"; do
                        for snull in "${snull_values[@]}"; do
                            echo "$n $B \"$beta\" $iterations $strength $strue $snull" >> params_test.txt
                       done
                    done
                done
            done
        done
    done
done
echo "params_test.txt (powerlaw) generated with all parameter combinations."

# Parameter values
beta_vectors=("1 1" "2.5 1" "5 1")
strength_values=(3)
strue_values=('brokenpowerlaw')

# brokenpowerlaw
# Generate combinations and append to params.txt
for n in "${n_values[@]}"; do
    for B in "${B_values[@]}"; do
        for beta in "${beta_vectors[@]}"; do
                for iterations in "${iterations_values[@]}"; do
                  for strength in "${strength_values[@]}"; do
                    for strue in "${strue_values[@]}"; do
                        for snull in "${snull_values[@]}"; do
                            echo "$n $B \"$beta\" $iterations $strength $strue $snull" >> params_test.txt
                       done
                    done
                done
            done
        done
    done
done
echo "params_test.txt (brokenpowerlaw) generated with all parameter combinations."

# Parameter values
beta_vectors=("0.1 1" "5 1")
strength_values=(2)
strue_values=('spectral_line')

# spectral_line_strength2
# Generate combinations and append to params.txt
for n in "${n_values[@]}"; do
    for B in "${B_values[@]}"; do
        for beta in "${beta_vectors[@]}"; do
                for iterations in "${iterations_values[@]}"; do
                  for strength in "${strength_values[@]}"; do
                    for strue in "${strue_values[@]}"; do
                        for snull in "${snull_values[@]}"; do
                            echo "$n $B \"$beta\" $iterations $strength $strue $snull" >> params_test.txt
                       done
                    done
                done
            done
        done
    done
done
echo "params_test.txt (spectral_line_strength2) generated with all parameter combinations."

# Parameter values
beta_vectors=("1 1")
strength_values=(0.1)
strue_values=('spectral_line')

# spectral_line_strength2
# Generate combinations and append to params.txt
for n in "${n_values[@]}"; do
    for B in "${B_values[@]}"; do
        for beta in "${beta_vectors[@]}"; do
                for iterations in "${iterations_values[@]}"; do
                  for strength in "${strength_values[@]}"; do
                    for strue in "${strue_values[@]}"; do
                        for snull in "${snull_values[@]}"; do
                            echo "$n $B \"$beta\" $iterations $strength $strue $snull" >> params_test.txt
                       done
                    done
                done
            done
        done
    done
done
echo "params_test.txt (spectral_line_strength0.1) generated with all parameter combinations."

# Parameter values
beta_vectors=("2.5 1" "5 1")
strength_values=(10)
strue_values=('spectral_line')

# spectral_line_strength10
# Generate combinations and append to params.txt
for n in "${n_values[@]}"; do
    for B in "${B_values[@]}"; do
        for beta in "${beta_vectors[@]}"; do
                for iterations in "${iterations_values[@]}"; do
                  for strength in "${strength_values[@]}"; do
                    for strue in "${strue_values[@]}"; do
                        for snull in "${snull_values[@]}"; do
                            echo "$n $B \"$beta\" $iterations $strength $strue $snull" >> params_test.txt
                       done
                    done
                done
            done
        done
    done
done
echo "params_test.txt (spectral_line_strength10) generated with all parameter combinations."

