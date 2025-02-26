import argparse

from testfunc import *


def main_func(n, B, beta, strue, snull, alpha, iters, epsilon_error=1e-5, maximum_iter=int(1e5)):

    B1=B
    B2=B
    # Convert beta from string to a list of floats
    beta_list = list(map(float, beta.split()))
    print(f"value: {beta}, type: {type(beta_list)}")


    # Example of using the parameters
    print(f"Running with parameters:")
    print(f"n = {n}, B = {B}, beta = {beta_list}, alpha = {alpha}, iterations = {iters}, strue = {strue}, snull = {snull}")

    # Define the output directory and ensure it exists
    import os
    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)

    result_timed=single_test_timed(n, B, B1, B2, beta_list, strue, snull, alpha, iters,maximum_iter,epsilon_error,loc=0.5,strength=2,width=2)

    output_results_to_file(output_dir, result_timed, n, B1, B2, beta_list, strue, snull, alpha, iters, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument('--n', type=int, required=True, help='The value of n')
    print(f"n")
    parser.add_argument('--B', type=int, required=True, help='The value of B')
    parser.add_argument('--beta', type=str, required=True, help='The beta values as a space-separated string')
    parser.add_argument('--alpha', type=float, required=True, help='The value of alpha')
    parser.add_argument('--iterations', type=int, required=True, help='The number of iterations')
    parser.add_argument('--strue', type=str, required=True, help='The value of strue')
    parser.add_argument('--snull', type=str, required=True, help='The value of snull')

    args = parser.parse_args()


    # Print out each parsed argument
    print(f"n = {args.n}")
    print(f"B = {args.B}")
    print(f"beta = {args.beta}")
    print(f"alpha = {args.alpha}")
    print(f"iterations = {args.iterations}")
    print(f"strue = {args.strue}")
    print(f"snull = {args.snull}")

    main_func(args.n, args.B, args.beta, args.strue, args.snull, args.alpha, args.iterations)

