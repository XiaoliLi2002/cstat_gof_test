import argparse

from testfunc import *


def main_func(n,beta,strue,snull,B=1000,iters=1000,epsilon=1e-5,loc=0.5,strength=10,width=5):
    width=n*0.1 #width here
    # Convert beta from string to a list of floats
    beta_list = list(map(float, beta.split()))
    print(f"value: {beta}, type: {type(beta_list)}")

    # Example of using the parameters
    print(f"Running with parameters:")
    print(f"n = {n}, B = {B}, beta = {beta_list}, strength = {strength}, iterations = {iters}, strue = {strue}, snull = {snull}")

    single_test_timed_run_and_save_no_doubleB(n=n,beta=beta,strue=strue,snull=snull,B=B,iters=iters,epsilon=epsilon,loc=loc,width=width)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument('--n', type=int, required=True, help='The value of n')
    print(f"n")
    parser.add_argument('--B', type=int, required=True, help='The value of B')
    parser.add_argument('--beta', type=str, required=True, help='The beta values as a space-separated string')
    parser.add_argument('--strength', type=float, required=True, help='The value of strength')
    parser.add_argument('--iterations', type=int, required=True, help='The number of iterations')
    parser.add_argument('--strue', type=str, required=True, help='The value of strue')
    parser.add_argument('--snull', type=str, required=True, help='The value of snull')

    args = parser.parse_args()


    # Print out each parsed argument
    print(f"n = {args.n}")
    print(f"B = {args.B}")
    print(f"beta = {args.beta}")
    print(f"strength = {args.strength}")
    print(f"iterations = {args.iterations}")
    print(f"strue = {args.strue}")
    print(f"snull = {args.snull}")

    main_func(n=args.n, B=args.B, beta=args.beta, strue=args.strue, snull=args.snull, strength=args.strength, iters=args.iterations)

