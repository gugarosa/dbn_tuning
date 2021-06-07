import argparse
import glob
import pickle

import numpy as np
import opytimizer.visualization.convergence as c
from natsort import natsorted


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Analyzes an RBM optimization convergence.')

    parser.add_argument('input_files', help='General input history file (without seed)', type=str, nargs='+')

    parser.add_argument('variable_index', help='Variable index', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    input_files = args.input_files
    variable_index = args.variable_index

    # Instantiates a list of overall means
    input_files_means_pos = []
    input_files_means_fit = []

    # Iterates through all input files
    for i, input_file in enumerate(input_files):
        # Checks the folder and creates a list of model files
        input_file_with_seeds = natsorted(glob.glob(f'{input_file}*.pkl'))

        # Instantiates a list of seeded files means
        input_file_with_seeds_values_pos = []
        input_file_with_seeds_values_fit = []

        # Iterates over every seeded file
        for input_file_with_seed in input_file_with_seeds:
            # Loads the history file
            with open(input_file_with_seed, "rb") as input_file:
                # Loads object from file
                h = pickle.load(input_file)

            # Gathers the bets agent's fitness and appends to seeded list
            best_agent_pos, best_agent_fit = h.get_convergence('best_agent')
            input_file_with_seeds_values_pos.append(best_agent_pos)
            input_file_with_seeds_values_fit.append(best_agent_fit)

        # Calculates the mean and appends to overall
        mean_pos = np.mean(input_file_with_seeds_values_pos, axis=0)[variable_index]
        mean_fit = np.mean(input_file_with_seeds_values_fit, axis=0)
        input_files_means_pos.append(mean_pos)
        input_files_means_fit.append(mean_fit)

    # Plots the convergence of best variables and fitnesses
    c.plot(*input_files_means_pos)
    c.plot(*input_files_means_fit)
