import argparse

import numpy as np
import torch

import utils.loader as l
import utils.objects as o
import utils.optimizer as opt
import utils.target as t


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Optimizes a DBN using HS-based meta-heuristics.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['fmnist', 'kmnist', 'mnist'])

    parser.add_argument('mh', help='Meta-heuristic identifier', choices=['ghs', 'goghs', 'hs', 'ihs', 'nghs', 'sghs'])

    parser.add_argument('-n_visible', help='Number of visible units', type=int, default=784)

    parser.add_argument('-n_layers', help='Number of DBN layers', type=int, default=3)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-epochs', help='Number of training epochs', nargs='+', type=int, default=[3, 3, 3])

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=10)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=15)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    parser.add_argument('--use_gpu', help='Usage of GPU', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    seed = args.seed

    # Gathering RBM-related variable
    n_visible = args.n_visible
    n_layers = args.n_layers
    batch_size = args.batch_size
    epochs = tuple(args.epochs)
    use_gpu = args.use_gpu

    # Gathering optimization variables
    n_agents = args.n_agents
    n_variables = 4 * n_layers
    n_iterations = args.n_iter
    mh_name = args.mh
    mh = o.get_mh(mh_name).obj
    hyperparams = o.get_mh(args.mh).hyperparams

    # Loads the data
    train, val, _ = l.load_dataset(name=dataset, seed=seed)

    # Defines the seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Defines the optimization task
    fn = t.reconstruction(n_layers, n_visible, batch_size, epochs, use_gpu, train, val)

    # Defines the variables boundaries
    # Each list should have `n_layers` size
    n_hidden_lb, n_hidden_ub = [128] * n_layers, [256] * n_layers
    lr_lb, lr_ub = [0.01] * n_layers, [0.1] * n_layers
    momentum_lb, momentum_ub = [0.01] * n_layers, [0.1] * n_layers
    decay_lb, decay_ub = [0.01] * n_layers, [0.1] * n_layers

    # Defines the final boundaries
    lb = n_hidden_lb + lr_lb + momentum_lb + decay_lb
    ub = n_hidden_ub + lr_ub + momentum_ub + decay_ub

    # Checks if lower and upper bound are correctly defined
    if len(lb) != n_variables or len(ub) != n_variables:
        raise Exception('Lower or upper bounds should have `n_variables`')

    # Running the optimization task
    opt.optimize(mh, fn, n_agents, n_variables, n_iterations, lb, ub, hyperparams)
