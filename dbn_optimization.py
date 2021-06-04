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
    parser = argparse.ArgumentParser(usage='Optimizes a DBN using standard meta-heuristics.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['fmnist', 'kmnist', 'mnist'])

    parser.add_argument('mh', help='Meta-heuristic identifier', choices=['hs', 'ihs'])

    parser.add_argument('-n_visible', help='Number of visible units', type=int, default=784)

    parser.add_argument('-n_layers', help='Number of DBN layers', type=int, default=3)

    # parser.add_argument('-n_hidden', help='Number of hidden units', nargs='+', type=int, default=[128, 256, 256])

    # parser.add_argument('-steps', help='Number of CD steps', nargs='+', type=int, default=[1, 1, 1])

    # parser.add_argument('-lr', help='Learning rate', nargs='+', type=float, default=[0.1, 0.1, 0.1])

    # parser.add_argument('-momentum', help='Momentum', nargs='+', type=float, default=[0, 0, 0])

    # parser.add_argument('-decay', help='Weight decay', nargs='+', type=float, default=[0, 0, 0])

    # parser.add_argument('-temperature', help='Temperature', nargs='+', type=float, default=[1, 1, 1])

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

    if n_layers != len(epochs):
        raise Exception('Number of epochs should be equal to number of layers')

    # Gathering optimization variables
    n_agents = args.n_agents
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
    # fn = t.reconstruction(visible_shape, n_channels, steps, use_gpu, batch_size, epochs, train, val)

    # Defines the variables boundaries
    # [n_hidden, lr, momentum, decay] per layer
    n_variables = n_layers * 4
    lb = [128, 0.01, 0.01, 0.01]
    ub = [256, 0.1, 0.1, 0.1]

    # Running the optimization task
    # opt.optimize(mh, fn, n_agents, n_variables, n_iterations, lb, ub, hyperparams)

