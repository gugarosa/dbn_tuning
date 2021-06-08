import argparse
import pickle

import numpy as np
import torch

import utils.loader as l
from core.dbn import DBN


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Evaluates a fine-tuned DBN.')

    parser.add_argument('history', help='History file identifier', type=str)

    parser.add_argument('dataset', help='Dataset identifier', choices=['fmnist', 'kmnist', 'mnist'])

    parser.add_argument('-n_visible', help='Number of visible units', type=int, default=784)

    parser.add_argument('-n_layers', help='Number of DBN layers', type=int, default=3)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-epochs', help='Number of training epochs', nargs='+', type=int, default=[3, 3, 3])

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    parser.add_argument('--use_gpu', help='Usage of GPU', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    history = args.history
    dataset = args.dataset
    seed = args.seed

    # Gathering RBM-related variable
    n_visible = args.n_visible
    n_layers = args.n_layers
    batch_size = args.batch_size
    epochs = tuple(args.epochs)
    use_gpu = args.use_gpu

    # Loads the data
    train, _, test = l.load_dataset(name=dataset, seed=seed)

    # Defines the seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Loads the history file
    with open(history, "rb") as input_file:
        # Loads object from file
        h = pickle.load(input_file)

    # Gathers the best parameters
    p = h.best_agent[-1][0]

    # Fine-tuned parameters
    n_hidden = tuple([int(_p[0]) for _p in p[:n_layers]])
    steps = tuple([1] * n_layers)
    lr = tuple([float(_p[0]) for _p in p[n_layers:n_layers*2]])
    momentum = tuple([float(_p[0]) for _p in p[n_layers*2:n_layers*3]])
    decay = tuple([float(_p[0]) for _p in p[n_layers*3:]])
    temperature = tuple([1] * n_layers)

    # Initializes the model
    model = DBN('bernoulli', n_visible, n_hidden, steps,
                lr, momentum, decay, temperature, use_gpu)

    # Trains the model using the training set
    model.fit(train, batch_size, epochs)

    # Reconstructs over the testing set
    mse, _ = model.reconstruct(test)

    # Outputs the MSE to a file
    with open(history + '.txt', 'w') as output_file:
        output_file.write(str(mse.detach().numpy().item()))
