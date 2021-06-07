import numpy as np

import torch
import torchvision as tv

from learnergy.core import Dataset

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST,
    'mnist': tv.datasets.MNIST
}


def load_dataset(name='mnist', val_split=0.2, seed=0):
    """Loads an input dataset.

    Args:
        name (str): Name of dataset to be loaded.
        val_split (float): Percentage of split for the validation set.
        seed (int): Random seed.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Defining the torch seed
    torch.manual_seed(seed)

    # Loads the training data
    train_data = DATASETS[name](root='./data', train=True, download=True,
                           transform=tv.transforms.Compose(
                               [tv.transforms.ToTensor()])
                           )

    # Loads the testing data
    test_data = DATASETS[name](root='./data', train=False, download=True,
                          transform=tv.transforms.Compose(
                              [tv.transforms.ToTensor()])
                          )

    # Calculating the number of train and validation samples
    train_size = int(train_data.data.shape[0] * (1 - val_split))
    idx = np.random.permutation(train_data.data.shape[0])

    # Creates customized datasets
    # This will fix the problem when iterating over DBN layers
    train = Dataset(train_data.data[idx[:train_size]],
                    train_data.targets[idx[:train_size]].numpy(),
                    train_data.transform)
    val = Dataset(train_data.data[idx[train_size:]],
                  train_data.targets[idx[:train_size]].numpy(),
                  train_data.transform)
    test = Dataset(test_data.data,
                   test_data.targets.numpy(),
                   test_data.transform)
    

    return train, val, test
