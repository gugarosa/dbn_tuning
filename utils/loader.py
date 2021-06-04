import random

import torch
import torchvision as tv

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST,
    'mnist': tv.datasets.MNIST
}


def load_dataset(name='mnist', size=(28, 28), val_split=0.2, seed=0):
    """Loads an input dataset.

    Args:
        name (str): Name of dataset to be loaded.
        size (tuple): Height and width to be resized.
        val_split (float): Percentage of split for the validation set.
        seed (int): Random seed.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Defining the torch seed
    torch.manual_seed(seed)

    # Loads the training data
    train = DATASETS[name](root='./data', train=True, download=True,
                           transform=tv.transforms.Compose(
                               [tv.transforms.ToTensor(),
                                tv.transforms.Resize(size)])
                           )

    # Splitting the training data into training/validation
    train, val = torch.utils.data.random_split(
        train, [int(len(train) * (1 - val_split)), int(len(train) * val_split)])

    # Loads the testing data
    test = DATASETS[name](root='./data', train=False, download=True,
                          transform=tv.transforms.Compose(
                              [tv.transforms.ToTensor(),
                               tv.transforms.Resize(size)])
                          )

    return train, val, test
