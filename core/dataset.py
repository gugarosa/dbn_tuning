"""Dataset-related classes.
"""

import learnergy.utils.exception as e
import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    """A custom dataset class, inherited from PyTorch's dataset.

    """

    def __init__(self, data, targets, transform=None):
        """Initialization method.

        Args:
            data (np.array): An n-dimensional array containing the data.
            targets (np.array): An 1-dimensional array containing the data's labels.
            transform (callable): Optional transform to be applied over a sample.

        """

        # Samples array
        self.data = data

        # Labels array
        self.targets = targets

        # Transform callable
        self.transform = transform

    @property
    def data(self):
        """np.array: An n-dimensional array containing the data.

        """

        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            raise e.TypeError('`data` should be a numpy array or a tensor')

        self._data = data

    @property
    def targets(self):
        """np.array: An 1-dimensional array containing the data's labels.

        """

        return self._targets

    @targets.setter
    def targets(self, targets):
        if not isinstance(targets, np.ndarray):
            raise e.TypeError('`targets` should be a numpy array')

        self._targets = targets

    @property
    def transform(self):
        """callable: Optional transform to be applied over a sample.

        """

        return self._transform

    @transform.setter
    def transform(self, transform):
        if not (hasattr(transform, '__call__') or transform is None):
            raise e.TypeError('`transform` should be a callable or None')

        self._transform = transform

    def __getitem__(self, idx):
        """A private method that will be the base for PyTorch's iterator getting a new sample.

        Args:
            idx (int): The idx of desired sample.

        """

        # Gets a sample and its label based on its index
        x = self.data[idx]
        y = self.targets[idx]

        # If there is any transform to be applied
        if self.transform:
            # Applies the transform
            x = self.transform(x)

        return x, y

    def __len__(self):
        """A private method that will be the base for PyTorch's iterator getting dataset's length.

        """

        return len(self.data)
