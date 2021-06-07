import utils.objects as o
from core.dbn import DBN


def reconstruction(n_layers, n_visible, batch_size, epochs, use_gpu, train, val):
    """Wraps the reconstruction task for optimization purposes.

    Args:
        n_layers (int): Number of layers.
        n_visible (int): Number of visible units.
        batch_size (int): Amount of samples per batch.
        epochs (int): Number of training epochs.
        use_gpu (boolean): Whether GPU should be used or not. 
        train (Dataset): Training dataset.
        val (Dataset): Validation dataset.

    """

    def f(p):
        """Instantiates a model, gathers variables from meta-heuritics, trains and evaluates over validation data.

        Args:
            p (float): Array of variables/parameters.

        Returns:
            Mean squared error (MSE) of validation set.

        """

        # Optimization parameters
        n_hidden = tuple([int(_p) for _p in p[:n_layers]])
        steps = tuple([1] * n_layers)
        lr = tuple([float(_p) for _p in p[n_layers:n_layers*2]])
        momentum = tuple([float(_p) for _p in p[n_layers*2:n_layers*3]])
        decay = tuple([float(_p) for _p in p[n_layers*3:]])
        temperature = tuple([1] * n_layers)

        # Initializes the model
        model = DBN('bernoulli', n_visible, n_hidden, steps,
                    lr, momentum, decay, temperature, use_gpu)

        # Trains the model using the training set
        model.fit(train, batch_size, epochs)

        # Reconstructs over the validation set
        mse, _ = model.reconstruct(val)

        return mse.item()

    return f
