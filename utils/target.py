from learnergy.models.deep import DBN

import utils.objects as o


def reconstruction(n_visible, batch_size, epochs, use_gpu, train, val):
    """Wraps the reconstruction task for optimization purposes.

    Args:
        n_visible (int): Number of visible units.
        n_hidden (int): Number of hidden units.
        steps (tuple): Number of Gibbs' sampling steps.
        lr (tuple): Learning rate
        momentum (tuple): Momentum.
        decay (tuple): Weight decay.
        temperature (tuple): Temperature.
        batch_size (int): Amount of samples per batch.
        epochs (int): Number of training epochs.
        use_gpu (boolean): Whether GPU should be used or not. 
        train (torchtext.data.Dataset): Training dataset.
        val (torchtext.data.Dataset): Validation dataset.

    """

    def f(p):
        """Instantiates a model, gathers variables from meta-heuritics, trains and evaluates over validation data.

        Args:
            p (float): Array of variables/parameters.

        Returns:
            Mean squared error (MSE) of validation set.

        """

        # Optimization parameters
        n_hidden = ()
        steps = ()
        lr = ()
        momentum = ()
        decay = ()

        # Initializes the model
        model = DBN('bernoulli', n_visible, n_hidden, steps,
                    lr, momentum, decay, temperature, use_gpu)

        # Trains the model using the training set
        model.fit(train, batch_size, epochs)

        # Reconstructs over the validation set
        mse, _ = model.reconstruct(val)

        return mse.item()

    return f
