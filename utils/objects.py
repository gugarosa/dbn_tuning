from opytimizer.optimizers.evolutionary import HS, IHS


class MetaHeuristic:
    """A MetaHeuristic class to help users in selecting distinct meta-heuristics from the command line.

    """

    def __init__(self, obj, hyperparams):
        """Initialization method.

        Args:
            obj (Optimizer): An Optimizer-child instance.
            hyperparams (dict): Meta-heuristic hyperparams.

        """

        # Creates a property to hold the class itself
        self.obj = obj

        # Creates a property to hold the hyperparams
        self.hyperparams = hyperparams


# Defines a meta-heuristic dictionary constant with the possible values
MH = dict(
    hs=MetaHeuristic(HS, dict(HMCR=0.7, PAR=0.7, bw=1.0)),
    ihs=MetaHeuristic(IHS, dict(HMCR=0.7, PAR_min=0.0, PAR_max=1.0, bw_min=1.0, bw_max=10.0))
)


def get_mh(name):
    """Gets a meta-heuristic by its identifier.

    Args:
        name (str): Meta-heuristic's identifier.

    Returns:
        An instance of the MetaHeuristic class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return MH[name]

    # If object is not found
    except:
        # Raises a RuntimeError
        raise RuntimeError(
            f'Meta-heuristic {name} has not been specified yet.')
