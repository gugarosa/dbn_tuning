from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace


def optimize(opt, target, n_agents, n_variables, n_iterations, lb, ub, hyperparams):
    """Abstracts all Opytimizer's mechanisms into a single method.

    Args:
        opt (Optimizer): An Optimizer-child class.
        target (callable): The method to be optimized.
        n_agents (int): Number of agents.
        n_variables (int): Number of variables.
        n_iterations (int): Number of iterations.
        lb (list): List of lower bounds.
        ub (list): List of upper bounds.
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        A History object containing all optimization's information.

    """

    # Creates space, optimizer and function
    space = SearchSpace(n_agents, n_variables, lb, ub)
    optimizer = opt(hyperparams)
    function = Function(target)

    # Bundles every piece into Opytimizer class
    task = Opytimizer(space, optimizer, function)

    # Initializes the task
    task.start(n_iterations)

    # Dumps the object to file
    file_path = f'{opt.algorithm}_{n_agents}ag_{n_variables}var_{n_iterations}it.pkl'
    task.save(file_path)