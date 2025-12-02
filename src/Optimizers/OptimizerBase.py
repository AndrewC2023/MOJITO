import numpy as np

class CostFunction:
    """Base class for cost functions used in optimization."""
    def evaluate(self, point: np.ndarray, **kwargs):
        """Evaluate the cost given a state and control input.
        
        Parameters
        ----------
        point : (np.ndarray): 
            the point at which to evaluate the cost., this could be a control input or value. 
            This is heavily dependent on the implementation.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class OptimizerBase:
    """Base class for optimizers."""
    def optimizeStep(self, cost_function: CostFunction, initial_guess: np.ndarray, **kwargs):
        """Optimize the given cost function starting from an initial guess.
        
        Parameters
        ----------
        cost_function : CostFunction
            The cost function to minimize.
        initial_guess : np.ndarray
            The starting point for the optimization.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")