import numpy as np
from  gncpy.dynamics.basic import DynamicsBase
from Optimizers.OptimizerBase import OptimizerBase, CostFunction
""" what does this need?, needs a vehicle model to step, need a cost function to evaluate as it steps, and it needs an optimizer to optimize the control inputs?  """

class NACMPC:
    """Nonlinear Arbitrary Cost Model Predictive Controller (NAC-MPC)."""
    def __init__(self, model: DynamicsBase, costFunction: CostFunction, optimizer: OptimizerBase):
        self.model = model
        self.costFunction = costFunction
        self.optimizer = optimizer

        self.physics_dt = 0.01  # Default physics time step
        self.Control_Steps = 10 # defualt number of time steps that a control input is applied for


    def plan(self, x0, u0, goal):
        """Plan control inputs using NAC-MPC approach.
        Parameters
        ----------
        x0 : np.ndarray
            Initial state.
        u0 : np.ndarray
            Initial control input guess.
        goal : np.ndarray
            Desired goal state.
            
        Returns
        -------
        np.ndarray
            Optimized control inputs.
        """

        # Optimizer will return some list of control inputs





        # evaluate the cost
        



        