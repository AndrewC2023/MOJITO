import numpy as np


class CostFunction:
    """Base class for cost functions used in optimization.

    The cost function is intentionally agnostic to the particular optimizer.
    It is called by :class:`NBBMPC` during trajectory rollout and can use
    arbitrary information about the state, control, time, configuration space,
    etc. via ``**kwargs``.
    """

    def evaluate(self, state: np.ndarray, control: np.ndarray, dt: float, **kwargs) -> float:
        """Evaluate instantaneous cost at a given state, control and time.

        Parameters
        ----------
        state : np.ndarray
            Current system state at time ``t``.
        control : np.ndarray
            Current control input at time ``t`` (normalized 0-1 as per project
            convention).
        dt : float
            The time step duration.
        **kwargs : dict
            Extra context (e.g. goal state, configuration space, penalties).

        Returns
        -------
        float
            Scalar cost contribution at this instant.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class OptimizerBase:
    """Base class for optimizers used with NBBMPC.

    The optimizer treats the controller as a black-box cost oracle. It does
    **not** know about dynamics or constraints directly; instead it proposes a
    candidate decision vector (keyframe values, and optionally horizon) and
    asks the controller to return the scalar cost.
    """

    def optimize(self, initial_guess: np.ndarray, controller, **kwargs) -> np.ndarray:
        """Run the optimization procedure.

        Parameters
        ----------
        initial_guess : np.ndarray
            Initial decision vector (e.g. concatenated keyframe values and, if
            enabled, horizon).
        controller : object
            NBBMPC-like object exposing ``evaluate_decision_vector(vec, **kw)``
            that returns a scalar cost for a proposed decision vector.
        **kwargs : dict
            Extra options for the optimizer (max iterations, seeds, etc.).

        Returns
        -------
        np.ndarray
            Optimized decision vector.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

