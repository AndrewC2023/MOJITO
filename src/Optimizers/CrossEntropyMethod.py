"""Cross-Entropy Method (CEM) optimizer for black-box optimization.

CEM is a derivative-free optimization method particularly well-suited for:
- High-dimensional nonconvex problems
- Noisy or discontinuous cost functions
- Model Predictive Control with arbitrary cost functions

Algorithm:
1. Sample N candidates from current Gaussian distribution
2. Evaluate cost for each candidate using black-box function
3. Select top K elite samples (lowest cost)
4. Refit Gaussian mean and covariance to elite samples
5. Iterate until convergence or max iterations

References:
- Rubinstein & Kroese (2004), "The Cross-Entropy Method"
- Chua et al. (2018), "Deep Reinforcement Learning in a Handful of Trials"
"""

import numpy as np
from typing import Optional, Callable
from Optimizers.OptimizerBase import OptimizerBase


class CrossEntropyMethod(OptimizerBase):
    """Cross-Entropy Method optimizer for trajectory optimization.
    
    Parameters
    ----------
    population_size : int
        Number of samples per iteration (default: 500)
    elite_frac : float
        Fraction of samples to use as elites (default: 0.1)
    max_iterations : int
        Maximum number of CEM iterations (default: 10)
    epsilon : float
        Convergence threshold for mean change (default: 1e-3)
    alpha : float
        Smoothing parameter for mean/covariance update (default: 0.0)
        New = alpha * old + (1-alpha) * elite_fit
        Set to 0 for no smoothing (recommended for MPC)
    initial_std : float or np.ndarray
        Initial standard deviation(s) for sampling (default: 0.2)
    min_std : float
        Minimum standard deviation to prevent collapse (default: 1e-3)
    bounds : tuple of (np.ndarray, np.ndarray), optional
        (lower_bounds, upper_bounds) for decision variables
        If None, assumes normalized controls [0, 1]
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Print iteration progress (default: False)
    """
    
    def __init__(
        self,
        population_size: int = 500,
        elite_frac: float = 0.1,
        max_iterations: int = 10,
        epsilon: float = 1e-3,
        alpha: float = 0.0,
        initial_std: float = 0.2,
        min_std: float = 1e-3,
        bounds: Optional[tuple] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        self.population_size = population_size
        self.elite_frac = elite_frac
        self.num_elites = max(1, int(population_size * elite_frac))
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial_std = initial_std
        self.min_std = min_std
        self.bounds = bounds
        self.seed = seed
        self.verbose = verbose
        
        # Internal state
        self.rng = np.random.default_rng(seed)
        self.iteration_costs = []  # Track best cost per iteration
    
    def optimize(self, initial_guess: np.ndarray, controller, **kwargs) -> np.ndarray:
        """Run CEM optimization to find best decision vector.
        
        Parameters
        ----------
        initial_guess : np.ndarray
            Initial mean for sampling distribution (shape: (dim,))
        controller : NACMPC
            Controller with evaluate_decision_vector(vec) -> cost method
        **kwargs : dict
            Optional overrides for CEM parameters
            
        Returns
        -------
        np.ndarray
            Optimized decision vector (shape: (dim,))
        """
        initial_guess = np.asarray(initial_guess, dtype=float)
        dim = len(initial_guess)
        
        # Extract optional parameter overrides
        max_iters = kwargs.get('max_iterations', self.max_iterations)
        verbose = kwargs.get('verbose', self.verbose)
        
        # Initialize distribution
        mean = initial_guess.copy()
        
        # Initialize covariance (diagonal)
        if isinstance(self.initial_std, (int, float)):
            std = np.full(dim, self.initial_std)
        else:
            std = np.asarray(self.initial_std).flatten()
            if len(std) == 1:
                std = np.full(dim, std[0])
            elif len(std) != dim:
                raise ValueError(f"initial_std must be scalar or length {dim}, got {len(std)}")
        
        # Set up bounds
        if self.bounds is None:
            # Default: assume normalized controls [0, 1] for all dimensions
            lower_bounds = np.zeros(dim)
            upper_bounds = np.ones(dim)
        else:
            lower_bounds, upper_bounds = self.bounds
            lower_bounds = np.asarray(lower_bounds).flatten()
            upper_bounds = np.asarray(upper_bounds).flatten()
        
        # Reset iteration tracking
        self.iteration_costs = []
        best_solution = mean.copy()
        best_cost = float('inf')
        
        if verbose:
            print(f"\n=== Cross-Entropy Method Optimization ===")
            print(f"Decision vector dimension: {dim}")
            print(f"Population size: {self.population_size}")
            print(f"Elite samples: {self.num_elites}")
            print(f"Max iterations: {max_iters}")
        
        import time
        
        # Main CEM loop
        for iteration in range(max_iters):
            if verbose:
                print(f"\n[CEM] Iteration {iteration+1}/{max_iters} starting...")
                iter_start = time.time()
            
            # Sample population from current distribution
            if verbose:
                print(f"[CEM]   Sampling {self.population_size} candidates...")
                sample_start = time.time()
            samples = self._sample_population(mean, std, lower_bounds, upper_bounds, dim)
            if verbose:
                print(f"[CEM]   Sampling done in {time.time()-sample_start:.2f}s")
            
            # Evaluate all samples
            if verbose:
                print(f"[CEM]   Evaluating {len(samples)} samples...")
                eval_start = time.time()
            
            costs = []
            for idx, sample in enumerate(samples):
                if verbose and idx > 0 and idx % 50 == 0:
                    elapsed = time.time() - eval_start
                    rate = idx / elapsed
                    remaining = (len(samples) - idx) / rate
                    print(f"[CEM]     Progress: {idx}/{len(samples)} ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining, {rate:.1f} evals/s)")
                costs.append(controller.evaluate_decision_vector(sample))
            costs = np.array(costs)
            
            if verbose:
                eval_time = time.time() - eval_start
                print(f"[CEM]   Evaluation done in {eval_time:.2f}s ({self.population_size/eval_time:.2f} evals/s)")
            
            # Select elite samples (lowest cost)
            if verbose:
                print(f"[CEM]   Selecting top {self.num_elites} elites...")
            elite_indices = np.argsort(costs)[:self.num_elites]
            elite_samples = samples[elite_indices]
            elite_costs = costs[elite_indices]
            
            # Track best solution
            iter_best_idx = np.argmin(costs)
            iter_best_cost = costs[iter_best_idx]
            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_solution = samples[iter_best_idx].copy()
            
            self.iteration_costs.append(best_cost)
            
            if verbose:
                mean_elite_cost = np.mean(elite_costs)
                std_elite_cost = np.std(elite_costs)
                iter_time = time.time() - iter_start
                print(f"[CEM] Iter {iteration+1}/{max_iters} COMPLETE in {iter_time:.2f}s: "
                      f"Best={best_cost:.4f}, "
                      f"Elite mean={mean_elite_cost:.4f}±{std_elite_cost:.4f}, "
                      f"Mean std={np.mean(std):.4f}")
            
            # Update distribution based on elite samples
            old_mean = mean.copy()
            
            # Compute new mean and std from elites
            elite_mean = np.mean(elite_samples, axis=0)
            elite_std = np.std(elite_samples, axis=0)
            
            # Apply smoothing (if alpha > 0)
            mean = self.alpha * old_mean + (1.0 - self.alpha) * elite_mean
            std = self.alpha * std + (1.0 - self.alpha) * elite_std
            
            # Enforce minimum std to prevent premature convergence
            std = np.maximum(std, self.min_std)
            
            # Check convergence based on mean change
            mean_change = np.linalg.norm(mean - old_mean)
            if mean_change < self.epsilon:
                if verbose:
                    print(f"Converged: mean change {mean_change:.6f} < {self.epsilon}")
                break
        
        if verbose:
            print(f"Optimization complete: best cost = {best_cost:.4f}")
            print("=" * 50)
        
        return best_solution
    
    def _sample_population(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        dim: int
    ) -> np.ndarray:
        """Sample population from truncated Gaussian distribution.
        
        Uses rejection sampling to enforce bounds while maintaining
        proper distribution shape.
        
        Parameters
        ----------
        mean : np.ndarray
            Mean of sampling distribution (shape: (dim,))
        std : np.ndarray
            Standard deviation for each dimension (shape: (dim,))
        lower_bounds : np.ndarray
            Lower bounds for each dimension (shape: (dim,))
        upper_bounds : np.ndarray
            Upper bounds for each dimension (shape: (dim,))
        dim : int
            Dimension of decision vector
            
        Returns
        -------
        np.ndarray
            Sampled population (shape: (population_size, dim))
        """
        samples = []
        max_attempts = 10000  # Prevent infinite loop
        
        for _ in range(self.population_size):
            # Sample with rejection to enforce bounds
            for attempt in range(max_attempts):
                sample = self.rng.normal(mean, std)
                
                # Check bounds
                if np.all(sample >= lower_bounds) and np.all(sample <= upper_bounds):
                    samples.append(sample)
                    break
            else:
                # Fallback: clip to bounds if rejection fails
                sample = np.clip(self.rng.normal(mean, std), lower_bounds, upper_bounds)
                samples.append(sample)
        
        return np.array(samples)
    
    def get_iteration_history(self) -> np.ndarray:
        """Get best cost at each iteration.
        
        Returns
        -------
        np.ndarray
            Array of best costs (shape: (num_iterations,))
        """
        return np.array(self.iteration_costs)
