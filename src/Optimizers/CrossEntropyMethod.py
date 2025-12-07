"""Cross-Entropy Method (CEM) optimizer for black-box optimization.

CEM is a derivative-free optimization method particularly well-suited for:
- High-dimensional nonconvex problems
- Noisy or discontinuous cost functions

Algorithm:
1. Sample N candidates from current Gaussian distribution
2. Evaluate cost for each candidate using black-box function
3. Select top K elite samples (lowest cost)
4. Refit Gaussian mean and covariance to elite samples
5. Iterate until convergence or max iterations


The Best Reference I could fine for CEM is this book: Rubinstein & Kroese, "The Cross-Entropy Method"
Theres an MIT tutorial as well that I reference in the paper
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
    num_best_retained : int
        Number of best solutions to retain across iterations (default: 0)
        These solutions are injected into the elite set each iteration
        to prevent losing good solutions. Set to 0 to disable.
    bounds : tuple of (float/array, float/array), optional
        (lower_bounds, upper_bounds) for decision variables.
        Can be:
        - None: assumes [0, 1] for all dimensions
        - (scalar, scalar): same bounds for all dimensions
        - (array, array): per-dimension bounds (must match decision_dim)
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
        num_best_retained: int = 0,
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
        self.num_best_retained = max(0, num_best_retained)
        self.bounds = bounds
        self.seed = seed
        self.verbose = verbose
        
        # Internal state
        self.rng = np.random.default_rng(seed)
        self.iteration_costs = []  # Track best cost per iteration
        self.best_solutions_list = []  # Track (cost, solution) tuples of n best
    
    def optimize(self, initial_guess: np.ndarray, controller, **kwargs) -> np.ndarray:
        """Run CEM optimization to find best decision vector.
        
        Parameters
        ----------
        initial_guess : np.ndarray
            Initial mean for sampling distribution (shape: (n,))
        controller : NACMPC
            Controller with evaluate_decision_vector(vec) -> cost method
        **kwargs : dict
            Optional overrides for CEM parameters
            
        Returns
        -------
        np.ndarray
            Optimized decision vector (shape: (n,))
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
        
        # Set up bounds - backwards compatible with scalar or per-dimension
        if self.bounds is None:
            # Default: assume normalized controls [0, 1] for all dimensions
            lower_bounds = np.zeros(dim)
            upper_bounds = np.ones(dim)
        else:
            lower, upper = self.bounds
            lower_bounds = np.atleast_1d(lower).flatten()
            upper_bounds = np.atleast_1d(upper).flatten()
            
            # If scalar bounds provided, broadcast to all dimensions
            if lower_bounds.size == 1:
                lower_bounds = np.full(dim, lower_bounds[0])
            if upper_bounds.size == 1:
                upper_bounds = np.full(dim, upper_bounds[0])
            
            # Validate dimensions
            if lower_bounds.size != dim:
                raise ValueError(f"lower_bounds size {lower_bounds.size} != decision_dim {dim}")
            if upper_bounds.size != dim:
                raise ValueError(f"upper_bounds size {upper_bounds.size} != decision_dim {dim}")
        
        # Reset iteration tracking
        self.iteration_costs = []
        self.best_solutions_list = []  # Reset best solutions list
        best_solution = mean.copy()
        best_cost = float('inf')
        
        if verbose:
            print(f"\n=== Cross-Entropy Method Optimization ===", flush=True)
            print(f"Decision vector dimension: {dim}", flush=True)
            print(f"Population size: {self.population_size}", flush=True)
            print(f"Elite samples: {self.num_elites}", flush=True)
            print(f"Best solutions retained: {self.num_best_retained}", flush=True)
            print(f"Max iterations: {max_iters}", flush=True)
        
        import time
        
        # Main CEM loop
        for iteration in range(max_iters):
            if verbose:
                print(f"\n[CEM] Iteration {iteration+1}/{max_iters} starting...", flush=True)
                iter_start = time.time()
            
            # Sample population from current distribution
            eval_start = time.time()
            samples = self._sample_population(mean, std, lower_bounds, upper_bounds, dim)
            
            if verbose and iteration == 0:
                print(f"[CEM] Sampled population shape: {samples.shape}, should be ({self.population_size}, {dim})")
                print(f"[CEM] Sample[0] shape: {samples[0].shape}, range: [{samples[0].min():.3f}, {samples[0].max():.3f}]")
            
            # Evaluate all samples
            costs = []
            for idx, sample in enumerate(samples):
                cost = controller.evaluate_decision_vector(sample)
                if verbose and iteration == 0 and idx == 0:
                    print(f"[CEM] First cost type: {type(cost)}, value: {cost}")
                    if not np.isscalar(cost) and hasattr(cost, 'shape'):
                        raise ValueError(f"Cost is not scalar! shape: {cost.shape}") # was having errors with cost shape
                costs.append(cost)
            costs = np.array(costs)
            
            if verbose and iteration == 0:
                print(f"[CEM] Costs array shape: {costs.shape}, should be ({self.population_size},)")
                print(f"[CEM] Costs range: [{costs.min():.1f}, {costs.max():.1f}]")
            
            # Update best solutions list with any new best from this iteration
            self._update_best_solutions(samples, costs)
            
            # Select elite samples from population (lowest cost)
            elite_indices = np.argsort(costs)[:self.num_elites]
            
            # Inject retained best solutions into elites (without double-counting)
            elite_samples, elite_costs = self._inject_best_into_elites(
                samples, costs, elite_indices
            )
            
            # Track best solution
            iter_best_idx = np.argmin(costs)
            iter_best_cost = costs[iter_best_idx]
            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_solution = samples[iter_best_idx].copy()
            
            self.iteration_costs.append(best_cost)
            
            if verbose:
                mean_elite_cost = np.mean(elite_costs)
                iter_time = time.time() - iter_start
                mean_str = np.array2string(mean, precision=3, suppress_small=True, separator=', ')
                std_str = np.array2string(std, precision=3, suppress_small=True, separator=', ')
                print(f"[CEM] Iter {iteration+1}/{max_iters}: Best Cost={best_cost:.1f}, Mean Elite Cost={mean_elite_cost:.1f}, Time={iter_time:.1f}s", flush=True)
                print(f"[CEM] Sampling Mean:\n {mean_str}", flush=True)
                print(f"[CEM] Sampling Std:\ns {std_str}", flush=True)
            
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
                    print(f"Converged: mean change {mean_change:.6f} < {self.epsilon}", flush=True)
                break
        
        if verbose:
            print(f"Optimization complete: best cost = {best_cost:.4f}", flush=True)
            print("=" * 50, flush=True)
        
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
    
    def _update_best_solutions(self, samples: np.ndarray, costs: np.ndarray) -> None:
        """Update the list of best solutions found so far.
        
        Parameters
        ----------
        samples : np.ndarray
            Current population samples (shape: (population_size, dim))
        costs : np.ndarray
            Costs for each sample (shape: (population_size,))
        """
        if self.num_best_retained == 0:
            return
        
        # Add all samples to consideration
        for sample, cost in zip(samples, costs):
            # Check if this solution should be in the best list
            if len(self.best_solutions_list) < self.num_best_retained:
                # List not full, add it
                self.best_solutions_list.append((cost, sample.copy()))
            else:
                # Check if better than worst in list
                worst_idx = np.argmax([c for c, s in self.best_solutions_list])
                worst_cost = self.best_solutions_list[worst_idx][0]
                if cost < worst_cost:
                    # Replace worst with this solution
                    self.best_solutions_list[worst_idx] = (cost, sample.copy())
        
        # Keep list sorted by cost (best first)
        self.best_solutions_list.sort(key=lambda x: x[0])
    
    def _inject_best_into_elites(
        self,
        samples: np.ndarray,
        costs: np.ndarray,
        elite_indices: np.ndarray
    ) -> tuple:
        """Inject retained best solutions into elite set without double-counting.
        
        Parameters
        ----------
        samples : np.ndarray
            Current population samples (shape: (population_size, dim))
        costs : np.ndarray
            Costs for each sample (shape: (population_size,))
        elite_indices : np.ndarray
            Indices of elite samples from population (shape: (num_elites,))
            
        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (elite_samples, elite_costs) with best solutions injected
        """
        if self.num_best_retained == 0 or len(self.best_solutions_list) == 0:
            # No retention, return original elites
            return samples[elite_indices], costs[elite_indices]
        
        # Start with population elites
        elite_set = []
        for idx in elite_indices:
            elite_set.append((costs[idx], samples[idx]))
        
        # Track which retained bests are already in population elites
        retained_in_pop = set()
        
        # Check if any retained best is already in the elite set from population
        for i, (best_cost, best_sol) in enumerate(self.best_solutions_list):
            for j, (elite_cost, elite_sol) in enumerate(elite_set):
                # Check if this is the same solution (within tolerance)
                if np.allclose(best_sol, elite_sol, rtol=1e-9, atol=1e-12):
                    retained_in_pop.add(i)
                    break
        
        # Add retained bests that aren't already in population elites
        num_to_add = 0
        for i, (best_cost, best_sol) in enumerate(self.best_solutions_list):
            if i not in retained_in_pop:
                elite_set.append((best_cost, best_sol.copy()))
                num_to_add += 1
        
        # Sort combined set by cost and take top num_elites
        elite_set.sort(key=lambda x: x[0])
        elite_set = elite_set[:self.num_elites]
        
        # Extract samples and costs
        elite_costs = np.array([c for c, s in elite_set])
        elite_samples = np.array([s for c, s in elite_set])
        
        return elite_samples, elite_costs
    
    def get_iteration_history(self) -> np.ndarray:
        """Get best cost at each iteration.
        
        Returns
        -------
        np.ndarray
            Array of best costs (shape: (num_iterations,))
        """
        return np.array(self.iteration_costs)
