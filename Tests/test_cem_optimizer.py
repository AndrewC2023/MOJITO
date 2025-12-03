"""Test Cross-Entropy Method optimizer on known test functions.

Validates CEM implementation before using it in full MPC pipeline.
Tests on classic optimization benchmarks with known global minima.
"""

import numpy as np
import sys
sys.path.append('/workspaces/MOJITO/src')

from Optimizers.CrossEntropyMethod import CrossEntropyMethod


class SimpleTestFunction:
    """Wrapper to provide evaluate_decision_vector interface for test functions."""
    
    def __init__(self, cost_func):
        self.cost_func = cost_func
        self.num_evaluations = 0
    
    def evaluate_decision_vector(self, vec):
        self.num_evaluations += 1
        return self.cost_func(vec)


def quadratic_bowl(x):
    """Simple quadratic: f(x) = ||x - x_opt||^2
    Global minimum: f(0.5, 0.5, ...) = 0
    """
    x_opt = np.full_like(x, 0.5)
    return np.sum((x - x_opt)**2)


def rosenbrock(x):
    """Rosenbrock function (banana valley): challenging nonconvex problem.
    f(x) = sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    Global minimum: f(1, 1, ...) = 0
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin(x):
    """Rastrigin function: highly multimodal (many local minima).
    f(x) = 10n + sum[x_i^2 - 10*cos(2*pi*x_i)]
    Global minimum: f(0, 0, ...) = 0
    Search domain typically [-5.12, 5.12]
    """
    n = len(x)
    # Shift to [0, 1] domain: x_scaled = 10.24 * x - 5.12
    x_scaled = 10.24 * x - 5.12
    return 10.0 * n + np.sum(x_scaled**2 - 10.0 * np.cos(2 * np.pi * x_scaled))


def test_cem_on_quadratic():
    """Test CEM on simple quadratic bowl."""
    print("\n" + "="*60)
    print("TEST 1: Quadratic Bowl (Simple Convex)")
    print("="*60)
    
    dim = 10
    initial_guess = np.zeros(dim)  # Start far from optimum
    
    controller = SimpleTestFunction(quadratic_bowl)
    
    optimizer = CrossEntropyMethod(
        population_size=100,
        elite_frac=0.1,
        max_iterations=20,
        epsilon=1e-4,
        initial_std=0.3,
        verbose=True
    )
    
    solution = optimizer.optimize(initial_guess, controller)
    final_cost = controller.evaluate_decision_vector(solution)
    
    print(f"\n--- Results ---")
    print(f"Solution: {solution}")
    print(f"Expected: {np.full(dim, 0.5)}")
    print(f"Final cost: {final_cost:.6e}")
    print(f"Expected: 0.0")
    print(f"Error: {np.linalg.norm(solution - 0.5):.6e}")
    print(f"Total evaluations: {controller.num_evaluations}")
    
    # Verify convergence
    assert final_cost < 1e-3, f"Failed to converge: cost = {final_cost}"
    assert np.allclose(solution, 0.5, atol=0.05), f"Solution error too large"
    
    print("✅ PASSED: Converged to global minimum")


def test_cem_on_rosenbrock():
    """Test CEM on Rosenbrock function (nonconvex banana valley)."""
    print("\n" + "="*60)
    print("TEST 2: Rosenbrock Function (Nonconvex Banana Valley)")
    print("="*60)
    
    dim = 5
    initial_guess = np.zeros(dim)  # Start at [0, 0, 0, 0, 0]
    
    controller = SimpleTestFunction(rosenbrock)
    
    optimizer = CrossEntropyMethod(
        population_size=300,
        elite_frac=0.15,
        max_iterations=30,
        epsilon=1e-4,
        initial_std=0.4,
        verbose=True
    )
    
    solution = optimizer.optimize(initial_guess, controller)
    final_cost = controller.evaluate_decision_vector(solution)
    
    print(f"\n--- Results ---")
    print(f"Solution: {solution}")
    print(f"Expected: {np.ones(dim)}")
    print(f"Final cost: {final_cost:.6e}")
    print(f"Expected: 0.0")
    print(f"Error: {np.linalg.norm(solution - 1.0):.6e}")
    print(f"Total evaluations: {controller.num_evaluations}")
    
    # Rosenbrock is harder - allow larger tolerance
    assert final_cost < 5.0, f"Failed to converge reasonably: cost = {final_cost}"
    
    print("✅ PASSED: Found good solution in banana valley")


def test_cem_on_rastrigin():
    """Test CEM on Rastrigin function (highly multimodal)."""
    print("\n" + "="*60)
    print("TEST 3: Rastrigin Function (Highly Multimodal)")
    print("="*60)
    
    dim = 4
    # Start at [0.5, 0.5, 0.5, 0.5] which maps to origin in scaled space
    initial_guess = np.full(dim, 0.5)
    
    controller = SimpleTestFunction(rastrigin)
    
    optimizer = CrossEntropyMethod(
        population_size=500,
        elite_frac=0.1,
        max_iterations=40,
        epsilon=1e-4,
        initial_std=0.2,
        verbose=True
    )
    
    solution = optimizer.optimize(initial_guess, controller)
    final_cost = controller.evaluate_decision_vector(solution)
    
    # Expected solution maps origin to [0.5, 0.5, ...]
    x_opt = np.full(dim, 0.5)
    
    print(f"\n--- Results ---")
    print(f"Solution: {solution}")
    print(f"Expected: {x_opt}")
    print(f"Final cost: {final_cost:.6e}")
    print(f"Expected: 0.0")
    print(f"Error: {np.linalg.norm(solution - x_opt):.6e}")
    print(f"Total evaluations: {controller.num_evaluations}")
    
    # Rastrigin is very hard - just check we improve significantly
    initial_cost = rastrigin(initial_guess)
    improvement = initial_cost - final_cost
    print(f"Initial cost: {initial_cost:.4f}")
    print(f"Improvement: {improvement:.4f} ({100*improvement/initial_cost:.1f}%)")
    
    # Should at least reach near the global basin
    assert final_cost < 20.0, f"Failed to find good region: cost = {final_cost}"
    
    print("✅ PASSED: Successfully navigated multimodal landscape")


def test_cem_convergence_tracking():
    """Test that CEM properly tracks convergence history."""
    print("\n" + "="*60)
    print("TEST 4: Convergence History Tracking")
    print("="*60)
    
    dim = 8
    initial_guess = np.zeros(dim)
    
    controller = SimpleTestFunction(quadratic_bowl)
    
    optimizer = CrossEntropyMethod(
        population_size=100,
        elite_frac=0.1,
        max_iterations=15,
        epsilon=1e-5,
        initial_std=0.3,
        verbose=False
    )
    
    solution = optimizer.optimize(initial_guess, controller)
    history = optimizer.get_iteration_history()
    
    print(f"Iterations completed: {len(history)}")
    print(f"Cost history: {history}")
    
    # Verify monotonic improvement
    for i in range(len(history) - 1):
        assert history[i+1] <= history[i], \
            f"Cost increased from {history[i]} to {history[i+1]} at iteration {i}"
    
    print("✅ PASSED: Cost monotonically decreases")


def test_cem_with_bounds():
    """Test CEM with explicit bounds on decision variables."""
    print("\n" + "="*60)
    print("TEST 5: Constrained Optimization with Bounds")
    print("="*60)
    
    def constrained_quadratic(x):
        """Optimal point at [0.5, 0.5] but constrained to [0.2, 0.8]."""
        return np.sum((x - 0.5)**2)
    
    dim = 6
    initial_guess = np.full(dim, 0.5)
    
    # Tight bounds
    lower = np.full(dim, 0.2)
    upper = np.full(dim, 0.8)
    
    controller = SimpleTestFunction(constrained_quadratic)
    
    optimizer = CrossEntropyMethod(
        population_size=100,
        elite_frac=0.1,
        max_iterations=20,
        initial_std=0.1,
        bounds=(lower, upper),
        verbose=True
    )
    
    solution = optimizer.optimize(initial_guess, controller)
    
    print(f"\n--- Results ---")
    print(f"Solution: {solution}")
    print(f"Lower bounds: {lower}")
    print(f"Upper bounds: {upper}")
    
    # Verify bounds are satisfied
    assert np.all(solution >= lower), "Solution violates lower bounds"
    assert np.all(solution <= upper), "Solution violates upper bounds"
    
    # Should find unconstrained optimum since it's within bounds
    assert np.allclose(solution, 0.5, atol=0.05), "Failed to find optimum"
    
    print("✅ PASSED: Bounds satisfied and optimal solution found")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Cross-Entropy Method Optimizer Validation Suite")
    print("#"*60)
    
    test_cem_on_quadratic()
    test_cem_on_rosenbrock()
    test_cem_on_rastrigin()
    test_cem_convergence_tracking()
    test_cem_with_bounds()
    
    print("\n" + "#"*60)
    print("# ✅ ALL TESTS PASSED")
    print("#"*60)
