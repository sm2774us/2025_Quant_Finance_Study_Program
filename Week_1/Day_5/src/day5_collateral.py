# src/day5_collateral.py
from mosek.fusion import *
import numpy as np

def collateral_optimization(costs: np.ndarray, values: np.ndarray, requirement: float) -> np.ndarray:
    """
    Optimize collateral allocation to minimize cost.
    
    Args:
        costs: Array of collateral costs
        values: Array of collateral values
        requirement: Minimum required value
    Returns:
        Array of optimal allocations
    """
    n = len(costs)
    with Model("Collateral") as M:
        x = M.variable("x", n, Domain.greaterThan(0.0))
        M.constraint("value", Expr.dot(values, x), Domain.greaterThan(requirement))
        M.objective("obj", ObjectiveSense.Minimize, Expr.dot(costs, x))
        M.solve()
        return x.level()

if __name__ == "__main__":
    np.random.seed(42)
    costs = np.array([0.1, 0.2, 0.15])  # Cost per unit
    values = np.array([1.0, 0.9, 1.1])  # Value per unit
    requirement = 10.0  # Total value needed
    allocation = collateral_optimization(costs, values, requirement)
    print(f"Collateral Allocation: {allocation}")
