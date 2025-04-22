# src/day11_liquidity.py
from mosek.fusion import *
import numpy as np

def liquidity_optimization(costs: np.ndarray, values: np.ndarray, haircuts: np.ndarray, requirement: float, haircut_limit: float) -> np.ndarray:
    """
    Optimize collateral allocation with LCR and haircut constraints.
    
    Args:
        costs: Array of collateral costs
        values: Array of collateral values
        haircuts: Array of haircut percentages
        requirement: Minimum required value
        haircut_limit: Maximum allowable haircut
    Returns:
        Array of optimal allocations
    """
    n = len(costs)
    with Model("Liquidity") as M:
        x = M.variable("x", n, Domain.greaterThan(0.0))
        M.constraint("value", Expr.dot(values, x), Domain.greaterThan(requirement))
        M.constraint("haircut", Expr.dot(haircuts, x), Domain.lessThan(haircut_limit))
        M.objective("obj", ObjectiveSense.Minimize, Expr.dot(costs, x))
        M.solve()
        return x.level()

if __name__ == "__main__":
    np.random.seed(42)
    costs = np.array([0.1, 0.2, 0.15])
    values = np.array([1.0, 0.9, 1.1])
    haircuts = np.array([0.05, 0.1, 0.07])
    requirement = 10.0
    haircut_limit = 0.5
    allocation = liquidity_optimization(costs, values, haircuts, requirement, haircut_limit)
    print(f"Liquidity Allocation: {allocation}")
