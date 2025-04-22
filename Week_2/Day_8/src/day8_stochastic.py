# src/day8_stochastic.py
from mosek.fusion import *
import numpy as np

def stochastic_optimization(returns: np.ndarray, scenarios: list, var_limit: float) -> np.ndarray:
    """
    Optimize margins under stochastic scenarios with VaR constraint.
    
    Args:
        returns: Array of asset returns
        scenarios: List of scenario returns
        var_limit: Maximum allowable VaR
    Returns:
        Array of optimal margins
    """
    n = returns.shape[1]
    with Model("Stochastic") as M:
        m = M.variable("m", n, Domain.greaterThan(0.0))
        for i, s in enumerate(scenarios):
            M.constraint_modal(f"var_{i}", Expr.dot(m, s), Domain.lessThan(var_limit))
        M.objective("obj", ObjectiveSense.Minimize, Expr.sum(m))
        M.solve()
        return m.level()

if __name__ == "__main__":
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, (252, 2))
    scenarios = [returns[i:i+10].mean(axis=0) for i in range(0, 252, 10)]
    margins = stochastic_optimization(returns, scenarios, 0.05)
    print(f"Stochastic Margins: {margins}")
