# src/day7_margin_model.py
from mosek.fusion import *
import numpy as np
import pandas as pd

def margin_optimization(returns: pd.DataFrame, var_limit: float) -> np.ndarray:
    """
    Optimize margin allocation subject to VaR constraint.
    
    Args:
        returns: DataFrame of asset returns
        var_limit: Maximum allowable VaR
    Returns:
        Array of optimal margins
    """
    n = returns.shape[1]
    with Model("Margin") as M:
        m = M.variable("m", n, Domain.greaterThan(0.0))
        portfolio_returns = Expr.dot(m, returns.T)
        M.constraint("var", Expr.mul(-1.645, Expr.stdDev(portfolio_returns)), Domain.lessThan(var_limit))  # 95% VaR
        M.constraint("regulatory", m, Domain.greaterThan(0.25))  # FINRA 25% margin
        M.objective("obj", ObjectiveSense.Minimize, Expr.sum(m))
        M.solve()
        return m.level()

if __name__ == "__main__":
    np.random.seed(42)
    data = pd.DataFrame({
        "Equity1": np.random.normal(0, 0.01, 252),
        "Equity2": np.random.normal(0, 0.015, 252),
        "Credit": np.random.normal(0, 0.008, 252)
    })
    margins = margin_optimization(data, 0.05)
    print(f"Margin Allocation: {margins}")
