# src/day14_sensitivity.py
import numpy as np
import pandas as pd
from mosek.fusion import *

def sensitivity_analysis(returns: pd.DataFrame, var_limit: float, lcr_limit: float, hqla_factors: np.ndarray, outflows: float, vol_scale: float = 1.0) -> tuple:
    """
    Perform sensitivity analysis on margins by scaling volatility.
    
    Args:
        returns: DataFrame of asset returns
        var_limit: Maximum VaR
        lcr_limit: Minimum LCR
        hqla_factors: HQLA eligibility factors
        outflows: Estimated cash outflows
        vol_scale: Volatility scaling factor
    Returns:
        Tuple of (margins, total margin)
    """
    scaled_returns = returns * vol_scale
    n = returns.shape[1]
    with Model("Sensitivity") as M:
        m = M.variable("m", n, Domain.greaterThan(0.25))
        portfolio_returns = Expr.dot(m, scaled_returns.T)
        M.constraint("var", Expr.mul(-1.645, Expr.stdDev(portfolio_returns)), Domain.lessThan(var_limit))
        M.constraint("lcr", Expr.dot(hqla_factors, m), Domain.greaterThan(lcr_limit * outflows))
        M.objective("obj", ObjectiveSense.Minimize, Expr.sum(m))
        M.solve()
        margins = m.level()
        total_margin = np.sum(margins)
    return margins, total_margin

if __name__ == "__main__":
    np.random.seed(42)
    data = pd.DataFrame({
        "Equity": np.random.normal(0, 0.01, 252),
        "Credit": np.random.normal(0, 0.015, 252),
        "Commodity": np.random.normal(0, 0.02, 252)
    })
    hqla_factors = np.array([0.8, 0.9, 0.6])
    scales = [1.0, 1.1, 1.2]
    results = []
    for scale in scales:
        margins, total = sensitivity_analysis(data, 0.05, 1.0, hqla_factors, 10.0, scale)
        results.append(total)
    print(f"Sensitivity Results (Total Margin): {dict(zip(scales, results))}")
