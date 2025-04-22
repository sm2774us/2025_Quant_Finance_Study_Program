# src/day13_capstone.py
from mosek.fusion import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def capstone_optimization(returns: pd.DataFrame, var_limit: float, lcr_limit: float, hqla_factors: np.ndarray, outflows: float) -> tuple:
    """
    Optimize multi-asset margins with VaR, LCR, and regulatory constraints.
    
    Args:
        returns: DataFrame of asset returns
        var_limit: Maximum VaR
        lcr_limit: Minimum LCR
        hqla_factors: HQLA eligibility factors
        outflows: Estimated cash outflows
    Returns:
        Tuple of (margins, contributions)
    """
    n = returns.shape[1]
    with Model("Capstone") as M:
        m = M.variable("m", n, Domain.greaterThan(0.25))  # FINRA 25%
        portfolio_returns = Expr.dot(m, returns.T)
        M.constraint("var", Expr.mul(-1.645, Expr.stdDev(portfolio_returns)), Domain.lessThan(var_limit))
        M.constraint("lcr", Expr.dot(hqla_factors, m), Domain.greaterThan(lcr_limit * outflows))
        M.objective("obj", ObjectiveSense.Minimize, Expr.sum(m))
        M.solve()
        margins = m.level()
    
    # Simplified Shapley for attribution
    contributions = np.zeros(n)
    portfolio_vol = np.sqrt(np.diag(returns.cov() * 252))
    for i in range(n):
        contributions[i] = portfolio_vol[i] * np.corrcoef(returns.iloc[:, i], returns.mean(axis=1))[0, 1]
    contributions = contributions / np.sum(contributions) * 100
    
    return margins, contributions

if __name__ == "__main__":
    np.random.seed(42)
    data = pd.DataFrame({
        "Equity": np.random.normal(0, 0.01, 252),
        "Credit": np.random.normal(0, 0.015, 252),
        "Commodity": np.random.normal(0, 0.02, 252)
    })
    hqla_factors = np.array([0.8, 0.9, 0.6])  # HQLA eligibility
    margins, contributions = capstone_optimization(data, 0.05, 1.0, hqla_factors, 10.0)
    print(f"Capstone Margins: {margins}")
    plt.bar(data.columns, contributions)
    plt.title("Margin Contributions (%)")
    plt.ylabel("Contribution")
    plt.savefig("capstone_contributions.png")
    plt.close()
