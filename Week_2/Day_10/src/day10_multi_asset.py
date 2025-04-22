# src/day10_multi_asset.py
import numpy as np
import pandas as pd

def multi_asset_margin(returns: pd.DataFrame) -> float:
    """
    Compute margin for a multi-asset portfolio with cross-margining.
    
    Args:
        returns: DataFrame of asset returns
    Returns:
        Total portfolio margin
    """
    cov = returns.cov() * 252
    weights = np.ones(returns.shape[1]) / returns.shape[1]
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    margin = 1.645 * portfolio_vol  # 95% VaR
    return margin

if __name__ == "__main__":
    np.random.seed(42)
    data = pd.DataFrame({
        "Equity": np.random.normal(0, 0.01, 252),
        "Credit": np.random.normal(0, 0.015, 252),
        "Commodity": np.random.normal(0, 0.02, 252)
    })
    margin = multi_asset_margin(data)
    print(f"Multi-Asset Margin: {margin:.4f}")
