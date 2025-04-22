import numpy as np
import pandas as pd

def compute_margin(returns: pd.DataFrame, confidence: float = 0.95, scaling_factor: float = 1.0) -> float:
    """
    Compute portfolio margin using VaR and cross-margining.
    
    Args:
        returns: DataFrame of asset returns
        confidence: VaR confidence level
        scaling_factor: Margin scaling factor
    Returns:
        Total portfolio margin
    """
    portfolio_returns = returns.mean(axis=1)
    var = np.percentile(portfolio_returns, 100 * (1 - confidence))
    cov_matrix = returns.cov() * 252
    weights = np.ones(returns.shape[1]) / returns.shape[1]
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    margin = -var * np.sqrt(252) * scaling_factor
    return margin

if __name__ == "__main__":
    np.random.seed(42)
    data = pd.DataFrame({
        "Asset1": np.random.normal(0, 0.01, 252),
        "Asset2": np.random.normal(0, 0.015, 252)
    })
    margin = compute_margin(data)
    print(f"Portfolio Margin: {margin:.4f}")
