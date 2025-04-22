# src/day1_portfolio.py
import numpy as np
import pandas as pd

def compute_portfolio_metrics(prices: pd.DataFrame, weights: np.ndarray) -> tuple:
    """
    Compute annualized portfolio return and volatility.
    
    Args:
        prices: DataFrame with asset prices
        weights: Array of portfolio weights
    Returns:
        Tuple of (portfolio return, portfolio volatility)
    """
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

if __name__ == "__main__":
    # Simulate price data
    np.random.seed(42)
    data = pd.DataFrame({
        "Stock1": np.random.normal(100, 10, 252),
        "Stock2": np.random.normal(100, 15, 252)
    })
    weights = np.array([0.6, 0.4])
    ret, vol = compute_portfolio_metrics(data, weights)
    print(f"Portfolio Return: {ret:.2%}, Volatility: {vol:.2%}")
