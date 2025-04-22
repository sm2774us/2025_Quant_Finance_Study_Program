# src/day9_attribution.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def shapley_margin(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute simplified Shapley values for margin contributions.
    
    Args:
        returns: DataFrame of asset returns
    Returns:
        Array of margin contributions
    """
    n = returns.shape[1]
    shapley = np.zeros(n)
    portfolio_vol = np.sqrt(np.diag(returns.cov() * 252))
    for i in range(n):
        shapley[i] = portfolio_vol[i] * np.corrcoef(returns.iloc[:, i], returns.mean(axis=1))[0, 1]
    return shapley / np.sum(shapley) * 100  # Normalize to percentages

if __name__ == "__main__":
    np.random.seed(42)
    data = pd.DataFrame({
        "Equity1": np.random.normal(0, 0.01, 252),
        "Equity2": np.random.normal(0, 0.015, 252),
        "Credit": np.random.normal(0, 0.008, 252)
    })
    contributions = shapley_margin(data)
    plt.bar(data.columns, contributions)
    plt.title("Margin Contributions (%)")
    plt.ylabel("Contribution")
    plt.savefig("margin_contributions.png")
    plt.close()
