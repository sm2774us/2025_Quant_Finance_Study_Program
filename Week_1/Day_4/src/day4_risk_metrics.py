# src/day4_risk_metrics.py
import numpy as np
import pandas as pd

def compute_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Compute annualized VaR."""
    return np.percentile(returns, 100 * (1 - confidence)) * np.sqrt(252)

def compute_beta(stock_returns: np.ndarray, market_returns: np.ndarray) -> float:
    """Compute stock beta relative to market."""
    cov = np.cov(stock_returns, market_returns)[0, 1]
    var = np.var(market_returns)
    return cov / var

def compute_spread_volatility(yields: np.ndarray, risk_free: float) -> float:
    """Compute credit spread volatility."""
    spreads = yields - risk_free
    return np.std(spreads) * np.sqrt(252)

if __name__ == "__main__":
    np.random.seed(42)
    stock = np.random.normal(0, 0.01, 252)
    market = np.random.normal(0, 0.008, 252)
    bond_yields = np.random.normal(0.05, 0.002, 252)
    risk_free = 0.03
    var = compute_var(stock)
    beta = compute_beta(stock, market)
    spread_vol = compute_spread_volatility(bond_yields, risk_free)
    print(f"Equity VaR: {var:.4f}, Beta: {beta:.4f}, Credit Spread Volatility: {spread_vol:.4f}")
