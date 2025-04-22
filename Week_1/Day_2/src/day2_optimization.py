from mosek.fusion import *
import numpy as np

def portfolio_optimization(returns: np.ndarray, cov_matrix: np.ndarray, target_return: float) -> np.ndarray:
    """
    Optimize portfolio weights to minimize volatility for a target return.
    
    Args:
        returns: Array of asset returns
        cov_matrix: Covariance matrix
        target_return: Desired portfolio return
    Returns:
        Array of optimal weights
    """
    n = len(returns)
    with Model("Portfolio") as M:
        x = M.variable("x", n, Domain.greaterThan(0.0))
        M.constraint("budget", Expr.sum(x), Domain.equalsTo(1.0))
        M.constraint("return", Expr.dot(returns, x), Domain.greaterThan(target_return))
        M.objective("obj", ObjectiveSense.Minimize, Expr.sqrt(Expr.dot(x, Expr.mul(cov_matrix, x))))
        M.solve()
        return x.level()

if __name__ == "__main__":
    np.random.seed(42)
    returns = np.array([0.1, 0.15, 0.08])
    cov_matrix = np.array([[0.05, 0.01, 0.01], [0.01, 0.07, 0.02], [0.01, 0.02, 0.06]])
    weights = portfolio_optimization(returns, cov_matrix, 0.1)
    print(f"Optimal Weights: {weights}")
