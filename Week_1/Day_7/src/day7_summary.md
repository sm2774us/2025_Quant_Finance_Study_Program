# src/day7_summary.md
# Mini-Project Summary: Margin Allocation Model

## Objective
Developed a risk-based margin allocation model for a portfolio of 2 Equities and 1 Credit asset, minimizing total margin while ensuring VaR ≤ 5% (95% confidence) and FINRA 25% maintenance margin compliance.

## Methodology
- **Data**: Simulated 252 days of returns (normal distribution).
- **Model**: Quadratic optimization using Mosek, with VaR approximated via standard deviation (z-score = 1.645 for 95%).
- **Constraints**:
  - VaR ≤ 0.05 (annualized).
  - Minimum margin per asset ≥ 0.25 (FINRA).
- **Output**: Margin per asset and total margin.

## Assumptions
- Returns are normally distributed.
- Correlations based on historical covariance.
- Annualized VaR using 252 trading days.

## Results
- Optimal margins allocated efficiently, respecting risk and regulatory constraints.
- Model is extensible to additional assets and constraints.

## Next Steps
- Incorporate stochastic returns.
- Add attribution for margin drivers.
