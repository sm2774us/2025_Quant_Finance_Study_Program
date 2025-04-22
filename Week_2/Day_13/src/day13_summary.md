# src/day13_summary.md
# Capstone Project: Multi-Asset Margin Model

## Objective
Developed a comprehensive margin allocation model for a portfolio of Equity, Credit, and Commodity assets, minimizing total margin while meeting VaR (≤ 5%, 95% confidence), LCR (≥ 1), and FINRA (25% minimum) constraints. Included attribution for transparency.

## Methodology
- **Data**: Simulated 252 days of returns (normal distribution).
- **Model**: Quadratic optimization using Mosek.
- **Constraints**:
  - VaR ≤ 0.05 (z-score = 1.645).
  - LCR ≥ 1 (HQLA/outflows).
  - Minimum margin ≥ 0.25 per asset.
- **Attribution**: Simplified Shapley values based on volatility and correlation.
- **Outputs**: Margins per asset, total margin, contribution chart.

## Assumptions
- Normal returns distribution.
- Constant HQLA factors (Equity: 0.8, Credit: 0.9, Commodity: 0.6).
- Annualized metrics (252 trading days).

## Results
- Optimal margins allocated, balancing risk, liquidity, and regulatory requirements.
- Attribution highlights Commodity as the largest margin driver due to high volatility.

## Next Steps
- Incorporate stochastic scenarios for robustness.
- Enhance attribution with full Shapley calculations.
- Deploy model with IT support.
