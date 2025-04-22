# src/day6_requirements.md
# Margin Allocation Model Requirements

## Objective
Develop a risk-based margin allocation model to optimize capital utilization across Equities and Credit portfolios.

## Stakeholders
- **Portfolio Managers**: Require minimized margins to maximize returns.
- **Risk Team**: Ensure VaR does not exceed 5% at 95% confidence.
- **IT**: Need model in Python with CSV data inputs.

## Requirements
1. **Model Scope**: Allocate margins for a portfolio of 10 Equities and 5 Credit assets.
2. **Risk Constraint**: Portfolio VaR ≤ 0.05 (95% confidence, annualized).
3. **Optimization**: Minimize total margin using Mosek solver.
4. **Data Inputs**: Daily returns in CSV format (columns: date, asset_id, return).
5. **Outputs**: Margin per asset, total margin, and VaR.
6. **Explainability**: Provide attribution of margin drivers (volatility, correlation).

## Constraints
- Regulatory: Comply with FINRA 25% maintenance margin.
- Technical: Run on Python 3.9 with NumPy, Pandas, Mosek.

## Deliverables
- Python script with optimization model.
- Documentation explaining methodology.
- Presentation for stakeholders.

## Timeline
- Prototype: 1 week
- Testing and Deployment: 2 weeks
