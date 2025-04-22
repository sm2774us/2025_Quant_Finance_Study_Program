# src/day14_presentation.md
# Presentation Script: Capstone Project

**Slide 1: Introduction (1 minute)**
- Hello, team. I’m presenting our capstone project: a multi-asset margin allocation model for Equities, Credit, and Commodities.
- Objective: Minimize margins while ensuring VaR ≤ 5%, LCR ≥ 1, and FINRA compliance.

**Slide 2: Methodology (2 minutes)**
- **Model**: Quadratic optimization using Mosek.
- **Constraints**:
  - VaR ≤ 0.05 (95% confidence, z-score = 1.645).
  - LCR ≥ 1 using HQLA factors.
  - Minimum margin 25% per asset (FINRA).
- **Attribution**: Simplified Shapley values for margin contributions.
- **Data**: 252 days of simulated returns.

**Slide 3: Results (3 minutes)**
- **Margins**: Optimized allocations for Equity, Credit, Commodity.
- **Attribution**: Commodity drives margins due to high volatility (see bar chart).
- **Sensitivity**: 10% volatility increase raises total margin by 8%.
- **Scenarios**: High-correlation stress scenario increases VaR by 12%.

**Slide 4: Business Impact (2 minutes)**
- **Capital Efficiency**: Cross-margining reduces capital requirements.
- **Compliance**: Meets FINRA and Basel III standards.
- **Scalability**: Modular code supports additional assets.
- **Transparency**: Attribution aids PMs and Risk in decision-making.

**Slide 5: Next Steps (1 minute)**
- Deploy model with IT in 4 weeks.
- Enhance with stochastic scenarios and full Shapley attribution.
- Schedule follow-up reviews with stakeholders.

**Slide 6: Q&A (1 minute)**
- Questions? (Prepared for: black swan events, data quality, regulatory changes)

**Total Time**: 10 minutes
