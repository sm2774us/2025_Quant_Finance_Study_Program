# src/day12_presentation.md
# Presentation Script: Margin Methodology Upgrade

**Slide 1: Introduction (30 seconds)**
- Good afternoon, team. Today, I’ll present our plan to upgrade the margin methodology to support Equities, Credit, and Commodities.
- Objective: Optimize capital efficiency while meeting regulatory requirements.

**Slide 2: Business Benefits (1 minute)**
- **Capital Efficiency**: Cross-margining reduces total margins by leveraging diversification.
- **Compliance**: Ensures VaR ≤ 5% and FINRA 25% maintenance margin.
- **Transparency**: Attribution tools explain margin drivers for PMs and Risk.

**Slide 3: Project Plan (2 minutes)**
- **Scope**: Multi-asset margin model using Python and Mosek.
- **Timeline**: 4 weeks, with milestones:
  - Week 1: Data collection.
  - Week 2: Prototype.
  - Week 3: Testing.
  - Week 4: Deployment.
- **Dependencies**: IT for data, Risk for VaR, PMs for weights.
- **Risks**: Data quality (mitigated by validation), regulatory changes (modular code).

**Slide 4: Stakeholder Roles (1 minute)**
- **PMs**: Provide asset weights, benefit from lower margins.
- **Risk**: Approve VaR thresholds, ensure compliance.
- **IT**: Deliver data pipelines, deploy model.

**Slide 5: Next Steps (30 seconds)**
- Kick off data collection next week.
- Schedule stakeholder reviews for prototype and testing.
- Questions?

**Total Time**: 5 minutes
