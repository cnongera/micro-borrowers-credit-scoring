# MBMCS - Micro-Borrowers Mobile-Money Credit Scoring Framework 

**MBMCS** is a transparent, domain-guided alternative credit scoring architecture designed for Kenyan micro-entrepreneurs, gig workers and informal sector participants who lack formal payslips or collateral. 

Instead of using opaque black-box machine learning, I convert a standard Safaricom M-Pesa statement into a 300–850 credit score using five interpretable pillars:
1. **Cash Flow Velocity** (inflow volume, transaction density)
2. **Repayment Capacity** (income stability, investment-adjusted net flow)
3. **Liquidity Buffer** (average balance, minimum-balance floor)
4. **Financial Diversity** (Herfindahl-Hirschman Index over exclusive income categories)
5. **Credit Discipline** (absence of Fuliza overdraft distress, zero delinquency)

## Key Findings (Single-Case Proof of Concept)
Applied to a real 365-day ledger (1,215 transactions; KES 1.0M+ turnover), the framework yields a score of **641/850 (Grade A, Tier 3 Fair)**. The conservative, median-based maximum safe unsecured principal is **KES 25,766** over 3 months at a 25% burden threshold. 

The ledger shows perfect discipline (zero Fuliza penalties) and real diversity (HHI = 0.567 over four categories), but is held back by high income volatility (CV = 1.13) and a thin liquidity floor (minimum balance of KES 0.15).

## Repository Contents
* `credit_scorer.py`: The complete, standalone Python scoring engine.
* `MBMCS_Paper.md`: The full research paper detailing the methodology, math, and limitations.
* `MBMCS_Paper.pdf`: The typeset PDF version of the paper (generated via Kami(https://github.com/tw93/Kami)).
* `mpesa_corrected.json`: The raw output of the scoring engine applied to the author's real ledger.

## Usage
You can run the scorer on any standard M-Pesa CSV file. 
```bash
# Run with default settings (KES 30,000 loan, 3 months, 10% interest)
python3 credit_scorer.py "Your_Mpesa_Statement.csv"

# Run with custom loan parameters
python3 credit_scorer.py "Your Mpesa Statement.csv" --loan 50000 --months 6 --interest 12
