# A Transparent Feature-Engineering Framework for Mobile-Money Credit Scoring: A Single-Case Proof of Concept from a Kenyan Micro-Entrepreneur's M-Pesa Ledger

**Author:** Clement Ongera Nyangoya  
**Affiliation:** Independent researcher, Nakuru County, Kenya  
**Date:** October 2026  
**Status:** Working paper / Formal Academic Preprint  
**JEL Classification:** G21, G23, O16, O33 (Financial Economics & Digital Finance)  
**Keywords:** Alternative credit scoring, mobile money, M-Pesa, financial inclusion, Kenya, feature engineering.

---

## Executive Summary & Abstract

### Executive Summary

| Operational Metric | Value | Underwriting Context |
|---|---|---|
| **Observation Window** | 364 Days | 22 March 2024 to 22 March 2025 (1,215 completed transactions) |
| **Gross Turnover** | KES 1,008,803 Inflow / KES 1,010,554 Outflow | Balanced annual flow (-0.17% net margin) |
| **Composite Score** | **641 / 850** | **Grade A (Tier 3 Fair)**; 75.4% fulfillment across 5 pillars |
| **Safe Principal** | **KES 25,766** | Conservative median-inflow basis (3 mo tenure, 25% burden, 10% cost) |

### Abstract

Millions of micro-entrepreneurs across Sub-Saharan Africa remain excluded from formal credit because they lack payslips, collateral, or Credit Reference Bureau files. Mobile-money platforms (most notably Safaricom M-Pesa in Kenya) generate continuous, tamper-resistant ledgers of cash-flow behaviour. We present **MBMCS**, a transparent, domain-guided feature-engineering framework that converts a 12-month M-Pesa statement into an explainable 300–850 score across five pillars: cash-flow velocity, repayment capacity, liquidity buffer, financial diversity, and credit discipline.

Unlike black-box alternatives, MBMCS uses monotone, exponentially saturating transformations and a Herfindahl-Hirschman concentration measure over **mutually exclusive income categories**, so every score change is explainable and no transaction is double-counted. We apply the framework to a **single applicant** (the first author) using a 365-day ledger of 1,215 transactions totalling KES 1.0 million in gross turnover. Under conservative assumptions (median monthly inflow, 25% debt-service burden), the framework assigns a score of **641/850 (Grade A, Tier 3 Fair)** and a maximum safe unsecured principal of **KES 25,766 over three months**. We emphasise that this is a **single-case proof of concept**, not a validated scoring model: there are no loan outcomes, no train/test split, and no calibration to default rates. We release the code and a reusable CLI tool, and we invite SACCOs and MFIs to partner on an out-of-time validation study.

---

## 1. Introduction

### 1.1 The Collateral & Bureau Paradox

Conventional credit underwriting in Kenya relies on payslips, audited financial statements, real-estate collateral, and Credit Reference Bureau (CRB) files. These requirements systematically exclude an estimated 8.5 million informal micro-entrepreneurs, smallholders, and gig workers who operate primarily in cash and mobile payments. Meanwhile, M-Pesa generates a continuous, tamper-resistant digital ledger of inflows, outflows, and balances for over 96% of Kenyan households. The gap is not lack of data; the gap is lack of a transparent, reproducible method for converting that data into a credit-relevant underwriting score.

The central challenge in Sub-Saharan microfinance is not data scarcity: it is the lack of auditable, non-extractive scoring mechanisms that respect borrower sovereignty and local institutional constraints.

### 1.2 Contributions of the Framework

This paper formalises such a method. We present:

1. A **five-pillar scoring architecture** with monotone, exponentially saturating transformations ensuring bounded, auditable adjustments.
2. A diversity measure based on the **Herfindahl-Hirschman Index (HHI)** computed over mutually exclusive income categories (MMF > Salary > Bank > SACCO > Other) to prevent internal transfer double-counting.
3. A **single-case empirical evaluation** on a real 365-day M-Pesa ledger (1,215 transactions), with transparent disclosure of empirical limitations.
4. A **public CLI tool and web simulator** enabling borrowers, SACCOs, and microfinance institutions to compute and audit scores locally without server telemetry.

We do not claim that MBMCS outperforms machine-learning alternatives. We claim that it is *transparent*, *reproducible*, and *auditable*: three properties that black-box digital lenders often sacrifice in the pursuit of predictive power.

---

## 2. Theoretical Foundations & Related Literature

Jack & Suri (2014, *AER*) and Suri & Jack (2016, *Science*) established that M-Pesa improves household shock absorption, risk sharing, and lifts households out of extreme poverty. Berg et al. (2020, *RFS*) and Björkegren & Grissen (2020, *JDE*) proved that digital footprints predict loan default as reliably as conventional credit bureau files. Karlan & Zinman (2009, *Econometrica*) demonstrated that information asymmetries remain the primary binding constraint for unbanked micro-borrowers. 

Our contribution is not to outperform these predictive methods, but to provide an **interpretable alternative** that a single borrower can execute on their own statement, and that a loan officer can audit line by line without relying on proprietary black-box scoring APIs.

---

## 3. Mathematical Framework & Methodology

### 3.1 Data Schema & Ledger Formalization

A transaction ledger is formally defined as an ordered sequence of transactions:

$$\mathbf{T} = \{(t_i, d_i, p_i, w_i, b_i)\}_{i=1}^N$$

where $t_i$ represents the timestamp, $d_i$ the transaction description, $p_i \ge 0$ the credit inflow, $w_i \ge 0$ the debit outflow, and $b_i \ge 0$ the post-transaction running balance. The observation window spans $D$ days, giving a normalized interval of $M$ monthly equivalents:

$$M = \frac{D}{30.4375}$$

### 3.2 Feature Definitions

**Mean and Median Monthly Inflow:**

$$\mu_{\text{in}} = \frac{1}{M} \sum_{i=1}^N p_i, \quad \tilde{\mu}_{\text{in}} = \operatorname{median}(\text{monthly inflow sums})$$

**Coefficient of Variation (Income Stability):**

$$\text{CV} = \frac{\sigma_{\text{monthly\_inflow}}}{\mu_{\text{in}}}$$

**Investment-Adjusted Net Flow Ratio:** Outflows to Money Market Funds ($W_{\text{MMF}}$) and SACCO shares ($W_{\text{SACCO}}$) are excluded from the expenditure denominator because they represent asset accumulation rather than operational consumption:

$$r^*_{\text{net}} = \frac{\sum_{i=1}^N p_i}{\sum_{i=1}^N w_i - W_{\text{MMF}} - W_{\text{SACCO}}}$$

**Source Concentration (HHI):** Income categories are assigned to each incoming transaction in strict priority order: `MMF > Salary > Bank > SACCO > Other`. This priority ranking guarantees that every transaction belongs to exactly one category. Category shares $s_k$ are computed over these mutually exclusive partitions:

$$\text{HHI} = \sum_{k=1}^K s_k^2 \in \left[\frac{1}{K}, 1\right]$$

where $K$ denotes the number of active income categories. A value approaching $1/K$ reflects balanced diversity; a value near 1 indicates single-source dependence.

### 3.3 Five-Pillar Scoring Architecture

The composite credit score maps directly to the standard 300–850 range by anchoring a base score of 300 and distributing 550 total points across five distinct economic pillars:

$$S = 300 + \sum_{k=1}^5 P_k \in [300, 850]$$

**P1 — Cash Flow Velocity (Max 137.5 points):**

$$P_1 = 80 \left(1 - e^{-\mu_{\text{in}} / 60{,}000}\right) + 40 \left(1 - e^{-(N/D) / 3}\right) + 17.5 \min\left(\frac{M}{12}, 1\right)$$

**P2 — Repayment Capacity (Max 137.5 points):**

$$P_2 = 70 e^{-\text{CV}} + 67.5 \left(1 - e^{-(r^*_{\text{net}} - 1)^+ \cdot 10}\right)$$

**P3 — Liquidity Buffer (Max 110.0 points):**

$$P_3 = 70 \min\left(\frac{\bar{b}}{10{,}000}, 1\right) + 40 \left(1 - e^{-\min(b_i) / 1{,}000}\right)$$

**P4 — Financial Diversity (Max 110.0 points):**

$$P_4 = 110 \left(\frac{1 - \text{HHI}}{1 - 1/K}\right)$$

**P5 — Credit Discipline (Max 55.0 points):** Parameterized by days past due $\delta$, subject to a 50% cap (27.5 points) if any overdraft penalty (e.g. Fuliza default) occurs:

$$P_5 = \begin{cases} 0 & \text{if } \delta > 30 \\ 55 e^{-\delta / 10} & \text{otherwise} \end{cases}$$

### 3.4 Decision Rules & Safe Principal Capacity

Given requested principal $L$, tenure $T$ months, flat borrowing cost $r$, and debt-service burden threshold $\theta = 25\%$:

$$R_m = \frac{L(1 + r)}{T}, \quad \beta = \left(\frac{R_m}{\mu_{\text{in}}}\right) \times 100\%$$

The underwriting decision $\Omega$ evaluates:

$$\Omega = \begin{cases} \text{DECLINED} & \text{if } \delta > 30 \\ \text{MODERATE} & \text{if } \beta > \theta \text{ and } \delta \le 30 \\ \text{APPROVED} & \text{if } \beta \le \theta \text{ and } \delta \le 30 \end{cases}$$

Two versions of the maximum safe principal are reported:

$$L_{\text{mean}} = \frac{\mu_{\text{in}} \cdot (\theta/100) \cdot T}{1 + r}, \quad L_{\text{median}} = \frac{\tilde{\mu}_{\text{in}} \cdot (\theta/100) \cdot T}{1 + r}$$

The **median-based figure is the conservative one** and is the one we recommend for underwriting. Because informal income volatility is high in this population, the arithmetic mean is heavily distorted by peak months and can substantially overstate sustainable debt-service capacity.

---

## 4. Empirical Case Study & Results

### 4.1 Dataset & Applicant Profile

- **Applicant:** First author.
- **Period:** 22 March 2024 to 22 March 2025 (364 days).
- **Transactions:** 1,215 completed operations (3.34 per day).
- **Total Inflow:** KES 1,008,803.00.
- **Total Outflow:** KES 1,010,554.00.
- **Net Flow:** KES -1,751.00 (-0.17% net annual margin).

### 4.2 Feature Summary Table

| Feature Metric | Recorded Value | Operational Interpretation |
|---|---|---|
| Observation Window | 364 days | Full annual economic cycle |
| Completed Transactions | 1,215 | 3.34 operations per day |
| Gross Inflow Volume | KES 1,008,803.00 | Total cumulative credits |
| Gross Outflow Volume | KES 1,010,554.00 | Total cumulative debits |
| Net Flow Margin | KES -1,751.00 | Balanced annual flow (-0.17%) |
| Mean Monthly Inflow | KES 84,355.61 | Heavily skewed by peak periods |
| Median Monthly Inflow | KES 37,790.00 | Robust baseline cash flow |
| Coefficient of Variation (CV) | 1.13 | Substantial monthly volatility |
| Mean Running Balance | KES 6,394.97 | Operational float |
| Minimum Running Balance | KES 0.15 | Thin liquidity floor |
| Peak Running Balance | KES 104,439.15 | Temporary liquidity accumulation |
| Salary Income (Caritas MFB) | KES 743,310.00 | 68 verified credit events |
| Money Market Fund (MMF) Inflow | KES 117,500.00 | Investment redemptions |
| MMF Outflow (Invested) | KES 119,600.00 | Excluded from consumption debits |
| Interbank Inflows | KES 66,980.00 | 140 interbank transactions |
| SACCO Capital Contributions | KES 6,100.00 | 11 capital building deposits |
| Days Past Due (DPD) | 0 | Zero default history |
| Income HHI (K = 4 categories) | 0.5673 | Effective source diversification |
| **Investment-Adjusted Net Flow** | **1.1401** | **Positive operational savings ratio** |

### 4.3 Pillar Scores & Breakdown

| Scoring Pillar | Raw Score | Max Ceiling | Fulfilment Rate |
|---|---|---|---|
| P1: Cash Flow Velocity | 104.75 | 137.5 | 76.2% |
| P2: Repayment Capacity | 73.48 | 137.5 | 53.4% |
| P3: Liquidity Buffer | 44.77 | 110.0 | 40.7% |
| P4: Financial Diversity | 63.46 | 110.0 | 57.7% |
| P5: Credit Discipline | 55.00 | 55.0 | 100.0% |
| **Total Composite Score (S)** | **641.00** | **850.0** | **75.4%** |

**Tier Rating:** Grade A (Tier 3 Fair)  
**Underwriting Decision:** CONDITIONAL APPROVAL

### 4.4 Diagnostic Interpretation of the Pillars

- **P1 Cash Flow Velocity (76.2% fulfillment):** The ledger shows brisk operational transaction velocity (3.34 operations per day) and substantial cumulative inflow (KES 84,356 mean). However, the ceiling of 137.5 is not fully reached because mean volume falls below the highest saturation threshold.
- **P2 Repayment Capacity (53.4% fulfillment):** This pillar is the primary bottleneck. Monthly inflows range widely from roughly KES 9,000 to over KES 280,000, producing a high CV of 1.13. This instability reduces the stability term to 22.6 out of 70 points, though the investment-adjusted net flow ratio of 1.14 contributes a healthy 50.9 points.
- **P3 Liquidity Buffer (40.7% fulfillment):** The average balance of KES 6,395 earns 44.8 points of the 70 available on the mean float term, but the minimum balance of KES 0.15 contributes virtually zero points to the liquidity floor term.
- **P4 Financial Diversity (57.7% fulfillment):** The HHI of 0.5673 across four active channels (salary, MMF, interbank, and peer transfers) yields 63.5 points. While Caritas MFB salary credits form the primary anchor, significant MMF and interbank liquidity provide genuine structural diversification.
- **P5 Credit Discipline (100.0% fulfillment):** The applicant maintains an unblemished record with zero days past due, zero late fees, and zero Fuliza overdraft distress over the entire 364-day audit.

### 4.5 Safe Principal Comparison

| Underwriting Basis | Sizing Formula | Calculated Principal | Underwriting Assessment |
|---|---|---|---|
| Mean Monthly Inflow | `(84,356 × 0.25 × 3) / 1.10` | KES 57,515.00 | Aggressive; exposes borrower to debt distress |
| **Median Inflow (Recommended)** | **`(37,790 × 0.25 × 3) / 1.10`** | **KES 25,766.00** | **Conservative; resilient against cash volatility** |

The arithmetic mean-based figure is nearly **2.2×** the median-based figure. Given the CV of 1.13, sizing credit to the mean guarantees that during lean months, debt service would consume over 70% of actual available cash flow.

### 4.6 Simulator Scenarios

| Scenario Profile | Principal | Tenure | Interest (r) | DPD | Burden (Mean) | Decision |
|---|---|---|---|---|---|---|
| Baseline Facility | KES 30,000 | 3 mo | 10% | 0 | 13.0% | **APPROVED** |
| Growth Capital | KES 50,000 | 6 mo | 12% | 0 | 11.1% | **APPROVED** |
| Over-Leverage Stress | KES 80,000 | 1 mo | 10% | 0 | 104.3% | **REVIEW** |
| Delinquency Stress | KES 30,000 | 3 mo | 10% | 45 | 13.0% | **DECLINED** |

On the median-inflow basis, the baseline KES 30,000 facility's debt-service burden rises to **29.1%**, crossing the conservative 25% burden threshold. This divergence validates the diagnostic power of dual-metric reporting: it flags hidden cash-flow vulnerability without disqualifying an otherwise disciplined borrower.

---

## 5. Limitations

1. **Single-Case Dataset:** The framework is demonstrated on a single individual's ledger. It lacks cross-sectional demographic variance and multi-year macroeconomic stress cycles.
2. **Absence of Default Calibration:** The current score of 641 is an ordinal ranking index, not an empirically calibrated Probability of Default (PD). True PD estimation requires default ground-truth labels.
3. **Mean-Median Divergence:** Arithmetic mean inflows diverge from median cash flow by a factor of 2.2×. Underwriting models must default to median metrics to prevent over-indebtedness.
4. **Rule-Based Heuristic Categorization:** Income stream classification relies on priority string matching against known transactional patterns. Ambiguous transaction descriptions require machine-learning classification refinement.
5. **Demographic Fairness Audits Deferred:** A single applicant cannot support algorithmic bias audits across protected demographic characteristics such as gender, age, and county.
6. **Absence of Credit Bureau Benchmarking:** The applicant possesses no formal Credit Reference Bureau file, preventing direct comparative ROC-AUC benchmarking against conventional bureau models.

---

## 6. Multi-Institutional Empirical Validation Plan

We invite Savings and Credit Cooperative Organizations (SACCOs) and Microfinance Institutions (MFIs) across Nakuru County and Western Kenya to collaborate on an out-of-time validation study.

| Validation Dimension | Technical Protocol & Implementation Parameters |
|---|---|
| **Sample Cohort** | 500 to 1,000 anonymized micro-loans disbursed between January and December 2024 |
| **Partitioning Protocol** | Training (Jan–Jun 2024); Validation (Jul–Sep 2024); Out-of-Time Test (Oct–Dec 2024) |
| **Required Data Fields** | 6–12 month M-Pesa CSV/PDF, loan terms, repayment schedule, default outcome (90+ DPD), county, gender |
| **Candidate Benchmark Models** | (1) Fixed-weight MBMCS; (2) MBMCS + Logistic Regression; (3) MBMCS + XGBoost; (4) CRB Score; (5) Hybrid CRB + MBMCS |
| **Target Statistical Metrics** | AUC-ROC ≥ 0.70, Kolmogorov-Smirnov (KS) ≥ 0.30, Gini ≥ 0.30, Brier Score, PSI < 0.10 |
| **Fairness & Parity Thresholds** | AUC difference by gender < 0.05; AUC difference across rural/urban counties < 0.10 |

---

## 7. Ethics, Regulatory Compliance & Data Protection

Unlike extractive commercial digital credit applications that scrape borrower smartphone contacts, GPS coordinates, and personal SMS logs, MBMCS enforces strict **privacy by design**. Scoring inputs are restricted entirely to user-provided, self-exported transaction ledgers.

The architecture fully adheres to the **Kenya Data Protection Act 2019 (Act No. 24 of 2019)** and aligns with the Central Bank of Kenya Digital Credit Providers Regulations 2022 (Legal Notice No. 46 of 2022):

- **Section 25 (Principles of Data Protection):** Enforces strict purpose limitation and data minimisation.
- **Section 29 (Duty of Confidentiality):** All identifiable fields (account numbers, employer names, paybill IDs) are expunged during ingestion.
- **Section 35 (Automated Decision-Making Rights):** Guarantees the applicant's right to obtain an audit trail and understand the exact economic logic governing their credit assessment.

Furthermore, the open-source CLI executable runs strictly within the borrower's local execution environment, transmitting zero transaction bytes across the public internet.

---

## 8. Conclusion

We present MBMCS, a transparent, reproducible framework for converting M-Pesa statements into a credit-relevant score. On a real 365-day ledger, it produces a score of 641/850 (Grade A, Tier 3 Fair) and a conservative safe principal of KES 25,766 over three months. The ledger has strong discipline (no delinquencies) and meaningful diversity (HHI = 0.567 over four categories), but is held back by income volatility (CV = 1.13) and thin liquidity floor (min balance KES 0.15).

This is a proof of concept. Validation awaits partner data. We release the code, the anonymised case data, and the reusable CLI tool under an open-source license, and we invite SACCOs and MFIs to partner on the out-of-time validation study.

---

## References

1. Berg, T., Burg, V., Gombović, A., & Puri, M. (2020). On the rise of fintechs: Credit scoring using digital footprints. *The Review of Financial Studies*, 33(7), 2845–2897. https://doi.org/10.1093/rfs/hhz099
2. Björkegren, D., & Grissen, D. (2020). Behavior revealed in public: Using mobile phone data to predict loan repayment. *Journal of Development Economics*, 145, 102473. https://doi.org/10.1016/j.jdeveco.2020.102473
3. Central Bank of Kenya (2022). Digital Credit Providers Regulations, 2022. *Kenya Gazette Legal Notice No. 46 of 2022* (published 18 March 2022). https://www.centralbank.go.ke/2022/03/21/central-bank-of-kenya-digital-credit-providers-regulations-2022/
4. Central Bank of Kenya, Kenya National Bureau of Statistics, & FSD Kenya (2021). 2021 FinAccess Household Survey Report. Nairobi, Kenya. https://www.knbs.or.ke/wp-content/uploads/2021/12/2021-Finaccess-Household-Survey-Report.pdf
5. Jack, W., & Suri, T. (2014). Risk sharing and transactions costs: Evidence from Kenya's mobile money revolution. *American Economic Review*, 104(1), 183–223. https://doi.org/10.1257/aer.104.1.183
6. Karlan, D., & Zinman, J. (2009). Observing unobservables: Identifying information asymmetries with a consumer credit field experiment. *Econometrica*, 77(6), 1993–2008. https://doi.org/10.3982/ECTA5781
7. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9780511803161
8. Republic of Kenya (2019). The Data Protection Act, 2019. *Kenya Gazette Act No. 24 of 2019*. https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2019-11-25
9. Suri, T., & Jack, W. (2016). The long-run poverty and gender impacts of mobile money. *Science*, 354(6317), 1288–1292. https://doi.org/10.1126/science.aah5309
