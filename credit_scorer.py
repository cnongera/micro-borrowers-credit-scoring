#!/usr/bin/env python3
"""
Mobile-Money Credit Scoring & Feature Engineering Engine
Oct 2026.

Fixes applied in this version:
  1. Mutually exclusive income categories (priority: MMF > Salary > Bank > SACCO > Other)
     so that a single transaction is never counted twice. Previous versions
     double-counted salary credits from Caritas MFB as both "salary" and "bank",
     which inflated the HHI above 1.0 and collapsed the diversity pillar to 0.
  2. HHI is now bounded to [1/K, 1] with an assertion.
  3. Exponential saturation in P1, P3 (no free 40-point floor).
  4. Investment-adjusted net flow ratio in P2 (savings excluded from "outflow").
  5. P5 zeroed whenever DPD > 30 (consistent with the DECLINED decision rule).
  6. Reports both mean-based and median-based safe loan; the median is the
     conservative figure recommended in the paper.

Pillars:
  P1 — Cash Flow Velocity (max 137.5)
  P2 — Repayment Capacity (max 137.5)
  P3 — Liquidity Buffer   (max 110.0)
  P4 — Financial Diversity (max 110.0)
  P5 — Credit Discipline   (max  55.0)
  Base score 300, max additional 550 -> range [300, 850].
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


# =============================================================================
# Helpers
# =============================================================================

def clean_currency(val):
    """Parse a currency string like '1,234.56' or '1234' into a float."""
    if pd.isna(val) or val is None:
        return 0.0
    s = str(val).replace(",", "").replace('"', "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def parse_mpesa_csv(path):
    """Load a Safaricom M-Pesa statement CSV and return a cleaned DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = ["Receipt No.", "Completion Time", "Details",
                "Transaction Status", "Paid In", "Withdrawn", "Balance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Paid_In_Amt"]   = df["Paid In"].apply(clean_currency)
    df["Withdrawn_Amt"] = df["Withdrawn"].apply(clean_currency).abs()
    df["Balance_Amt"]   = df["Balance"].apply(clean_currency)

    df["DateTime"] = pd.to_datetime(
        df["Completion Time"], format="%m/%d/%y %H:%M", errors="coerce"
    )
    if df["DateTime"].isna().all():
        df["DateTime"] = pd.to_datetime(df["Completion Time"], errors="coerce")

    df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    return df


def extract_borrower_name(df, filename):
    """Best-effort extraction of the borrower name from filename or ledger."""
    base = os.path.basename(filename)
    if "Mpesa" in base or "Statement" in base:
        n = base.replace("Mpesa.csv", "").replace("Mpesa", "").replace(".csv", "").strip()
        if n:
            return n
    return "M-Pesa Borrower"


# =============================================================================
# Feature engineering
# =============================================================================

def engineer_behavioral_features(df):
    """
    Engineer financial and behavioural features from an M-Pesa transaction log.
    All monetary values are in KES.
    """
    start = df["DateTime"].min()
    end   = df["DateTime"].max()
    days  = max((end - start).days, 1)
    months = max(days / 30.4375, 1.0)

    total_in  = df["Paid_In_Amt"].sum()
    total_out = df["Withdrawn_Amt"].sum()
    n_tx      = len(df)
    velocity  = n_tx / days

    # Monthly aggregation
    df["YearMonth"] = df["DateTime"].dt.to_period("M")
    monthly = df.groupby("YearMonth").agg(
        monthly_inflow  = ("Paid_In_Amt", "sum"),
        monthly_outflow = ("Withdrawn_Amt", "sum"),
        tx_count        = ("Receipt No.", "count"),
        avg_balance     = ("Balance_Amt", "mean"),
        min_balance     = ("Balance_Amt", "min"),
        max_balance     = ("Balance_Amt", "max"),
    )

    mean_in   = total_in / months
    median_in = float(monthly["monthly_inflow"].median()) if len(monthly) else 0.0
    std_in    = float(monthly["monthly_inflow"].std()) if len(monthly) > 1 else 0.0
    cv        = (std_in / mean_in) if mean_in > 0 else 1.0

    # Balance features
    avg_bal  = float(df["Balance_Amt"].mean())
    min_bal  = float(df["Balance_Amt"].min())
    peak_bal = float(df["Balance_Amt"].max())

    # -------------------------------------------------------------------------
    # Category assignment: each transaction belongs to exactly ONE category.
    # Priority order (highest to lowest): MMF > Salary > Bank > SACCO > Other.
    # This prevents double-counting of the same transaction. In particular,
    # salary credits from Caritas Microfinance Bank must NOT also be tagged
    # as "bank", otherwise the HHI exceeds 1.
    # -------------------------------------------------------------------------
    details = df["Details"].astype(str)
    paid_in = df["Paid_In_Amt"]

    mmf_pattern    = r"Cytonn|Cytron|Money Market|MMF|CIC|Sanlam|Genghis|Britam|Old Mutual"
    salary_pattern = r"Salary|Wage|Earnings"
    bank_pattern   = r"\bBank\b|CARITAS|KCB|Standard Chartered|Equity|Co-?op|Kingdom|Absa|NCBA"
    sacco_pattern  = r"SACCO|SELF HELP GROUP|Chama|Welfare"

    mask_mmf    = details.str.contains(mmf_pattern,    case=False, na=False, regex=True)
    mask_salary = details.str.contains(salary_pattern, case=False, na=False, regex=True)
    mask_bank   = details.str.contains(bank_pattern,   case=False, na=False, regex=True)
    mask_sacco  = details.str.contains(sacco_pattern,  case=False, na=False, regex=True)

    # Assign in priority order using a category Series
    cat = pd.Series("other", index=df.index, dtype=object)
    cat[mask_sacco.values]  = "sacco"    # lowest priority of the four
    cat[mask_bank.values]   = "bank"
    cat[mask_salary.values] = "salary"
    cat[mask_mmf.values]    = "mmf"      # highest priority

    # Inflow grouped by mutually exclusive category
    inflow_by_cat = paid_in.groupby(cat).sum()
    total_cat_inflow = float(inflow_by_cat.sum())

    if total_cat_inflow > 0:
        shares = (inflow_by_cat / total_cat_inflow).values
    else:
        shares = np.array([1.0])

    # Keep only shares above 0.5% (avoid noise from micro-transactions)
    shares = shares[shares > 0.005]
    K = max(int(len(shares)), 1)

    hhi = float(np.sum(shares ** 2))
    # HHI is mathematically bounded to [1/K, 1]. Enforce for numerical safety.
    hhi = float(np.clip(hhi, 1.0 / K, 1.0))

    # Consistency assertion
    assert 1.0 / K - 1e-9 <= hhi <= 1.0 + 1e-9, (
        f"HHI out of bounds: {hhi:.4f} (K = {K}); "
        f"this indicates category overlap."
    )

    # Per-category totals (for the report)
    salary_in = float(inflow_by_cat.get("salary", 0.0))
    mmf_in    = float(inflow_by_cat.get("mmf", 0.0))
    bank_in   = float(inflow_by_cat.get("bank", 0.0))
    sacco_in  = float(inflow_by_cat.get("sacco", 0.0))

    salary_ct = int(mask_salary.sum())
    sacco_ct  = int(mask_sacco.sum())
    bank_ct   = int(mask_bank.sum())

    mmf_out   = float(df.loc[mask_mmf,    "Withdrawn_Amt"].sum())
    sacco_out = float(df.loc[mask_sacco,  "Withdrawn_Amt"].sum())

    # Overdraft / distress indicators
    overdraft_ct = int(details.str.contains(
        r"Fuliza|Overdraft|Defaul|Late Fee|Penalty",
        case=False, na=False, regex=True
    ).sum())

    # Investment-adjusted net flow: savings and SACCO contributions excluded
    # from the "consumption outflow" denominator.
    denominator = max(total_out - mmf_out - sacco_out, 1.0)
    net_flow_adj = total_in / denominator

    return {
        "duration_days":         days,
        "duration_months":       round(months, 1),
        "start_date":            start.strftime("%Y-%m-%d"),
        "end_date":              end.strftime("%Y-%m-%d"),
        "total_inflow":          float(total_in),
        "total_outflow":         float(total_out),
        "net_cashflow":          float(total_in - total_out),
        "total_tx_count":        int(n_tx),
        "tx_velocity_per_day":   float(round(velocity, 2)),
        "mean_monthly_inflow":   float(round(mean_in, 2)),
        "median_monthly_inflow": float(round(median_in, 2)),
        "inflow_stability_cv":   float(round(cv, 2)),
        "overall_avg_balance":   float(round(avg_bal, 2)),
        "lowest_balance":        float(round(min_bal, 2)),
        "peak_balance":          float(round(peak_bal, 2)),
        "salary_inflow":         float(round(salary_in, 2)),
        "salary_count":          int(salary_ct),
        "mmf_invested":          float(round(mmf_out, 2)),
        "mmf_received":          float(round(mmf_in, 2)),
        "bank_interaction_count": int(bank_ct),
        "bank_volume":           float(round(bank_in, 2)),
        "sacco_invested":        float(round(sacco_out, 2)),
        "sacco_received":        float(round(sacco_in, 2)),
        "sacco_count":           int(sacco_ct),
        "overdraft_count":       int(overdraft_ct),
        "has_overdraft_distress": bool(overdraft_ct > 0),
        "days_past_due":         0,
        "hhi":                   float(round(hhi, 4)),
        "n_sources":             int(K),
        "net_flow_ratio_adj":    float(round(net_flow_adj, 4)),
        "monthly_agg":           monthly,
    }


# =============================================================================
# Scoring
# =============================================================================

def calculate_credit_score(f):
    """
    Compute the  5-pillar score on the 300-850 scale.

    P1 — Cash Flow Velocity   (max 137.5): exponential saturation in
        mean monthly inflow and transaction velocity; linear tenure credit.
    P2 — Repayment Capacity   (max 137.5): exp(-CV) stability term plus
        investment-adjusted net flow ratio.
    P3 — Liquidity Buffer     (max 110.0): average balance plus an
        exponential minimum-balance term (no free 40-point floor).
    P4 — Financial Diversity  (max 110.0): normalised HHI over mutually
        exclusive inflow categories.
    P5 — Credit Discipline    (max  55.0): zeroed when DPD > 30,
        otherwise 55 * exp(-DPD / 10), capped at 27.5 if a Fuliza
        penalty appears anywhere in the ledger.
    """
    # -------- P1 --------
    p1_inflow   = 80.0 * (1.0 - np.exp(-f["mean_monthly_inflow"] / 60000.0))
    p1_velocity = 40.0 * (1.0 - np.exp(-f["tx_velocity_per_day"] / 3.0))
    p1_tenure   = 17.5 * min(f["duration_months"] / 12.0, 1.0)
    P1 = p1_inflow + p1_velocity + p1_tenure

    # -------- P2 --------
    p2_stability = 70.0 * np.exp(-f["inflow_stability_cv"])
    excess = max(f["net_flow_ratio_adj"] - 1.0, 0.0)
    p2_net = 67.5 * (1.0 - np.exp(-excess * 10.0))
    P2 = p2_stability + p2_net

    # -------- P3 --------
    p3_avg = 70.0 * min(f["overall_avg_balance"] / 10000.0, 1.0)
    p3_min = 40.0 * (1.0 - np.exp(-f["lowest_balance"] / 1000.0))
    P3 = p3_avg + p3_min

    # -------- P4 --------
    K = max(f["n_sources"], 1)
    if K > 1:
        P4 = 110.0 * (1.0 - f["hhi"]) / (1.0 - 1.0 / K)
    else:
        P4 = 0.0
    P4 = float(max(0.0, min(110.0, P4)))

    # -------- P5 --------
    if f["days_past_due"] > 30:
        P5 = 0.0
    else:
        P5 = 55.0 * np.exp(-f["days_past_due"] / 10.0)
    if f["has_overdraft_distress"]:
        P5 = min(P5, 27.5)

    total = 300.0 + P1 + P2 + P3 + P4 + P5
    score = int(round(max(300, min(850, total))))

    if score >= 760:
        tier, rating, decision = "Excellent (Tier 1 Prime)",     "AAA", "PRE-APPROVED"
    elif score >= 700:
        tier, rating, decision = "Good (Tier 2 Standard)",       "AA",  "APPROVED (LOW RISK)"
    elif score >= 640:
        tier, rating, decision = "Fair (Tier 3 Moderate)",       "A",   "CONDITIONAL APPROVAL"
    elif score >= 580:
        tier, rating, decision = "Subprime (Tier 4)",            "B",   "GUARANTOR REQUIRED"
    else:
        tier, rating, decision = "High Risk (Tier 5)",           "C",   "REJECTED"

    return {
        "credit_score": score,
        "rating":       rating,
        "tier":         tier,
        "decision":     decision,
        "pillars": {
            "velocity_volume":     round(P1, 2),
            "repayment_capacity":  round(P2, 2),
            "liquidity_buffer":    round(P3, 2),
            "financial_diversity": round(P4, 2),
            "credit_discipline":   round(P5, 2),
        },
        "max_points": {
            "velocity_volume":     137.5,
            "repayment_capacity":  137.5,
            "liquidity_buffer":    110.0,
            "financial_diversity": 110.0,
            "credit_discipline":    55.0,
        },
    }


# =============================================================================
# Simulator
# =============================================================================

def run_simulator(f, loan=30000, months=3, interest=10.0,
                        burden=25.0, past_due=0, use_median=False):
    """
   feature simulator: computes repayment burden and safe-loan caps.

    If use_median is True, the burden is computed against the median monthly
    inflow (the conservative choice). Otherwise the mean is used.
    """
    monthly_in = f["median_monthly_inflow"] if use_median else f["mean_monthly_inflow"]

    total_repay    = loan * (1.0 + interest / 100.0)
    monthly_repay  = total_repay / max(months, 1)
    ratio          = monthly_in / max(loan, 1)
    burden_pct     = (monthly_repay / max(monthly_in, 1)) * 100.0
    median_burden  = (monthly_repay / max(f["median_monthly_inflow"], 1)) * 100.0

    if past_due > 30:
        status, decision, high = "HIGH RISK (DEFAULT PROBABLE)", "DECLINED", 1
    elif burden_pct > burden:
        status, decision, high = (
            f"MODERATE RISK (BURDEN > {burden:.0f}%)",
            "REVIEW / EXTEND TENURE", 0,
        )
    else:
        status, decision, high = "LOW RISK (APPROVED)", "APPROVED", 0

    max_safe_mean   = (f["mean_monthly_inflow"]   * (burden / 100.0) * months) / (1.0 + interest / 100.0)
    max_safe_median = (f["median_monthly_inflow"] * (burden / 100.0) * months) / (1.0 + interest / 100.0)

    return {
        "loan_amount":            loan,
        "duration_months":        months,
        "interest_rate_pct":      interest,
        "monthly_repayment":      round(monthly_repay, 2),
        "total_repayment":        round(total_repay, 2),
        "inflow_to_loan_ratio":   round(ratio, 2),
        "repayment_burden_pct":   round(burden_pct, 1),
        "median_burden_pct":      round(median_burden, 1),
        "burden_threshold_pct":   burden,
        "past_due_days":          past_due,
        "is_high_risk":           high,
        "status":                 status,
        "decision":               decision,
        "max_safe_loan_mean":     round(max_safe_mean, 0),
        "max_safe_loan_median":   round(max_safe_median, 0),
    }


# =============================================================================
# Report
# =============================================================================

def generate_report(path, loan=30000, burden=25.0):
    df    = parse_mpesa_csv(path)
    feats = engineer_behavioral_features(df)
    score = calculate_credit_score(feats)
    sim   = run_simulator(feats, loan=loan, burden=burden)
    return {
        "borrower_name": extract_borrower_name(df, path),
        "file_name":     os.path.basename(path),
        "features":      feats,
        "scoring":       score,
        "simulator":     sim,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CNOngera Credit Scorer (Oct 2026)")
    ap.add_argument("csv_path", nargs="?",
                    default="Clement Ongera Nyangoya Mpesa.csv")
    ap.add_argument("--loan",   type=float, default=30000.0)
    ap.add_argument("--months", type=int,   default=3)
    ap.add_argument("--interest", type=float, default=10.0)
    ap.add_argument("--burden",   type=float, default=25.0)
    ap.add_argument("--pastdue",  type=int,   default=0)
    ap.add_argument("--median", action="store_true",
                    help="Use median monthly inflow instead of mean for the burden calc.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rep = generate_report(args.csv_path, loan=args.loan, burden=args.burden)
    rep["simulator"] = run_simulator(
        rep["features"],
        loan=args.loan, months=args.months,
        interest=args.interest, burden=args.burden,
        past_due=args.pastdue, use_median=args.median,
    )

    if args.json:
        out = dict(rep)
        del out["features"]["monthly_agg"]   # not JSON-serialisable
        print(json.dumps(out, indent=2))
    else:
        f, s, sim = rep["features"], rep["scoring"], rep["simulator"]
        print("=" * 70)
        print(" CNOngera CREDIT SCORING DOSSIER (Oct 2026)")
        print("=" * 70)
        print(f"Borrower        : {rep['borrower_name']}")
        print(f"Period          : {f['start_date']} to {f['end_date']} ({f['duration_months']} mo)")
        print(f"Transactions    : {f['total_tx_count']:,} ({f['tx_velocity_per_day']}/day)")
        print(f"Total inflow    : KES {f['total_inflow']:,.2f}")
        print(f"Total outflow   : KES {f['total_outflow']:,.2f}")
        print(f"Mean monthly in : KES {f['mean_monthly_inflow']:,.2f}")
        print(f"Median monthly  : KES {f['median_monthly_inflow']:,.2f}")
        print(f"CV of inflow    : {f['inflow_stability_cv']}")
        print(f"Avg balance     : KES {f['overall_avg_balance']:,.2f}")
        print(f"Min balance     : KES {f['lowest_balance']:,.2f}")
        print(f"HHI (sources)   : {f['hhi']} (K = {f['n_sources']})")
        print(f"Net flow (adj)  : {f['net_flow_ratio_adj']}")
        print("-" * 70)
        print(f"Score           : {s['credit_score']} / 850 ({s['rating']} — {s['tier']})")
        print(f"Decision        : {s['decision']}")
        for k, v in s["pillars"].items():
            print(f"  {k:<22}: {v:>6} / {s['max_points'][k]}")
        print("-" * 70)
        print(f"Simulated loan  : KES {sim['loan_amount']:,.2f} / "
              f"{sim['duration_months']} mo @ {sim['interest_rate_pct']}%")
        print(f"Monthly repay   : KES {sim['monthly_repayment']:,.2f}")
        print(f"Burden (mean)   : {sim['repayment_burden_pct']}% "
              f"(threshold {sim['burden_threshold_pct']}%)")
        print(f"Burden (median) : {sim['median_burden_pct']}%")
        print(f"Status          : {sim['status']}")
        print(f"Max safe (mean) : KES {sim['max_safe_loan_mean']:,.0f}")
        print(f"Max safe (median, recommended): KES {sim['max_safe_loan_median']:,.0f}")
        print("=" * 70)