#!/usr/bin/env python3
"""
Longitudinal analysis of Dortmund Vital Study (5-year follow-up).
Within-subject change in PTA metrics: the strongest evidence for aging effect.
"""

import numpy as np
import pandas as pd
from scipy import stats
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "results" / "dortmund_longitudinal" / "longitudinal_pairs.csv"
OUT_PATH = ROOT / "results" / "longitudinal_stats"
OUT_PATH.mkdir(exist_ok=True)


def cohens_d_paired(x):
    """Cohen's d for paired samples (d_z = mean / std)."""
    return np.nanmean(x) / (np.nanstd(x, ddof=1) + 1e-15)


def bootstrap_ci(x, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap 95% CI for the mean."""
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    means = np.array([np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def analyze_band(df, band):
    """Full longitudinal analysis for one band."""
    s1_col = f"prop_positive_{band}_s1"
    s2_col = f"prop_positive_{band}_s2"
    d_col = f"d_prop_positive_{band}"

    # Also analyze sigma and mean_A
    sigma_d_col = f"d_sigma_A_{band}"
    mean_d_col = f"d_mean_A_{band}"

    # Filter valid subjects
    valid = df[d_col].notna()
    n_valid = valid.sum()
    if n_valid < 10:
        return {"band": band, "n_valid": int(n_valid), "skipped": True}

    delta = df.loc[valid, d_col].values
    s1 = df.loc[valid, s1_col].values
    s2 = df.loc[valid, s2_col].values
    ages = df.loc[valid, "age_baseline"].values

    # Basic descriptives
    mean_delta = np.nanmean(delta)
    median_delta = np.nanmedian(delta)
    std_delta = np.nanstd(delta, ddof=1)
    ci_lo, ci_hi = bootstrap_ci(delta)

    # Wilcoxon signed-rank test (H0: median change = 0)
    wilcox = stats.wilcoxon(delta[~np.isnan(delta)], alternative="two-sided")

    # Paired t-test
    ttest = stats.ttest_rel(s2[~np.isnan(s1) & ~np.isnan(s2)],
                            s1[~np.isnan(s1) & ~np.isnan(s2)])

    # One-sided: does prop_positive DECREASE?
    wilcox_onesided = stats.wilcoxon(delta[~np.isnan(delta)], alternative="less")

    # Cohen's d (paired)
    d_z = cohens_d_paired(delta)

    # Fraction showing decline
    frac_decline = np.mean(delta < 0)
    frac_increase = np.mean(delta > 0)
    frac_zero = np.mean(delta == 0)

    # Does change depend on baseline age?
    valid_both = ~np.isnan(delta) & ~np.isnan(ages)
    if valid_both.sum() > 10:
        rho_age, p_age = stats.spearmanr(ages[valid_both], delta[valid_both])
        r_age, p_r_age = stats.pearsonr(ages[valid_both], delta[valid_both])
    else:
        rho_age, p_age, r_age, p_r_age = np.nan, np.nan, np.nan, np.nan

    # Age-stratified: young (<40) vs old (>=50)
    young_mask = valid & (df["age_baseline"] < 40)
    old_mask = valid & (df["age_baseline"] >= 50)
    young_delta = df.loc[young_mask, d_col].dropna().values
    old_delta = df.loc[old_mask, d_col].dropna().values

    young_stats = {
        "n": len(young_delta),
        "mean": float(np.nanmean(young_delta)) if len(young_delta) > 0 else None,
        "median": float(np.nanmedian(young_delta)) if len(young_delta) > 0 else None,
    }
    old_stats = {
        "n": len(old_delta),
        "mean": float(np.nanmean(old_delta)) if len(old_delta) > 0 else None,
        "median": float(np.nanmedian(old_delta)) if len(old_delta) > 0 else None,
    }

    if len(young_delta) > 5:
        w_young = stats.wilcoxon(young_delta, alternative="less")
        young_stats["wilcoxon_p"] = float(w_young.pvalue)
        young_stats["d_z"] = float(cohens_d_paired(young_delta))
    if len(old_delta) > 5:
        w_old = stats.wilcoxon(old_delta, alternative="less")
        old_stats["wilcoxon_p"] = float(w_old.pvalue)
        old_stats["d_z"] = float(cohens_d_paired(old_delta))

    # Test-retest reliability
    valid_both_vals = ~np.isnan(s1) & ~np.isnan(s2)
    if valid_both_vals.sum() > 10:
        icc_r, icc_p = stats.pearsonr(s1[valid_both_vals], s2[valid_both_vals])
    else:
        icc_r, icc_p = np.nan, np.nan

    # Sigma(A) change
    sigma_delta_vals = df.loc[valid, sigma_d_col].dropna().values
    sigma_result = {}
    if len(sigma_delta_vals) > 10:
        w_sigma = stats.wilcoxon(sigma_delta_vals, alternative="two-sided")
        sigma_result = {
            "n": len(sigma_delta_vals),
            "mean_change": float(np.nanmean(sigma_delta_vals)),
            "wilcoxon_p": float(w_sigma.pvalue),
            "d_z": float(cohens_d_paired(sigma_delta_vals)),
        }

    # Mean A change
    mean_delta_vals = df.loc[valid, mean_d_col].dropna().values
    meanA_result = {}
    if len(mean_delta_vals) > 10:
        w_mean = stats.wilcoxon(mean_delta_vals, alternative="two-sided")
        meanA_result = {
            "n": len(mean_delta_vals),
            "mean_change": float(np.nanmean(mean_delta_vals)),
            "wilcoxon_p": float(w_mean.pvalue),
            "d_z": float(cohens_d_paired(mean_delta_vals)),
        }

    result = {
        "band": band,
        "n_valid": int(n_valid),
        "prop_positive": {
            "mean_s1": float(np.nanmean(s1)),
            "mean_s2": float(np.nanmean(s2)),
            "mean_delta": float(mean_delta),
            "median_delta": float(median_delta),
            "std_delta": float(std_delta),
            "ci_95": [float(ci_lo), float(ci_hi)],
            "wilcoxon_stat": float(wilcox.statistic),
            "wilcoxon_p_twosided": float(wilcox.pvalue),
            "wilcoxon_p_onesided_less": float(wilcox_onesided.pvalue),
            "paired_t_stat": float(ttest.statistic),
            "paired_t_p": float(ttest.pvalue),
            "cohens_d_z": float(d_z),
            "frac_decline": float(frac_decline),
            "frac_increase": float(frac_increase),
        },
        "delta_vs_age": {
            "spearman_rho": float(rho_age),
            "spearman_p": float(p_age),
            "pearson_r": float(r_age),
            "pearson_p": float(p_r_age),
        },
        "age_stratified": {
            "young_lt40": young_stats,
            "old_gte50": old_stats,
        },
        "test_retest": {
            "r": float(icc_r),
            "p": float(icc_p),
        },
        "sigma_A_change": sigma_result,
        "mean_A_change": meanA_result,
    }
    return result


def main():
    print("Loading longitudinal data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  N = {len(df)} subjects")
    print(f"  Age range: {df['age_baseline'].min():.0f}-{df['age_baseline'].max():.0f}")
    print(f"  Sex: {df['sex'].value_counts().to_dict()}")

    bands = ["delta", "theta", "alpha", "beta", "low_gamma"]
    results = {}

    for band in bands:
        print(f"\n{'='*60}")
        print(f"  BAND: {band}")
        print(f"{'='*60}")
        r = analyze_band(df, band)
        results[band] = r

        if r.get("skipped"):
            print(f"  Skipped (n={r['n_valid']})")
            continue

        pp = r["prop_positive"]
        print(f"  N valid: {r['n_valid']}")
        print(f"  S1 mean: {pp['mean_s1']:.4f}, S2 mean: {pp['mean_s2']:.4f}")
        print(f"  Mean delta: {pp['mean_delta']:.4f} [{pp['ci_95'][0]:.4f}, {pp['ci_95'][1]:.4f}]")
        print(f"  Wilcoxon (two-sided): p = {pp['wilcoxon_p_twosided']:.2e}")
        print(f"  Wilcoxon (one-sided, less): p = {pp['wilcoxon_p_onesided_less']:.2e}")
        print(f"  Cohen's d_z: {pp['cohens_d_z']:.3f}")
        print(f"  Fraction declining: {pp['frac_decline']:.1%}")
        print(f"  Test-retest r: {r['test_retest']['r']:.3f}")

        da = r["delta_vs_age"]
        print(f"  Delta ~ age: rho={da['spearman_rho']:.3f}, p={da['spearman_p']:.3f}")

    # Save
    out_file = OUT_PATH / "longitudinal_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Print hero result
    beta = results.get("beta", {})
    if not beta.get("skipped"):
        pp = beta["prop_positive"]
        print(f"\n{'='*60}")
        print(f"  HERO RESULT: Beta prop_positive longitudinal change")
        print(f"  N = {beta['n_valid']}, 5-year follow-up")
        print(f"  Mean change: {pp['mean_delta']:.4f} (95% CI: [{pp['ci_95'][0]:.4f}, {pp['ci_95'][1]:.4f}])")
        print(f"  Wilcoxon p (one-sided): {pp['wilcoxon_p_onesided_less']:.2e}")
        print(f"  Cohen's d_z: {pp['cohens_d_z']:.3f}")
        print(f"  {pp['frac_decline']:.1%} of subjects showed decline")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
