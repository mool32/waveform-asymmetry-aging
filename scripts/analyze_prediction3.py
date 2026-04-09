#!/usr/bin/env python3
"""
Prediction 3 Analysis: Cross-system coherence between EEG and HRV asymmetry.

Questions:
a. Does correlation between EEG-A and HRV-A exist (rho_cross)?
b. Does rho_cross degrade with age?
c. Does the EEG-HRV relationship differ young (<35) vs old (>55)?
d. Age trends in HRV metrics
e. Full cross-system coherence analysis
"""

import json
import pickle
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Load data ───
base = Path("/Users/teo/Desktop/research/Asymmetric Perception Cycles")
with open(base / "results/phase1_full/summary.json") as f:
    summary = json.load(f)

with open(base / "results/phase1_full/all_results.pkl", "rb") as f:
    all_results = pickle.load(f)

print("=" * 70)
print("PREDICTION 3: CROSS-SYSTEM COHERENCE ANALYSIS")
print("=" * 70)
print(f"Total subjects: {summary['n_subjects']}")
print(f"Subjects with HRV: {summary['n_with_hrv']}")

# ─── Extract HRV subjects from summary.json ───
hrv_subjects = [s for s in summary['subjects'] if s.get('hrv') is not None]
print(f"HRV subjects found in summary: {len(hrv_subjects)}")

# Also check pkl for richer data
hrv_pkl = [r for r in all_results if r.get('hrv') is not None]
print(f"HRV subjects found in pkl: {len(hrv_pkl)}")

# ─── Build arrays ───
ages = []
sexes = []
rho_pearson = []
rho_spearman = []
beta_pp = []
beta_tm = []
hrv_lf_A = []
hrv_hf_A = []
hrv_vlf_A = []
hrv_vhf_A = []
dc_ac_ratios = []
mean_hrs = []
subject_ids = []

for s in hrv_subjects:
    hrv = s['hrv']
    rho = s.get('rho_cross')
    if rho is None:
        continue

    ages.append(s['age'])
    sexes.append(s['sex'])
    subject_ids.append(s['id'])
    rho_pearson.append(rho['pearson'])
    rho_spearman.append(rho['spearman'])

    # EEG beta metrics
    beta_pp.append(s.get('prop_positive_beta'))
    beta_tm.append(s.get('trimmed_mean_A_beta'))

    # HRV band asymmetries
    prof = hrv['profile']
    hrv_lf_A.append(prof.get('LF', {}).get('A'))
    hrv_hf_A.append(prof.get('HF', {}).get('A'))
    hrv_vlf_A.append(prof.get('VLF', {}).get('A'))
    hrv_vhf_A.append(prof.get('VHF', {}).get('A'))

    dc_ac_ratios.append(hrv.get('dc_ac_ratio'))
    mean_hrs.append(hrv.get('mean_hr'))

ages = np.array(ages, dtype=float)
rho_pearson = np.array(rho_pearson, dtype=float)
rho_spearman = np.array(rho_spearman, dtype=float)
beta_pp = np.array(beta_pp, dtype=float)
beta_tm = np.array(beta_tm, dtype=float)
hrv_lf_A = np.array(hrv_lf_A, dtype=float)
hrv_hf_A = np.array(hrv_hf_A, dtype=float)
hrv_vlf_A = np.array(hrv_vlf_A, dtype=float)
hrv_vhf_A = np.array(hrv_vhf_A, dtype=float)
dc_ac_ratios = np.array(dc_ac_ratios, dtype=float)
mean_hrs = np.array(mean_hrs, dtype=float)

N = len(ages)
print(f"\nUsable subjects (HRV + rho_cross): {N}")
print(f"Age range: {np.nanmin(ages):.1f} - {np.nanmax(ages):.1f}")
print(f"Age mean ± SD: {np.nanmean(ages):.1f} ± {np.nanstd(ages):.1f}")
print(f"Sex: {sum(1 for s in sexes if s == 'M')} M, {sum(1 for s in sexes if s == 'F')} F")

# ═══════════════════════════════════════════════════════════════════
# A. Does ρ_cross exist? (Is EEG-HRV asymmetry correlated?)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("A. DOES rho_cross EXIST? (EEG-HRV asymmetry correlation)")
print("=" * 70)

valid_rho = ~np.isnan(rho_pearson)
rho_p_valid = rho_pearson[valid_rho]
rho_s_valid = rho_spearman[valid_rho]

print(f"\nρ_cross (Pearson) across {valid_rho.sum()} subjects:")
print(f"  Mean:   {np.mean(rho_p_valid):+.4f}")
print(f"  Median: {np.median(rho_p_valid):+.4f}")
print(f"  SD:     {np.std(rho_p_valid):.4f}")
print(f"  Range:  [{np.min(rho_p_valid):.4f}, {np.max(rho_p_valid):.4f}]")

# One-sample t-test: is mean rho_cross != 0?
t, p = stats.ttest_1samp(rho_p_valid, 0)
d = np.mean(rho_p_valid) / (np.std(rho_p_valid, ddof=1) + 1e-10)
print(f"\n  One-sample t-test (H0: ρ_cross = 0):")
print(f"    t({len(rho_p_valid)-1}) = {t:.3f}, p = {p:.4e}, Cohen's d = {d:.3f}")
print(f"    {'SIGNIFICANT' if p < 0.05 else 'NOT significant'} at α=0.05")

# Also Wilcoxon signed-rank
w, pw = stats.wilcoxon(rho_p_valid)
print(f"\n  Wilcoxon signed-rank (non-parametric):")
print(f"    W = {w:.1f}, p = {pw:.4e}")

# Distribution: how many positive vs negative?
n_pos = np.sum(rho_p_valid > 0)
n_neg = np.sum(rho_p_valid < 0)
_, p_binom = stats.binom_test(n_pos, len(rho_p_valid), 0.5) if hasattr(stats, 'binom_test') else (None, stats.binomtest(n_pos, len(rho_p_valid), 0.5).pvalue)
print(f"\n  Sign test: {n_pos} positive, {n_neg} negative (p = {p_binom:.4f})")

print(f"\nρ_cross (Spearman):")
print(f"  Mean:   {np.mean(rho_s_valid):+.4f}")
print(f"  Median: {np.median(rho_s_valid):+.4f}")

# ═══════════════════════════════════════════════════════════════════
# B. Does ρ_cross degrade with age?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("B. DOES rho_cross DEGRADE WITH AGE?")
print("=" * 70)

valid = valid_rho & ~np.isnan(ages)
a = ages[valid]
rp = rho_pearson[valid]

# Pearson correlation: rho_cross ~ age
r_val, p_val = stats.pearsonr(a, rp)
print(f"\n  Pearson: r({len(a)-2}) = {r_val:+.4f}, p = {p_val:.4e}")

# Spearman
rs_val, ps_val = stats.spearmanr(a, rp)
print(f"  Spearman: ρ = {rs_val:+.4f}, p = {ps_val:.4e}")

# Linear regression with CI
from numpy.polynomial import polynomial as P
slope, intercept = np.polyfit(a, rp, 1)
# Use scipy linregress for SE
lr = stats.linregress(a, rp)
print(f"\n  Linear regression: ρ_cross = {lr.intercept:+.4f} + ({lr.slope:+.6f}) * age")
print(f"    R² = {lr.rvalue**2:.4f}")
print(f"    slope SE = {lr.stderr:.6f}")
print(f"    slope 95% CI: [{lr.slope - 1.96*lr.stderr:.6f}, {lr.slope + 1.96*lr.stderr:.6f}]")
print(f"    p = {lr.pvalue:.4e}")

# Effect size: predicted rho_cross at age 25 vs 65
rho_at_25 = lr.intercept + lr.slope * 25
rho_at_65 = lr.intercept + lr.slope * 65
print(f"\n  Predicted ρ_cross at age 25: {rho_at_25:+.4f}")
print(f"  Predicted ρ_cross at age 65: {rho_at_65:+.4f}")
print(f"  Change: {rho_at_65 - rho_at_25:+.4f}")

# ═══════════════════════════════════════════════════════════════════
# C. Young (<35) vs Old (>55) comparison
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("C. YOUNG (<35) vs OLD (>55) COMPARISON")
print("=" * 70)

young_mask = ages < 35
old_mask = ages > 55
mid_mask = ~young_mask & ~old_mask

print(f"\n  Young (<35): N = {young_mask.sum()}")
print(f"  Middle (35-55): N = {mid_mask.sum()}")
print(f"  Old (>55): N = {old_mask.sum()}")

for name, arr, label in [
    ("ρ_cross (Pearson)", rho_pearson, "rho_cross"),
    ("ρ_cross (Spearman)", rho_spearman, "rho_cross_s"),
    ("β prop_positive", beta_pp, "beta_pp"),
    ("β trimmed_mean_A", beta_tm, "beta_tm"),
    ("HRV LF A", hrv_lf_A, "hrv_lf"),
    ("HRV HF A", hrv_hf_A, "hrv_hf"),
    ("DC/AC ratio", dc_ac_ratios, "dc_ac"),
    ("Mean HR", mean_hrs, "hr"),
]:
    y_vals = arr[young_mask & ~np.isnan(arr)]
    o_vals = arr[old_mask & ~np.isnan(arr)]

    if len(y_vals) < 3 or len(o_vals) < 3:
        print(f"\n  {name}: insufficient data (young={len(y_vals)}, old={len(o_vals)})")
        continue

    # Mann-Whitney U (non-parametric)
    u_stat, u_p = stats.mannwhitneyu(y_vals, o_vals, alternative='two-sided')

    # Welch's t-test
    t_stat, t_p = stats.ttest_ind(y_vals, o_vals, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt((np.var(y_vals, ddof=1) * (len(y_vals)-1) +
                          np.var(o_vals, ddof=1) * (len(o_vals)-1)) /
                         (len(y_vals) + len(o_vals) - 2))
    cohens_d = (np.mean(y_vals) - np.mean(o_vals)) / (pooled_std + 1e-10)

    print(f"\n  {name}:")
    print(f"    Young: {np.mean(y_vals):+.4f} ± {np.std(y_vals, ddof=1):.4f} (N={len(y_vals)})")
    print(f"    Old:   {np.mean(o_vals):+.4f} ± {np.std(o_vals, ddof=1):.4f} (N={len(o_vals)})")
    print(f"    Welch t = {t_stat:.3f}, p = {t_p:.4e}")
    print(f"    Mann-Whitney U = {u_stat:.1f}, p = {u_p:.4e}")
    print(f"    Cohen's d = {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'LARGE'})")

# ═══════════════════════════════════════════════════════════════════
# D. Age-related trends in HRV metrics
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D. AGE-RELATED TRENDS IN HRV METRICS")
print("=" * 70)

for name, arr in [
    ("HRV VLF A", hrv_vlf_A),
    ("HRV LF A", hrv_lf_A),
    ("HRV HF A", hrv_hf_A),
    ("HRV VHF A", hrv_vhf_A),
    ("DC/AC ratio", dc_ac_ratios),
    ("Mean HR", mean_hrs),
    ("ρ_cross (Pearson)", rho_pearson),
]:
    valid = ~np.isnan(arr) & ~np.isnan(ages)
    if valid.sum() < 5:
        print(f"\n  {name}: insufficient data (N={valid.sum()})")
        continue

    lr = stats.linregress(ages[valid], arr[valid])
    r_s, p_s = stats.spearmanr(ages[valid], arr[valid])

    print(f"\n  {name} ~ age (N={valid.sum()}):")
    print(f"    Pearson r = {lr.rvalue:+.4f}, p = {lr.pvalue:.4e}")
    print(f"    Spearman ρ = {r_s:+.4f}, p = {p_s:.4e}")
    print(f"    Slope = {lr.slope:+.6f} per year")
    print(f"    R² = {lr.rvalue**2:.4f}")

# ═══════════════════════════════════════════════════════════════════
# E. Cross-system coherence: EEG-A vs HRV-A relationship by age
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("E. CROSS-SYSTEM COHERENCE: EEG beta vs HRV metrics")
print("=" * 70)

# Correlation between EEG beta_pp and HRV metrics
for name, hrv_arr in [
    ("HRV LF A", hrv_lf_A),
    ("HRV HF A", hrv_hf_A),
    ("DC/AC ratio", dc_ac_ratios),
]:
    valid = ~np.isnan(beta_pp) & ~np.isnan(hrv_arr)
    if valid.sum() < 5:
        continue

    r_val, p_val = stats.pearsonr(beta_pp[valid], hrv_arr[valid])
    rs_val, ps_val = stats.spearmanr(beta_pp[valid], hrv_arr[valid])
    print(f"\n  EEG β prop_positive vs {name} (N={valid.sum()}):")
    print(f"    Pearson r = {r_val:+.4f}, p = {p_val:.4e}")
    print(f"    Spearman ρ = {rs_val:+.4f}, p = {ps_val:.4e}")

# Does EEG-HRV correlation differ in young vs old?
print("\n" + "-" * 50)
print("  EEG-HRV correlation WITHIN age groups:")

for grp, mask in [("Young (<35)", young_mask), ("Old (>55)", old_mask), ("All", np.ones(N, bool))]:
    for name, hrv_arr in [("HRV LF A", hrv_lf_A), ("HRV HF A", hrv_hf_A), ("DC/AC", dc_ac_ratios)]:
        valid = mask & ~np.isnan(beta_pp) & ~np.isnan(hrv_arr)
        if valid.sum() < 4:
            continue
        r_val, p_val = stats.pearsonr(beta_pp[valid], hrv_arr[valid])
        print(f"    {grp:12s}: β_pp vs {name:12s}: r={r_val:+.4f} p={p_val:.4f} (N={valid.sum()})")

# ═══════════════════════════════════════════════════════════════════
# F. Interaction test: does age moderate EEG-HRV relationship?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("F. AGE MODERATION: Does age moderate the EEG-HRV link?")
print("=" * 70)

try:
    import statsmodels.api as sm

    for hrv_name, hrv_arr in [("HRV_LF_A", hrv_lf_A), ("HRV_HF_A", hrv_hf_A), ("DC_AC", dc_ac_ratios)]:
        valid = ~np.isnan(beta_pp) & ~np.isnan(hrv_arr) & ~np.isnan(ages)
        if valid.sum() < 15:
            continue

        y = hrv_arr[valid]
        x_eeg = beta_pp[valid]
        x_age = ages[valid]
        x_interaction = x_eeg * x_age

        # Center predictors
        x_eeg_c = x_eeg - np.mean(x_eeg)
        x_age_c = x_age - np.mean(x_age)
        x_inter_c = x_eeg_c * x_age_c

        X = np.column_stack([x_eeg_c, x_age_c, x_inter_c])
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        print(f"\n  {hrv_name} ~ β_pp + age + β_pp*age (N={valid.sum()}):")
        print(f"    R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}")
        print(f"    β_pp effect:       coef = {model.params[1]:+.6f}, p = {model.pvalues[1]:.4e}")
        print(f"    age effect:        coef = {model.params[2]:+.6f}, p = {model.pvalues[2]:.4e}")
        print(f"    INTERACTION:       coef = {model.params[3]:+.6f}, p = {model.pvalues[3]:.4e}")
        print(f"    {'*** SIGNIFICANT INTERACTION' if model.pvalues[3] < 0.05 else '    interaction not significant'}")

except ImportError:
    print("  statsmodels not available, skipping moderation analysis")

# ═══════════════════════════════════════════════════════════════════
# G. COMPREHENSIVE SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("G. COMPREHENSIVE SUMMARY")
print("=" * 70)

# Also get EEG asymmetry by band for HRV subjects from pkl for richer analysis
print("\nAll HRV band asymmetries:")
for band in ["VLF", "LF", "HF", "VHF"]:
    vals = []
    for s in hrv_subjects:
        a = s['hrv']['profile'].get(band, {}).get('A')
        if a is not None and not np.isnan(a):
            vals.append(a)
    if vals:
        vals = np.array(vals)
        t, p = stats.ttest_1samp(vals, 0)
        d = np.mean(vals) / (np.std(vals, ddof=1) + 1e-10)
        print(f"  {band:4s}: A = {np.mean(vals):+.4f} ± {np.std(vals):.4f}, "
              f"t = {t:.2f}, p = {p:.4e}, d = {d:.3f}, "
              f"N = {len(vals)}, {'*' if p < 0.05 else 'ns'}")

print("\nKey findings summary:")
print("-" * 50)

# Check if rho_cross is significantly different from zero
valid_rho = ~np.isnan(rho_pearson)
mean_rho = np.mean(rho_pearson[valid_rho])
t_rho, p_rho = stats.ttest_1samp(rho_pearson[valid_rho], 0)
print(f"  1. Mean ρ_cross = {mean_rho:+.4f} (p = {p_rho:.4e})")
print(f"     → {'Cross-system coherence EXISTS' if p_rho < 0.05 else 'No significant cross-system coherence'}")

# Check age degradation
valid = valid_rho & ~np.isnan(ages)
lr = stats.linregress(ages[valid], rho_pearson[valid])
print(f"  2. ρ_cross ~ age: r = {lr.rvalue:+.4f}, p = {lr.pvalue:.4e}")
print(f"     → {'ρ_cross DEGRADES with age' if lr.pvalue < 0.05 and lr.slope < 0 else 'No significant age degradation'}")

# Young vs old
y_rho = rho_pearson[young_mask & valid_rho]
o_rho = rho_pearson[old_mask & valid_rho]
if len(y_rho) >= 3 and len(o_rho) >= 3:
    t_yo, p_yo = stats.ttest_ind(y_rho, o_rho, equal_var=False)
    print(f"  3. Young ρ_cross = {np.mean(y_rho):+.4f}, Old = {np.mean(o_rho):+.4f}, p = {p_yo:.4e}")
    print(f"     → {'Significant' if p_yo < 0.05 else 'No significant'} young-old difference")
else:
    print(f"  3. Insufficient subjects for young/old comparison")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
