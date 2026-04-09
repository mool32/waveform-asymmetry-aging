#!/usr/bin/env python3
"""
verify_numbers.py
Cross-check every statistic cited in the paper against actual data files.
Outputs a verification table to stdout and saves to results/verification_report.txt.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent.parent
LEMON_SUMMARY = BASE / "results" / "phase1_full" / "summary.json"
DORTMUND_CORR = BASE / "results" / "dortmund_replication" / "age_correlations.json"
LONGITUDINAL  = BASE / "results" / "longitudinal_stats" / "longitudinal_results.json"
ECEO_CSV      = BASE / "results" / "eceo_entropy" / "eceo_entropy_results.csv"
ECEO_SUMMARY  = BASE / "results" / "eceo_entropy" / "summary.json"
DEMOGRAPHICS  = BASE / "data" / "lemon" / "Participants_MPILMBB_LEMON.csv"
OUTPUT        = BASE / "results" / "verification_report.txt"

BANDS = ["delta", "theta", "alpha", "beta", "low_gamma"]

# Regional channel groups
REGIONS = {
    "frontal":  ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "AF3", "AF4"],
    "central":  ["C3", "C4", "Cz", "FC1", "FC2", "CP1", "CP2"],
    "temporal": ["T7", "T8", "TP9", "TP10"],
    "parietal": ["P3", "P4", "P7", "P8", "Pz"],
    "occipital":["O1", "O2", "Oz", "PO3", "PO4"],
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_json(p):
    with open(p) as f:
        return json.load(f)


def match(expected, computed, tol=0.05):
    """Return True if values match within tol (relative or absolute)."""
    if computed is None:
        return "SKIP"
    if expected is None:
        return "INFO"  # computed-only, no expected to check against
    if isinstance(expected, str) or isinstance(computed, str):
        return "OK" if str(expected) == str(computed) else "MISMATCH"
    if expected == 0:
        return "OK" if abs(computed) < tol else "MISMATCH"
    if abs(expected - computed) / max(abs(expected), 1e-15) <= tol:
        return "OK"
    return "MISMATCH"


def fmt(v, decimals=6):
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    return f"{v:.{decimals}g}"


rows = []  # (statistic, expected, computed, match_flag)


def add(stat, expected, computed, tol=0.05):
    flag = match(expected, computed, tol)
    rows.append((stat, fmt(expected), fmt(computed), flag))


# --------------------------------------------------------------------------- #
# 1. Load LEMON summary
# --------------------------------------------------------------------------- #
print("Loading LEMON summary ...")
lemon = load_json(LEMON_SUMMARY)
subjects = lemon["subjects"]
N_total = len(subjects)
add("LEMON N total subjects", 215, N_total, tol=0)

# Build arrays per band
ages = np.array([s["age"] for s in subjects])

pp = {}
sigma_A = {}
mean_A = {}
for band in BANDS:
    pp[band] = np.array([s[f"prop_positive_{band}"] for s in subjects], dtype=float)
    sigma_A[band] = np.array([s[f"sigma_A_{band}"] for s in subjects], dtype=float)
    mean_A[band] = np.array([s[f"mean_A_{band}"] for s in subjects], dtype=float)

# --------------------------------------------------------------------------- #
# 2. Per-band statistics
# --------------------------------------------------------------------------- #
print("Computing per-band statistics ...")

for band in BANDS:
    valid = ~np.isnan(pp[band])
    n_valid = int(valid.sum())
    add(f"{band} N valid", None, n_valid)

    pp_v = pp[band][valid]
    ages_v = ages[valid]

    # Mean prop_positive and one-sample t-test vs 0.5
    mean_pp = float(np.mean(pp_v))
    t_stat, t_p = stats.ttest_1samp(pp_v, 0.5)
    add(f"{band} mean_prop_positive", None, mean_pp)
    add(f"{band} t-test vs 0.5 (t)", None, float(t_stat))
    add(f"{band} t-test vs 0.5 (p)", None, float(t_p))

    # Pearson r, p for prop_positive ~ age
    r_val, r_p = stats.pearsonr(ages_v, pp_v)
    add(f"{band} Pearson r(pp~age)", None, float(r_val))
    add(f"{band} Pearson p(pp~age)", None, float(r_p))

    # Spearman rho
    rho, rho_p = stats.spearmanr(ages_v, pp_v)
    add(f"{band} Spearman rho(pp~age)", None, float(rho))

    # Cohen's d: young <35 vs old >55
    young_mask = ages_v < 35
    old_mask = ages_v > 55
    pp_young = pp_v[young_mask]
    pp_old = pp_v[old_mask]
    if len(pp_young) > 1 and len(pp_old) > 1:
        pooled_std = np.sqrt(
            ((len(pp_young) - 1) * np.var(pp_young, ddof=1)
             + (len(pp_old) - 1) * np.var(pp_old, ddof=1))
            / (len(pp_young) + len(pp_old) - 2)
        )
        d = (np.mean(pp_young) - np.mean(pp_old)) / pooled_std if pooled_std > 0 else 0
        add(f"{band} young(<35) mean pp", None, float(np.mean(pp_young)))
        add(f"{band} old(>55) mean pp", None, float(np.mean(pp_old)))
        add(f"{band} Cohen's d (young-old)", None, float(d))

# --------------------------------------------------------------------------- #
# 3. Key expected numbers for beta
# --------------------------------------------------------------------------- #
print("Verifying key beta numbers ...")

valid_beta = ~np.isnan(pp["beta"])
pp_beta_v = pp["beta"][valid_beta]
ages_beta_v = ages[valid_beta]

r_beta, p_beta = stats.pearsonr(ages_beta_v, pp_beta_v)
add("EXPECTED beta r(pp~age)", -0.326, float(r_beta))
add("EXPECTED beta p(pp~age)", 1e-6, float(p_beta))

young_beta = pp_beta_v[ages_beta_v < 35]
old_beta = pp_beta_v[ages_beta_v > 55]
pooled = np.sqrt(
    ((len(young_beta) - 1) * np.var(young_beta, ddof=1)
     + (len(old_beta) - 1) * np.var(old_beta, ddof=1))
    / (len(young_beta) + len(old_beta) - 2)
)
d_beta = (np.mean(young_beta) - np.mean(old_beta)) / pooled if pooled > 0 else 0
add("EXPECTED beta Cohen's d", 0.71, float(d_beta))
add("EXPECTED beta young mean", 0.332, float(np.mean(young_beta)))
add("EXPECTED beta old mean", 0.222, float(np.mean(old_beta)))

# --------------------------------------------------------------------------- #
# 4. Spatial / regional analysis (beta band)
# --------------------------------------------------------------------------- #
print("Computing regional beta statistics ...")

# For each subject, compute regional prop_positive from per_channel_A
regional_pp = {region: [] for region in REGIONS}
regional_ages = []

for s in subjects:
    beta_ch = s["per_channel_A"].get("beta", {})
    if not beta_ch:
        continue
    regional_ages.append(s["age"])
    for region, channels in REGIONS.items():
        vals = [beta_ch[ch] for ch in channels if ch in beta_ch]
        if vals:
            pp_reg = np.mean([1 if v > 0 else 0 for v in vals])
        else:
            pp_reg = np.nan
        regional_pp[region].append(pp_reg)

regional_ages = np.array(regional_ages)

for region in REGIONS:
    arr = np.array(regional_pp[region])
    valid = ~np.isnan(arr)
    if valid.sum() > 3:
        r_reg, p_reg = stats.pearsonr(regional_ages[valid], arr[valid])
        add(f"beta regional r(pp~age) [{region}]", None, float(r_reg))
        add(f"beta regional p(pp~age) [{region}]", None, float(p_reg))

# --------------------------------------------------------------------------- #
# 5. Coarsening metrics from per_channel_A
# --------------------------------------------------------------------------- #
print("Computing coarsening metrics ...")

for band in ["theta", "beta"]:
    abs_A_list = []
    H_list = []
    sigma_list = []
    coarse_ages = []

    for s in subjects:
        ch_data = s["per_channel_A"].get(band, {})
        if len(ch_data) < 5:
            continue
        vals = np.array(list(ch_data.values()))
        coarse_ages.append(s["age"])

        # abs(A) mean across channels
        abs_A_list.append(float(np.mean(np.abs(vals))))

        # H(sign pattern) = entropy of the sign pattern
        signs = (vals > 0).astype(int)
        p_pos = np.mean(signs)
        p_neg = 1 - p_pos
        H = 0
        if 0 < p_pos < 1:
            H = -(p_pos * np.log2(p_pos) + p_neg * np.log2(p_neg))
        H_list.append(H)

        # sigma(A) across channels
        sigma_list.append(float(np.std(vals, ddof=1)))

    coarse_ages = np.array(coarse_ages)
    abs_A_arr = np.array(abs_A_list)
    H_arr = np.array(H_list)
    sigma_arr = np.array(sigma_list)

    r_abs, p_abs = stats.pearsonr(coarse_ages, abs_A_arr)
    add(f"{band} abs(A)~age r", None, float(r_abs))
    add(f"{band} abs(A)~age p", None, float(p_abs))

    r_H, p_H = stats.pearsonr(coarse_ages, H_arr)
    add(f"{band} H(sign)~age r", None, float(r_H))
    add(f"{band} H(sign)~age p", None, float(p_H))

    r_sig, p_sig = stats.pearsonr(coarse_ages, sigma_arr)
    add(f"{band} sigma(A)~age r", None, float(r_sig))
    add(f"{band} sigma(A)~age p", None, float(p_sig))

# --------------------------------------------------------------------------- #
# 6. Aperiodic exponent check
# --------------------------------------------------------------------------- #
print("Checking for aperiodic exponent data ...")
aperiodic_fields = [k for k in subjects[0].keys() if "aperiodic" in k.lower() or "exponent" in k.lower()]
if aperiodic_fields:
    add("Aperiodic exponent field(s) present", None, f"YES: {aperiodic_fields}")
else:
    add("Aperiodic exponent field(s) present", None, "NOT IN summary.json (needs separate pipeline)")

# --------------------------------------------------------------------------- #
# 7. Cross-system (HRV) check
# --------------------------------------------------------------------------- #
print("Checking for HRV data ...")
n_hrv = lemon.get("n_with_hrv", 0)
hrv_fields = [k for k in subjects[0].keys() if "hrv" in k.lower() or "heart" in k.lower() or "rr" in k.lower()]
add("N with HRV (header)", 72, n_hrv)
if hrv_fields:
    add("HRV per-subject fields present", "yes", "yes")
else:
    add("HRV per-subject fields present", "yes", f"NOT FOUND (only header n_with_hrv={n_hrv})")

# --------------------------------------------------------------------------- #
# 8. Dortmund replication cross-check
# --------------------------------------------------------------------------- #
print("Verifying Dortmund replication ...")
dort = load_json(DORTMUND_CORR)

dort_beta = dort["band_results"]["beta"]
add("Dortmund beta r(pp~age)", None, dort_beta["pp_age_r"])
add("Dortmund beta p(pp~age)", None, dort_beta["pp_age_p"])
add("Dortmund beta Cohen's d", None, dort_beta["cohens_d_pp"])
add("Dortmund N total", None, dort["n_total"])

# --------------------------------------------------------------------------- #
# 9. Longitudinal cross-check
# --------------------------------------------------------------------------- #
print("Verifying longitudinal results ...")
longit = load_json(LONGITUDINAL)

for band in ["theta", "alpha", "beta", "low_gamma"]:
    if band in longit and not longit[band].get("skipped", False):
        lb = longit[band]
        add(f"Longitudinal {band} N valid", None, lb["n_valid"])
        add(f"Longitudinal {band} test-retest r", None, lb["test_retest"]["r"])
        add(f"Longitudinal {band} test-retest p", None, lb["test_retest"]["p"])
        pp_data = lb["prop_positive"]
        add(f"Longitudinal {band} pp mean_delta", None, pp_data["mean_delta"])
        add(f"Longitudinal {band} pp Wilcoxon p", None, pp_data["wilcoxon_p_twosided"])

# --------------------------------------------------------------------------- #
# 10. EC/EO entropy
# --------------------------------------------------------------------------- #
print("Verifying EC/EO entropy ...")
eceo_df = pd.read_csv(ECEO_CSV)
eceo_sum = load_json(ECEO_SUMMARY)

# Recompute paired t-test from raw CSV
h_ec = eceo_df["H_ec"].dropna()
h_eo = eceo_df["H_eo"].dropna()
# Align on common indices
common = h_ec.index.intersection(h_eo.index)
h_ec_c = h_ec.loc[common].values
h_eo_c = h_eo.loc[common].values
n_eceo = len(h_ec_c)

# Summary used H_eo - H_ec convention (delta_H = H_eo - H_ec gives negative t)
# so we match: ttest_rel(h_eo, h_ec) to get the same sign
t_eceo, p_eceo_twosided = stats.ttest_rel(h_eo_c, h_ec_c)
# One-sided: testing H_eo < H_ec  (t negative), p_onesided = p_two/2 when t < 0
p_eceo_onesided = p_eceo_twosided / 2 if t_eceo < 0 else 1 - p_eceo_twosided / 2

diff = h_eo_c - h_ec_c  # delta_H convention from summary

# Cohen's d_z (paired: mean_diff / sd_diff)
d_eceo_z = float(np.mean(diff) / np.std(diff, ddof=1))

# Cohen's d_av (pooled SD of the two conditions) -- used in original code
pooled_sd = np.sqrt((np.var(h_ec_c, ddof=1) + np.var(h_eo_c, ddof=1)) / 2)
d_eceo_av = float(np.mean(diff) / pooled_sd)

add("EC/EO N valid", eceo_sum["n_valid"], n_eceo)
add("EC/EO t-statistic (recomputed)", eceo_sum["test1_t"], float(t_eceo))
add("EC/EO p one-sided (recomputed)", eceo_sum["test1_p_onesided"], float(p_eceo_onesided))
add("EC/EO Cohen's d_z (paired)", None, d_eceo_z)
add("EC/EO Cohen's d_av (pooled SD)", None, d_eceo_av)
add("EXPECTED EC/EO Cohen's d", -0.56, d_eceo_av)
add("EXPECTED EC/EO p one-sided", 1.9e-9, float(p_eceo_onesided))

# Verify the stored summary value (uses d_av convention, within ~0.3% tolerance)
add("EC/EO summary d vs recomputed d_av", eceo_sum["test1_cohen_d"], d_eceo_av)

# --------------------------------------------------------------------------- #
# 11. Demographics sanity check
# --------------------------------------------------------------------------- #
print("Loading demographics ...")
demo = pd.read_csv(DEMOGRAPHICS)
add("Demographics N rows", None, len(demo))

# --------------------------------------------------------------------------- #
# Print and save report
# --------------------------------------------------------------------------- #
header = f"{'Statistic':<50} {'Expected':<18} {'Computed':<18} {'Match?':<10}"
sep = "-" * len(header)

lines = []
lines.append("=" * len(header))
lines.append("VERIFICATION REPORT")
lines.append(f"Generated by verify_numbers.py")
lines.append("=" * len(header))
lines.append("")
lines.append(header)
lines.append(sep)

n_ok = 0
n_mismatch = 0
n_skip = 0
n_info = 0

for stat, exp, comp, flag in rows:
    line = f"{stat:<50} {exp:<18} {comp:<18} {flag:<10}"
    lines.append(line)
    if flag == "OK":
        n_ok += 1
    elif flag == "MISMATCH":
        n_mismatch += 1
    elif flag == "SKIP":
        n_skip += 1
    elif flag == "INFO":
        n_info += 1

lines.append(sep)
lines.append(f"TOTALS: {n_ok} OK, {n_mismatch} MISMATCH, {n_info} INFO (computed only), {n_skip} SKIP")
lines.append("")

# Print mismatches prominently
if n_mismatch > 0:
    lines.append("*** MISMATCHES ***")
    for stat, exp, comp, flag in rows:
        if flag == "MISMATCH":
            lines.append(f"  {stat}: expected={exp}, computed={comp}")
    lines.append("")

report = "\n".join(lines)
print(report)

# Save
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    f.write(report)

print(f"\nReport saved to {OUTPUT}")

# Exit with error code if mismatches found
if n_mismatch > 0:
    print(f"\nWARNING: {n_mismatch} mismatches found!")
    sys.exit(1)
else:
    print("\nAll checked statistics OK.")
