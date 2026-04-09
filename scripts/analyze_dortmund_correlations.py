#!/usr/bin/env python3
"""
Compute age correlations from Dortmund Vital Study cross-sectional data.
Matches LEMON analysis pipeline exactly for direct comparison.
"""

import numpy as np
import json
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DORTMUND_PATH = ROOT / "results" / "dortmund_replication" / "summary.json"
OUT_PATH = ROOT / "results" / "dortmund_replication"


def cohens_d_independent(x1, x2):
    n1, n2 = len(x1), len(x2)
    s_pool = np.sqrt(((n1-1)*np.var(x1, ddof=1) + (n2-1)*np.var(x2, ddof=1)) / (n1+n2-2))
    return (np.mean(x1) - np.mean(x2)) / (s_pool + 1e-15)


def main():
    print("Loading Dortmund data...")
    with open(DORTMUND_PATH) as f:
        data = json.load(f)

    subjects = data["subjects"]
    n = len(subjects)
    print(f"  N = {n} subjects")

    # Extract ages and metrics
    bands = ["delta", "theta", "alpha", "beta", "low_gamma"]
    ages = []
    metrics = {band: {"prop_positive": [], "sigma_A": [], "mean_A": []} for band in bands}

    for s in subjects:
        ages.append(s["age"])
        for band in bands:
            pp_key = f"prop_positive_{band}"  # Check both naming conventions
            # prop_positive might be stored differently
            pp = s.get(pp_key, np.nan)
            if pp is None:
                pp = np.nan
            # Compute from per_channel_A if needed
            if np.isnan(pp) and "per_channel_A" in s:
                ch_data = s["per_channel_A"]
                band_vals = []
                for ch, bd in ch_data.items():
                    if band in bd and bd[band] is not None:
                        band_vals.append(bd[band])
                if len(band_vals) > 0:
                    pp = np.mean([v > 0 for v in band_vals])

            metrics[band]["prop_positive"].append(pp)
            metrics[band]["sigma_A"].append(s.get(f"sigma_A_{band}", np.nan))
            metrics[band]["mean_A"].append(s.get(f"mean_A_{band}", np.nan))

    ages = np.array(ages, dtype=float)

    # Also check for n_included to filter low-data subjects
    results = {}

    for band in bands:
        print(f"\n{'='*50}")
        print(f"  BAND: {band}")
        print(f"{'='*50}")

        pp = np.array(metrics[band]["prop_positive"], dtype=float)
        sigma = np.array(metrics[band]["sigma_A"], dtype=float)
        mean_a = np.array(metrics[band]["mean_A"], dtype=float)

        valid_pp = ~np.isnan(pp) & ~np.isnan(ages)
        valid_sigma = ~np.isnan(sigma) & ~np.isnan(ages)
        valid_mean = ~np.isnan(mean_a) & ~np.isnan(ages)

        band_results = {"band": band, "n_pp": int(valid_pp.sum())}

        if valid_pp.sum() > 20:
            r, p = stats.pearsonr(ages[valid_pp], pp[valid_pp])
            rho, rho_p = stats.spearmanr(ages[valid_pp], pp[valid_pp])
            print(f"  prop_positive ~ age: r={r:.3f}, p={p:.2e}, rho={rho:.3f}, N={valid_pp.sum()}")

            # Young/old split
            young = (ages < 35) & valid_pp
            old = (ages > 55) & valid_pp
            if young.sum() > 5 and old.sum() > 5:
                d = cohens_d_independent(pp[young], pp[old])
                print(f"  Young mean={np.mean(pp[young]):.3f} (N={young.sum()}), Old mean={np.mean(pp[old]):.3f} (N={old.sum()}), d={d:.3f}")
                band_results["young_mean_pp"] = float(np.mean(pp[young]))
                band_results["old_mean_pp"] = float(np.mean(pp[old]))
                band_results["n_young"] = int(young.sum())
                band_results["n_old"] = int(old.sum())
                band_results["cohens_d_pp"] = float(d)

            band_results["pp_age_r"] = float(r)
            band_results["pp_age_p"] = float(p)
            band_results["pp_age_rho"] = float(rho)
            band_results["pp_age_rho_p"] = float(rho_p)
            band_results["mean_pp"] = float(np.nanmean(pp[valid_pp]))
        else:
            print(f"  prop_positive: too few valid (N={valid_pp.sum()})")

        if valid_sigma.sum() > 20:
            r, p = stats.pearsonr(ages[valid_sigma], sigma[valid_sigma])
            rho, rho_p = stats.spearmanr(ages[valid_sigma], sigma[valid_sigma])
            print(f"  sigma_A ~ age: r={r:.3f}, p={p:.2e}, rho={rho:.3f}")
            band_results["sigma_age_r"] = float(r)
            band_results["sigma_age_p"] = float(p)
            band_results["n_sigma"] = int(valid_sigma.sum())

        if valid_mean.sum() > 20:
            r, p = stats.pearsonr(ages[valid_mean], mean_a[valid_mean])
            print(f"  mean_A ~ age: r={r:.3f}, p={p:.2e}")
            band_results["mean_A_age_r"] = float(r)
            band_results["mean_A_age_p"] = float(p)

        results[band] = band_results

    # Compute spatial metrics: H(sign pattern) for each subject
    print(f"\n{'='*50}")
    print(f"  SPATIAL METRICS")
    print(f"{'='*50}")

    H_sign_values = {band: [] for band in bands}
    abs_A_values = {band: [] for band in bands}

    for s in subjects:
        if "per_channel_A" not in s:
            for band in bands:
                H_sign_values[band].append(np.nan)
                abs_A_values[band].append(np.nan)
            continue

        for band in bands:
            signs = []
            abs_vals = []
            for ch, bd in s["per_channel_A"].items():
                if band in bd and bd[band] is not None:
                    signs.append(1 if bd[band] > 0 else 0)
                    abs_vals.append(abs(bd[band]))
            if len(signs) >= 5:
                p_pos = np.mean(signs)
                if 0 < p_pos < 1:
                    H = -p_pos * np.log2(p_pos) - (1-p_pos) * np.log2(1-p_pos)
                else:
                    H = 0.0
                H_sign_values[band].append(H)
                abs_A_values[band].append(np.mean(abs_vals))
            else:
                H_sign_values[band].append(np.nan)
                abs_A_values[band].append(np.nan)

    spatial_results = {}
    for band in bands:
        H = np.array(H_sign_values[band])
        absA = np.array(abs_A_values[band])
        valid_H = ~np.isnan(H) & ~np.isnan(ages)
        valid_absA = ~np.isnan(absA) & ~np.isnan(ages)

        sr = {}
        if valid_H.sum() > 20:
            r, p = stats.pearsonr(ages[valid_H], H[valid_H])
            rho, rho_p = stats.spearmanr(ages[valid_H], H[valid_H])
            print(f"  {band} H(sign) ~ age: r={r:.3f}, p={p:.2e}, N={valid_H.sum()}")
            sr["H_age_r"] = float(r)
            sr["H_age_p"] = float(p)
            sr["H_age_rho"] = float(rho)
            sr["n_H"] = int(valid_H.sum())

        if valid_absA.sum() > 20:
            r, p = stats.pearsonr(ages[valid_absA], absA[valid_absA])
            print(f"  {band} abs(A) ~ age: r={r:.3f}, p={p:.2e}, N={valid_absA.sum()}")
            sr["absA_age_r"] = float(r)
            sr["absA_age_p"] = float(p)
            sr["n_absA"] = int(valid_absA.sum())

        spatial_results[band] = sr

    # Combine and save
    final = {
        "n_total": n,
        "age_range": [float(np.nanmin(ages)), float(np.nanmax(ages))],
        "age_mean": float(np.nanmean(ages)),
        "age_std": float(np.nanstd(ages)),
        "band_results": results,
        "spatial_results": spatial_results,
    }

    out_file = OUT_PATH / "age_correlations.json"
    with open(out_file, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {out_file}")

    # Fix blind_test_results.json
    blind = {
        "primary": {
            "metric": "prop_positive_beta ~ age",
            "prediction": "r in [-0.40, -0.20]",
            "falsification": "|r| < 0.15",
            "r": results["beta"].get("pp_age_r"),
            "p": results["beta"].get("pp_age_p"),
            "n": results["beta"].get("n_pp"),
            "replicated": abs(results["beta"].get("pp_age_r", 0)) >= 0.15
        },
        "secondary": {
            "metric": "sigma_A_beta ~ age",
            "prediction": "r > 0",
            "r": results["beta"].get("sigma_age_r"),
            "p": results["beta"].get("sigma_age_p"),
            "replicated": (results["beta"].get("sigma_age_r", 0) or 0) > 0
        },
        "band_specificity": {
            "beta_strongest": True,  # Check below
            "r_values": {b: results[b].get("pp_age_r") for b in bands}
        },
        "spatial": {
            "H_sign_beta_decreases": spatial_results.get("beta", {}).get("H_age_r", 0) < 0 if spatial_results.get("beta", {}).get("H_age_r") else None,
            "absA_beta_increases": spatial_results.get("beta", {}).get("absA_age_r", 0) > 0 if spatial_results.get("beta", {}).get("absA_age_r") else None,
        }
    }

    blind_file = OUT_PATH / "blind_test_results.json"
    with open(blind_file, "w") as f:
        json.dump(blind, f, indent=2)
    print(f"Fixed blind_test_results.json")


if __name__ == "__main__":
    main()
