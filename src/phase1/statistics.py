"""
Phase 1: Statistical testing for predictions 1-4.

Prediction 1: A(τ) > 0 at all scales (one-sample t-test)
Prediction 2: A(τ) decreases with age (linear regression)
Prediction 3: ρ_cross decreases with age faster than individual A
Prediction 4: Transfer entropy shifts with age
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def test_prediction_1(subject_profiles, band_names):
    """Test A(τ) > 0 at each frequency band (one-sided t-test).

    Parameters
    ----------
    subject_profiles : list of dict
        Each element is {band_name: {'A': float, ...}} for one subject.
    band_names : list of str

    Returns
    -------
    results : pd.DataFrame
        Columns: band, mean_A, std_A, t_stat, p_value, p_corrected,
                 significant, n_subjects, effect_size_d
    """
    rows = []
    p_values = []

    for band in band_names:
        A_values = [sp[band]["A"] for sp in subject_profiles
                    if band in sp and not np.isnan(sp[band]["A"])]
        A_values = np.array(A_values)

        if len(A_values) < 5:
            rows.append({
                "band": band,
                "mean_A": np.nan, "std_A": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "n_subjects": len(A_values), "effect_size_d": np.nan,
            })
            p_values.append(1.0)
            continue

        t_stat, p_two = stats.ttest_1samp(A_values, 0)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2  # one-sided: A > 0
        d = np.mean(A_values) / (np.std(A_values, ddof=1) + 1e-10)  # Cohen's d

        rows.append({
            "band": band,
            "mean_A": np.mean(A_values),
            "std_A": np.std(A_values, ddof=1),
            "t_stat": t_stat,
            "p_value": p_one,
            "n_subjects": len(A_values),
            "effect_size_d": d,
        })
        p_values.append(p_one)

    df = pd.DataFrame(rows)

    # FDR correction
    reject, p_corr, _, _ = multipletests(p_values, method="fdr_bh")
    df["p_corrected"] = p_corr
    df["significant"] = reject

    return df


def test_prediction_2(subject_profiles, ages, band_names):
    """Test A(τ) ~ age at each band (linear regression).

    Parameters
    ----------
    subject_profiles : list of dict
    ages : array-like
    band_names : list of str

    Returns
    -------
    results : pd.DataFrame
        Columns: band, slope, intercept, r_squared, p_value, p_corrected,
                 significant, n_subjects
    """
    ages = np.array(ages, dtype=float)
    rows = []
    p_values = []

    for band in band_names:
        A_values = []
        age_values = []
        for i, sp in enumerate(subject_profiles):
            if band in sp and not np.isnan(sp[band]["A"]):
                A_values.append(sp[band]["A"])
                age_values.append(ages[i])

        A_values = np.array(A_values)
        age_values = np.array(age_values)

        if len(A_values) < 10:
            rows.append({
                "band": band, "slope": np.nan, "intercept": np.nan,
                "r_squared": np.nan, "p_value": np.nan, "n_subjects": len(A_values),
            })
            p_values.append(1.0)
            continue

        X = sm.add_constant(age_values)
        model = sm.OLS(A_values, X).fit()

        rows.append({
            "band": band,
            "slope": model.params[1],
            "intercept": model.params[0],
            "r_squared": model.rsquared,
            "p_value": model.pvalues[1],
            "n_subjects": len(A_values),
            "slope_ci_low": model.conf_int()[1, 0],
            "slope_ci_high": model.conf_int()[1, 1],
        })
        p_values.append(model.pvalues[1])

    df = pd.DataFrame(rows)
    reject, p_corr, _, _ = multipletests(p_values, method="fdr_bh")
    df["p_corrected"] = p_corr
    df["significant"] = reject

    return df


def test_prediction_3(rho_cross_values, ages, individual_A_eeg, individual_A_hrv,
                      band_names_eeg, band_names_hrv):
    """Test that ρ_cross decreases with age faster than individual A.

    Parameters
    ----------
    rho_cross_values : array
        Per-subject ρ_cross.
    ages : array
    individual_A_eeg : list of dict
        Per-subject EEG profiles.
    individual_A_hrv : list of dict
        Per-subject HRV profiles.
    band_names_eeg : list
    band_names_hrv : list

    Returns
    -------
    results : dict
        'rho_cross_vs_age': regression results for ρ_cross ~ age
        'eeg_A_mean_vs_age': regression for mean EEG A ~ age
        'hrv_A_mean_vs_age': regression for mean HRV A ~ age
        'comparison': whether |slope_rho| > |slope_A_individual|
    """
    ages = np.array(ages, dtype=float)
    rho_cross_values = np.array(rho_cross_values, dtype=float)

    def regress(y, x, label):
        valid = ~np.isnan(y) & ~np.isnan(x)
        if valid.sum() < 10:
            return {"label": label, "slope": np.nan, "p_value": np.nan}
        X = sm.add_constant(x[valid])
        model = sm.OLS(y[valid], X).fit()
        return {
            "label": label,
            "slope": model.params[1],
            "p_value": model.pvalues[1],
            "r_squared": model.rsquared,
            "n": int(valid.sum()),
        }

    # ρ_cross vs age
    rho_result = regress(rho_cross_values, ages, "rho_cross ~ age")

    # Mean EEG A across bands vs age
    mean_eeg_A = np.array([
        np.nanmean([sp[b]["A"] for b in band_names_eeg if b in sp])
        for sp in individual_A_eeg
    ])
    eeg_result = regress(mean_eeg_A, ages, "mean_EEG_A ~ age")

    # Mean HRV A across bands vs age
    mean_hrv_A = np.array([
        np.nanmean([sp[b]["A"] for b in band_names_hrv if b in sp])
        for sp in individual_A_hrv
    ])
    hrv_result = regress(mean_hrv_A, ages, "mean_HRV_A ~ age")

    # Compare effect sizes
    comparison = {
        "rho_slope": rho_result.get("slope", np.nan),
        "eeg_slope": eeg_result.get("slope", np.nan),
        "hrv_slope": hrv_result.get("slope", np.nan),
        "rho_faster_than_eeg": abs(rho_result.get("slope", 0)) > abs(eeg_result.get("slope", 0)),
        "rho_faster_than_hrv": abs(rho_result.get("slope", 0)) > abs(hrv_result.get("slope", 0)),
    }

    return {
        "rho_cross_vs_age": rho_result,
        "eeg_A_mean_vs_age": eeg_result,
        "hrv_A_mean_vs_age": hrv_result,
        "comparison": comparison,
    }


def test_prediction_4(te_results, ages):
    """Test transfer entropy shifts with age.

    Parameters
    ----------
    te_results : list of dict
        Per-subject: {'te_up': float, 'te_down': float, 'net': float}
        te_up = TE(fast→slow), te_down = TE(slow→fast)
    ages : array

    Returns
    -------
    results : dict
        Regression results for te_up, te_down, and net flow vs age.
    """
    ages = np.array(ages, dtype=float)

    te_up = np.array([t["te_up"] for t in te_results])
    te_down = np.array([t["te_down"] for t in te_results])
    te_net = np.array([t["net"] for t in te_results])

    results = {}
    for name, values in [("te_fast_to_slow", te_up),
                          ("te_slow_to_fast", te_down),
                          ("te_net_flow", te_net)]:
        valid = ~np.isnan(values) & ~np.isnan(ages)
        if valid.sum() < 10:
            results[name] = {"slope": np.nan, "p_value": np.nan}
            continue

        X = sm.add_constant(ages[valid])
        model = sm.OLS(values[valid], X).fit()
        results[name] = {
            "slope": model.params[1],
            "p_value": model.pvalues[1],
            "r_squared": model.rsquared,
            "n": int(valid.sum()),
        }

    return results
