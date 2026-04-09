"""
Dortmund Longitudinal Analysis: within-subject PTA change over ~5 years.

For 208 subjects with ses-1 and ses-2 data:
  Δ(prop_positive_beta) = ses2 - ses1
  Prediction: Δ < 0 (beta asymmetry decreases with aging)

This is the strongest possible test: each subject is their own control.
"""

import json
import sys
import os
import numpy as np
import mne
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.asymmetry import compute_spatial_asymmetry, EEG_REGIONS


def process_one_session(edf_path, sfreq_target=None):
    """Load EDF, preprocess, compute PTA metrics. Returns dict or None."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.pick('eeg', exclude='bads')
    raw.notch_filter(freqs=[50, 100], verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)

    _, _, spatial = compute_spatial_asymmetry(
        raw.get_data(), raw.ch_names, raw.info['sfreq'],
        regions=EEG_REGIONS, f_low=1.0, f_high=45.0, min_cycles=30
    )

    if not spatial:
        return None

    result = {}
    bands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma']
    for band in bands:
        result[f'prop_positive_{band}'] = spatial['prop_positive_per_band'].get(band, np.nan)
        result[f'sigma_A_{band}'] = spatial['sigma_A_per_band'].get(band, np.nan)
        result[f'mean_A_{band}'] = spatial['mean_A_per_band'].get(band, np.nan)
        result[f'n_included_{band}'] = spatial['n_included_per_band'].get(band, 0)
    result['sigma_global'] = spatial.get('sigma_A', np.nan)
    return result


def run_longitudinal(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants = pd.read_csv(data_dir / 'participants.tsv', sep='\t')
    follow_up = participants[participants['session2'] == 'yes']
    print(f"Subjects with follow-up: {len(follow_up)}")

    pairs = []
    for _, row in follow_up.iterrows():
        sub_id = row['participant_id']
        age = row['age']
        sex = row['sex']

        edf_s1 = data_dir / sub_id / 'ses-1' / 'eeg' / f"{sub_id}_ses-1_task-EyesClosed_acq-pre_eeg.edf"
        edf_s2 = data_dir / sub_id / 'ses-2' / 'eeg' / f"{sub_id}_ses-2_task-EyesClosed_acq-pre_eeg.edf"

        if not edf_s1.exists() or not edf_s2.exists():
            continue

        try:
            r1 = process_one_session(edf_s1)
            r2 = process_one_session(edf_s2)

            if r1 is None or r2 is None:
                print(f"  {sub_id} — SKIPPED")
                continue

            pair = {
                'id': sub_id, 'age_baseline': age, 'sex': sex,
            }
            # Store both sessions + delta
            bands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma']
            for band in bands:
                for metric in ['prop_positive', 'sigma_A', 'mean_A']:
                    k = f'{metric}_{band}'
                    v1 = r1.get(k, np.nan)
                    v2 = r2.get(k, np.nan)
                    pair[f'{k}_s1'] = v1
                    pair[f'{k}_s2'] = v2
                    if not np.isnan(v1) and not np.isnan(v2):
                        pair[f'd_{k}'] = v2 - v1
                    else:
                        pair[f'd_{k}'] = np.nan

            pairs.append(pair)
            pp1 = pair.get('prop_positive_beta_s1', np.nan)
            pp2 = pair.get('prop_positive_beta_s2', np.nan)
            d = pair.get('d_prop_positive_beta', np.nan)
            print(f"  [{len(pairs)}] {sub_id} age={age} "
                  f"pp_beta: {pp1:.3f} → {pp2:.3f} (Δ={d:+.3f})")

        except Exception as e:
            print(f"  {sub_id} — ERROR: {e}")

    print(f"\nProcessed pairs: {len(pairs)}")

    df = pd.DataFrame(pairs)
    df.to_csv(output_dir / 'longitudinal_pairs.csv', index=False)

    # ═══════════════════════════════════════════════════════════
    # STATISTICAL TESTS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("LONGITUDINAL RESULTS: WITHIN-SUBJECT CHANGE OVER ~5 YEARS")
    print("=" * 70)

    bands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma']

    print("\n── Prediction 6: Δ(prop_positive) < 0 (decline over 5 years) ──")
    for band in bands:
        d = df[f'd_prop_positive_{band}'].dropna()
        if len(d) < 5:
            continue
        n_neg = (d < 0).sum()
        t_stat, p_val = stats.wilcoxon(d)
        # One-sided: we predict decrease
        p_one = p_val / 2 if np.median(d) < 0 else 1 - p_val / 2
        mean_d = d.mean()
        marker = ' ← PRIMARY' if band == 'beta' else ''
        print(f"  {band:12s}: Δ = {mean_d:+.4f}, {n_neg}/{len(d)} negative ({100*n_neg/len(d):.0f}%), "
              f"Wilcoxon p(two) = {p_val:.4f}, p(one-sided) = {p_one:.4f}, N={len(d)}{marker}")

    print("\n── Δ(sigma_A) > 0 (variability increases) ──")
    for band in bands:
        d = df[f'd_sigma_A_{band}'].dropna()
        if len(d) < 5:
            continue
        n_pos = (d > 0).sum()
        t_stat, p_val = stats.wilcoxon(d)
        mean_d = d.mean()
        print(f"  {band:12s}: Δ = {mean_d:+.4f}, {n_pos}/{len(d)} positive ({100*n_pos/len(d):.0f}%), "
              f"Wilcoxon p = {p_val:.4f}, N={len(d)}")

    # Age-stratified: does Δ depend on baseline age?
    print("\n── Age-stratified Δ(prop_positive_beta) ──")
    d_beta = df[['age_baseline', 'd_prop_positive_beta']].dropna()
    if len(d_beta) >= 10:
        r_age_delta, p_age_delta = stats.pearsonr(d_beta['age_baseline'], d_beta['d_prop_positive_beta'])
        print(f"  Δ(pp_beta) ~ baseline_age: r = {r_age_delta:+.3f}, p = {p_age_delta:.4f}")
        print(f"  (Negative r = older at baseline decline more)")

        for age_cut in [35, 45, 55]:
            young = d_beta[d_beta['age_baseline'] < age_cut]['d_prop_positive_beta']
            old = d_beta[d_beta['age_baseline'] >= age_cut]['d_prop_positive_beta']
            if len(young) >= 5 and len(old) >= 5:
                u, p = stats.mannwhitneyu(young, old)
                print(f"  Below/above {age_cut}: young Δ = {young.mean():+.4f} (N={len(young)}), "
                      f"old Δ = {old.mean():+.4f} (N={len(old)}), p = {p:.4f}")

    # Test-retest reliability
    print("\n── Test-retest reliability (ses-1 vs ses-2) ──")
    for band in bands:
        s1 = df[f'prop_positive_{band}_s1'].dropna()
        s2 = df[f'prop_positive_{band}_s2'].dropna()
        valid = df[[f'prop_positive_{band}_s1', f'prop_positive_{band}_s2']].dropna()
        if len(valid) >= 10:
            icc_r, icc_p = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
            print(f"  {band:12s}: r(s1, s2) = {icc_r:.3f} (p = {icc_p:.2e}), N={len(valid)}")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    run_longitudinal(
        data_dir='/Users/teo/Desktop/research/dortmund_eeg',
        output_dir='/Users/teo/Desktop/research/Asymmetric Perception Cycles/results/dortmund_longitudinal',
    )
