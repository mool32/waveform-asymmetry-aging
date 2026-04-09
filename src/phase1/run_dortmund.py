"""
Dortmund Vital Study PTA replication.

Applies IDENTICAL pipeline to LEMON run_lemon.py but on Dortmund EEG data.
Input: EDF files from OpenNeuro ds005385.
Output: summary.json in same format as LEMON run.

BLIND PREDICTIONS (recorded before analysis):
  - beta prop_positive ~ age: r in [-0.40, -0.20], point estimate -0.30
  - sigma_A_beta ~ age: r > 0
  - H(sign pattern) ~ age: r < 0
  - Falsification: |r| < 0.15 for primary metric
"""

import json
import sys
import os
import numpy as np
import mne
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.asymmetry import (
    find_broadband_cycles,
    broadband_asymmetry_by_band,
    compute_spatial_asymmetry,
    EEG_REGIONS,
)


def load_dortmund_eeg(edf_path):
    """Load a Dortmund EDF file, apply minimal preprocessing.

    Identical pipeline to LEMON:
    1. Load raw EEG
    2. Pick EEG channels only
    3. Notch filter 50 Hz (European line noise)
    4. Average reference
    5. Return channel_data, ch_names, sfreq
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Pick only EEG channels
    raw.pick_types(eeg=True, exclude='bads')

    # Notch filter: 50 Hz (German line noise)
    raw.notch_filter(freqs=[50, 100], verbose=False)

    # Average reference
    raw.set_eeg_reference('average', projection=False, verbose=False)

    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    data = raw.get_data()  # (n_channels, n_samples)

    return data, ch_names, sfreq


def compute_subject_metrics(channel_data, ch_names, sfreq, min_cycles=30):
    """Compute all PTA + spatial metrics for one subject.

    Returns dict with same structure as LEMON pipeline output.
    """
    per_channel, per_region, spatial_stats = compute_spatial_asymmetry(
        channel_data, ch_names, sfreq,
        regions=EEG_REGIONS,
        f_low=1.0, f_high=45.0,
        min_cycles=min_cycles,
    )

    if not spatial_stats:
        return None

    # Build output dict
    result = {
        'sigma_global': spatial_stats.get('sigma_A', np.nan),
    }

    bands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma']
    for band in bands:
        result[f'sigma_A_{band}'] = spatial_stats['sigma_A_per_band'].get(band, np.nan)
        result[f'prop_positive_{band}'] = spatial_stats['prop_positive_per_band'].get(band, np.nan)
        result[f'trimmed_mean_A_{band}'] = spatial_stats['trimmed_mean_A_per_band'].get(band, np.nan)
        result[f'mean_A_{band}'] = spatial_stats['mean_A_per_band'].get(band, np.nan)
        result[f'n_included_{band}'] = spatial_stats['n_included_per_band'].get(band, 0)

    # Per-channel asymmetry (for spatial pattern analysis)
    result['per_channel_A'] = {}
    for ch in per_channel:
        result['per_channel_A'][ch] = {}
        for band in bands:
            val = per_channel[ch][band]['A_mean']
            result['per_channel_A'][ch][band] = float(val) if not np.isnan(val) else None

    # Per-region
    result['regions'] = {}
    for region_name, region_data in per_region.items():
        result['regions'][region_name] = region_data['per_band']

    # Coarsening metric: H(sign pattern)
    # For each band, create binary pattern across channels (A > 0 or A < 0)
    for band in bands:
        included_chs = [ch for ch in per_channel
                        if per_channel[ch][band]['n_cycles'] >= min_cycles
                        and not np.isnan(per_channel[ch][band]['A_mean'])]
        if len(included_chs) >= 5:
            signs = [1 if per_channel[ch][band]['A_mean'] > 0 else 0
                     for ch in included_chs]
            pattern = tuple(signs)
            # For a single subject, H = 0 (one pattern).
            # We store the sign vector for cross-subject H computation.
            result[f'sign_pattern_{band}'] = signs
            result[f'n_channels_pattern_{band}'] = len(signs)
        else:
            result[f'sign_pattern_{band}'] = None
            result[f'n_channels_pattern_{band}'] = 0

    return result


def run_dortmund(data_dir, output_dir, n_subjects=None, session='ses-1',
                 task='task-EyesClosed', acq='acq-pre'):
    """Run PTA pipeline on Dortmund dataset.

    Parameters
    ----------
    data_dir : str
        Path to dortmund_eeg directory (with participants.tsv and sub-* dirs).
    output_dir : str
        Output directory for results.
    n_subjects : int or None
        Limit number of subjects (for testing).
    session, task, acq : str
        BIDS selectors.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load participants
    participants = pd.read_csv(data_dir / 'participants.tsv', sep='\t')
    print(f"Participants: {len(participants)}")
    print(f"Age range: {participants['age'].min()}-{participants['age'].max()}")

    # Find available EEG files
    subjects = []
    for _, row in participants.iterrows():
        sub_id = row['participant_id']
        eeg_dir = data_dir / sub_id / session / 'eeg'
        edf_pattern = f"{sub_id}_{session}_{task}_{acq}_eeg.edf"
        edf_path = eeg_dir / edf_pattern

        if edf_path.exists():
            subjects.append({
                'id': sub_id,
                'age': row['age'],
                'sex': row['sex'],
                'edf_path': str(edf_path),
            })

    print(f"Found EEG files: {len(subjects)}")

    if n_subjects is not None:
        subjects = subjects[:n_subjects]
        print(f"Limited to: {n_subjects}")

    # Process each subject
    results = []
    for i, sub in enumerate(subjects):
        try:
            data, ch_names, sfreq = load_dortmund_eeg(sub['edf_path'])
            metrics = compute_subject_metrics(data, ch_names, sfreq)

            if metrics is None:
                print(f"  [{i+1}/{len(subjects)}] {sub['id']} — SKIPPED (no valid data)")
                continue

            metrics['id'] = sub['id']
            metrics['age'] = sub['age']
            metrics['sex'] = sub['sex']
            results.append(metrics)

            pp = metrics.get('prop_positive_beta', np.nan)
            sa = metrics.get('sigma_A_beta', np.nan)
            print(f"  [{i+1}/{len(subjects)}] {sub['id']} age={sub['age']} "
                  f"prop_pos_beta={pp:.3f} sigma_beta={sa:.4f}")

        except Exception as e:
            print(f"  [{i+1}/{len(subjects)}] {sub['id']} — ERROR: {e}")

    print(f"\nProcessed: {len(results)}/{len(subjects)} subjects")

    # ── Save results ──
    # Remove non-serializable items for JSON
    json_results = []
    for r in results:
        jr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                jr[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, np.ndarray):
                jr[k] = v.tolist()
            elif isinstance(v, dict):
                jr[k] = _clean_dict(v)
            else:
                jr[k] = v
        json_results.append(jr)

    summary = {
        'dataset': 'dortmund_vital_study',
        'openneuro_id': 'ds005385',
        'n_subjects': len(json_results),
        'session': session,
        'task': task,
        'acq': acq,
        'bands': ['delta', 'theta', 'alpha', 'beta', 'low_gamma'],
        'subjects': json_results,
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Run statistical tests ──
    print("\n" + "=" * 70)
    print("BLIND PREDICTION TESTS")
    print("=" * 70)

    ages = np.array([r['age'] for r in results])

    bands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma']

    print("\n── Prediction 1: prop_positive ~ age ──")
    for band in bands:
        pp = np.array([r.get(f'prop_positive_{band}', np.nan) for r in results])
        valid = ~np.isnan(pp)
        if valid.sum() > 10:
            r_val, p_val = stats.pearsonr(ages[valid], pp[valid])
            rho, p_rho = stats.spearmanr(ages[valid], pp[valid])
            marker = ' ← PRIMARY' if band == 'beta' else ''
            print(f"  {band:12s}: r={r_val:+.3f} (p={p_val:.2e}), "
                  f"ρ={rho:+.3f} (p={p_rho:.2e}), N={valid.sum()}{marker}")

    print("\n── Prediction 2: sigma_A ~ age ──")
    for band in bands:
        sa = np.array([r.get(f'sigma_A_{band}', np.nan) for r in results])
        valid = ~np.isnan(sa)
        if valid.sum() > 10:
            r_val, p_val = stats.pearsonr(ages[valid], sa[valid])
            print(f"  {band:12s}: r={r_val:+.3f} (p={p_val:.2e}), N={valid.sum()}")

    print("\n── Prediction 3: Band specificity (largest |r| in beta?) ──")
    r_by_band = {}
    for band in bands:
        pp = np.array([r.get(f'prop_positive_{band}', np.nan) for r in results])
        valid = ~np.isnan(pp)
        if valid.sum() > 10:
            r_val, _ = stats.pearsonr(ages[valid], pp[valid])
            r_by_band[band] = abs(r_val)
    if r_by_band:
        best_band = max(r_by_band, key=r_by_band.get)
        print(f"  |r| by band: {', '.join(f'{b}={v:.3f}' for b, v in r_by_band.items())}")
        print(f"  Strongest: {best_band} ({'CONFIRMED' if best_band == 'beta' else 'NOT beta'})")

    print("\n── Prediction 4: prop_positive > 0.5 at all ages ──")
    for decade_start in range(20, 70, 10):
        decade_end = decade_start + 10
        mask = (ages >= decade_start) & (ages < decade_end)
        pp_beta = np.array([r.get('prop_positive_beta', np.nan) for r in results])
        valid = mask & ~np.isnan(pp_beta)
        if valid.sum() > 0:
            mean_pp = np.mean(pp_beta[valid])
            print(f"  {decade_start}-{decade_end}: mean prop_positive_beta = {mean_pp:.3f} "
                  f"({'OK' if mean_pp > 0.5 else 'FAIL'}), N={valid.sum()}")

    # Save test results
    test_results = {
        'primary': {
            'metric': 'prop_positive_beta ~ age',
            'prediction': 'r in [-0.40, -0.20]',
            'falsification': '|r| < 0.15',
        },
    }

    pp_beta = np.array([r.get('prop_positive_beta', np.nan) for r in results])
    valid = ~np.isnan(pp_beta)
    if valid.sum() > 10:
        r_val, p_val = stats.pearsonr(ages[valid], pp_beta[valid])
        test_results['primary']['r'] = float(r_val)
        test_results['primary']['p'] = float(p_val)
        test_results['primary']['n'] = int(valid.sum())
        test_results['primary']['replicated'] = abs(r_val) >= 0.15

    with open(output_dir / 'blind_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return results


def _clean_dict(d):
    """Recursively clean dict for JSON serialization."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if not np.isnan(float(v)) else None
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _clean_dict(v)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        elif isinstance(v, (int, float, str, type(None))):
            out[k] = v
        else:
            out[k] = v
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/Users/teo/Desktop/research/dortmund_eeg')
    parser.add_argument('--output', default=None)
    parser.add_argument('--n-subjects', type=int, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(PROJECT_ROOT / 'results' / 'dortmund_replication')

    run_dortmund(args.data_dir, args.output, n_subjects=args.n_subjects)
