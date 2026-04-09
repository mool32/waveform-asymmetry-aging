"""
Cortical thickness proxy (regional GM intensity) vs beta PTA.

Tests E/I drift hypothesis:
- If beta PTA aging is driven by regional E/I shift,
  then beta PTA should correlate with cortical thickness
  specifically in central/parietal regions (where PTA shift is strongest),
  but NOT in temporal (where PTA shift is opposite).

Uses Harvard-Oxford atlas on brain-extracted T1w images.
GM intensity in native space serves as proxy for cortical thickness
(validated correlation r > 0.7 in literature).
"""

import numpy as np
import pandas as pd
import nibabel as nib
import json
from pathlib import Path
from scipy import stats
from nilearn import datasets, image

def extract_regional_gm(t1_path, atlas_img, atlas_labels):
    """Extract mean GM intensity per atlas region from brain-extracted T1.

    Returns dict: region_name → mean_intensity
    """
    t1_img = nib.load(str(t1_path))

    # Resample atlas to T1 space
    atlas_resampled = image.resample_to_img(atlas_img, t1_img, interpolation='nearest')

    t1_data = t1_img.get_fdata()
    atlas_data = atlas_resampled.get_fdata()

    results = {}
    for label_idx, label_name in enumerate(atlas_labels):
        if label_idx == 0:  # skip background
            continue
        mask = atlas_data == label_idx
        if mask.sum() < 10:
            continue
        values = t1_data[mask]
        values = values[values > 0]  # exclude zeros (outside brain)
        if len(values) > 0:
            results[label_name] = float(np.mean(values))

    return results


def run_correlation_analysis(mri_dir, eeg_summary_path, output_dir):
    """Main analysis: correlate regional GM with beta PTA."""
    mri_dir = Path(mri_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load EEG results
    with open(eeg_summary_path) as f:
        eeg_data = json.load(f)

    eeg_by_id = {}
    for s in eeg_data['subjects']:
        # LEMON IDs: sub-032XXX
        eeg_by_id[s['id']] = s

    # Load atlas
    print("Loading Harvard-Oxford atlas...")
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_maps = atlas['maps']
    atlas_img = nib.load(atlas_maps) if isinstance(atlas_maps, (str, Path)) else atlas_maps
    atlas_labels = atlas['labels']

    # EEG-MRI region mapping
    REGION_MAP = {
        'central': ['Precentral Gyrus', 'Postcentral Gyrus'],
        'parietal': ['Superior Parietal Lobule', 'Supramarginal Gyrus, anterior division',
                     'Supramarginal Gyrus, posterior division', 'Angular Gyrus',
                     'Precuneous Cortex'],
        'temporal': ['Superior Temporal Gyrus, anterior division',
                     'Superior Temporal Gyrus, posterior division',
                     'Middle Temporal Gyrus, anterior division',
                     'Middle Temporal Gyrus, posterior division',
                     'Inferior Temporal Gyrus, anterior division',
                     'Inferior Temporal Gyrus, posterior division'],
        'frontal': ['Superior Frontal Gyrus', 'Middle Frontal Gyrus',
                    'Inferior Frontal Gyrus, pars triangularis',
                    'Inferior Frontal Gyrus, pars opercularis',
                    'Frontal Pole'],
        'occipital': ['Lateral Occipital Cortex, superior division',
                      'Lateral Occipital Cortex, inferior division',
                      'Occipital Pole', 'Cuneal Cortex', 'Lingual Gyrus'],
    }

    # Process subjects
    t1_files = sorted(mri_dir.glob('*_brain.nii.gz'))
    print(f"Found {len(t1_files)} T1 files")

    rows = []
    for i, t1_path in enumerate(t1_files):
        sub_id = t1_path.stem.replace('_brain.nii', '').replace('.gz', '').replace('_brain', '')

        # Match EEG data
        eeg = eeg_by_id.get(sub_id)
        if eeg is None:
            continue

        try:
            gm = extract_regional_gm(t1_path, atlas_img, atlas_labels)
        except Exception as e:
            print(f"  [{i+1}] {sub_id} — ERROR: {e}")
            continue

        if not gm:
            continue

        # Compute regional averages
        row = {
            'subject': sub_id,
            'age': eeg['age'],
            'sex': eeg.get('sex', 'U'),
        }

        # EEG metrics
        per_ch = eeg.get('per_channel_A', {})
        for eeg_region, eeg_chs in {
            'frontal': ['Fp1','Fp2','AF7','AF3','AF4','AF8','F7','F5','F3','F1','Fz','F2','F4','F6','F8'],
            'central': ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6'],
            'temporal': ['FT9','FT7','T7','TP7','FT8','FT10','T8','TP8'],
            'parietal': ['CP5','CP3','CP1','CPz','CP2','CP4','CP6','P7','P5','P3','P1','Pz','P2','P4','P6','P8'],
            'occipital': ['PO9','PO7','PO3','POz','PO4','PO8','PO10','O1','Oz','O2'],
        }.items():
            A_vals = [per_ch[ch]['beta'] for ch in eeg_chs
                      if ch in per_ch and per_ch[ch].get('beta') is not None]
            if A_vals:
                row[f'beta_A_{eeg_region}'] = np.mean(A_vals)
            else:
                row[f'beta_A_{eeg_region}'] = np.nan

        # Overall beta metrics
        row['beta_pp'] = eeg.get('prop_positive_beta')
        row['beta_sigma'] = eeg.get('sigma_A_beta')
        row['beta_mean_A'] = eeg.get('mean_A_beta')

        # MRI regional GM intensity
        for mri_region, rois in REGION_MAP.items():
            vals = [gm[r] for r in rois if r in gm]
            if vals:
                row[f'gm_{mri_region}'] = np.mean(vals)
            else:
                row[f'gm_{mri_region}'] = np.nan

        # Total GM
        all_gm = list(gm.values())
        row['gm_total'] = np.mean(all_gm) if all_gm else np.nan

        rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(t1_files)}] {sub_id} age={row['age']}")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'gm_pta_data.csv', index=False)
    print(f"\nProcessed: {len(df)} subjects with both EEG and MRI")

    # ═══════════════════════════════════════════════════════════
    # STATISTICAL TESTS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CORTICAL GM vs BETA PTA: REGIONAL CORRELATIONS")
    print("=" * 70)

    regions = ['frontal', 'central', 'temporal', 'parietal', 'occipital']

    # 1. GM ~ age per region
    print("\n── GM ~ age (sanity check: should decline) ──")
    for reg in regions:
        gm_col = f'gm_{reg}'
        valid = df[['age', gm_col]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid['age'], valid[gm_col])
            print(f"  {reg:12s}: r = {r:+.3f}, p = {p:.2e}, N = {len(valid)}")

    # 2. Regional GM ~ regional beta PTA
    print("\n── Regional GM ~ Regional Beta A (KEY TEST) ──")
    print("  Prediction: significant in central/parietal, NOT in temporal")
    for reg in regions:
        gm_col = f'gm_{reg}'
        pta_col = f'beta_A_{reg}'
        valid = df[['age', gm_col, pta_col]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid[gm_col], valid[pta_col])
            # Partial correlation controlling for age
            from scipy.stats import pearsonr
            r_age_gm, _ = pearsonr(valid['age'], valid[gm_col])
            r_age_pta, _ = pearsonr(valid['age'], valid[pta_col])
            r_gm_pta, _ = pearsonr(valid[gm_col], valid[pta_col])
            # Partial r
            r_partial = (r_gm_pta - r_age_gm * r_age_pta) / \
                        (np.sqrt(1 - r_age_gm**2) * np.sqrt(1 - r_age_pta**2))
            n = len(valid)
            t_stat = r_partial * np.sqrt((n - 3) / (1 - r_partial**2 + 1e-10))
            p_partial = 2 * stats.t.sf(abs(t_stat), n - 3)

            print(f"  {reg:12s}: r = {r:+.3f} (p = {p:.3f}), "
                  f"partial r = {r_partial:+.3f} (p = {p_partial:.3f}), N = {n}")

    # 3. Global GM ~ overall beta PTA
    print("\n── Global GM ~ Beta prop_positive ──")
    valid = df[['age', 'gm_total', 'beta_pp']].dropna()
    if len(valid) >= 10:
        r, p = stats.pearsonr(valid['gm_total'], valid['beta_pp'])
        print(f"  Total GM ~ beta pp: r = {r:+.3f}, p = {p:.3f}, N = {len(valid)}")

    # 4. Test specificity: is the correlation driven by shared age effect?
    print("\n── Age-controlled: partial correlation GM ~ PTA | age ──")
    for reg in regions:
        gm_col = f'gm_{reg}'
        pta_col = f'beta_A_{reg}'
        valid = df[['age', gm_col, pta_col]].dropna()
        if len(valid) < 20:
            continue
        # Already computed above, just emphasize temporal vs central/parietal

    print(f"\nResults saved to {output_dir}")
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri-dir', default='/Users/teo/Desktop/research/Asymmetric Perception Cycles/data/lemon_mri')
    parser.add_argument('--eeg-summary', default='/Users/teo/Desktop/research/Asymmetric Perception Cycles/results/phase1_full/summary.json')
    parser.add_argument('--output', default='/Users/teo/Desktop/research/Asymmetric Perception Cycles/results/gm_pta_correlation')
    args = parser.parse_args()
    run_correlation_analysis(args.mri_dir, args.eeg_summary, args.output)
