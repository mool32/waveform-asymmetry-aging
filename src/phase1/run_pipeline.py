"""
Phase 1: Main pipeline — runs full LEMON analysis.

Usage:
    python -m src.phase1.run_pipeline --bids-root /path/to/lemon --output results/phase1/
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocess_eeg import (
    EEG_BANDS, load_lemon_eeg, preprocess_eeg, compute_eeg_asymmetry
)
from .preprocess_ecg import (
    HRV_BANDS, compute_hrv_asymmetry, compute_acceleration_deceleration
)
from .cross_system import compute_cross_coherence
from .statistics import (
    test_prediction_1, test_prediction_2, test_prediction_3, test_prediction_4
)
from ..utils.asymmetry import bandpass_decompose, surrogate_test


def load_participants(bids_root):
    """Load participant demographics from BIDS participants.tsv."""
    tsv_path = Path(bids_root) / "participants.tsv"
    if tsv_path.exists():
        df = pd.read_csv(tsv_path, sep="\t")
        return df
    # Try JSON sidecar
    json_path = Path(bids_root) / "participants.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    raise FileNotFoundError(f"No participants file found in {bids_root}")


def process_single_subject(subject_id, bids_root, task="EC"):
    """Process one subject through full pipeline.

    Returns
    -------
    result : dict or None
        Subject-level results including EEG and HRV asymmetry profiles,
        cross-coherence, and transfer entropy.
    """
    print(f"\n{'='*60}")
    print(f"Processing {subject_id}")
    print(f"{'='*60}")

    try:
        # Load EEG
        raw = load_lemon_eeg(subject_id, bids_root, task=task)

        # Preprocess
        raw_clean, ecg_signal, ecg_sfreq, ica = preprocess_eeg(raw)

        # EEG asymmetry
        eeg_per_channel, eeg_grand = compute_eeg_asymmetry(raw_clean)

        result = {
            "subject_id": subject_id,
            "eeg_profile": eeg_grand,
            "eeg_per_channel": eeg_per_channel,
            "n_eeg_channels": len(raw_clean.ch_names),
            "eeg_sfreq": raw_clean.info["sfreq"],
        }

        # ECG/HRV asymmetry
        if ecg_signal is not None:
            hrv_profile, rr_times, rr_intervals, hrv_signal = compute_hrv_asymmetry(
                ecg_signal, ecg_sfreq
            )

            if hrv_profile is not None:
                result["hrv_profile"] = hrv_profile
                result["hrv_sfreq"] = 4.0
                result["n_rr_intervals"] = len(rr_intervals)
                result["mean_hr"] = 60.0 / np.mean(rr_intervals) if len(rr_intervals) > 0 else np.nan

                # Acceleration/deceleration capacity
                ac, dc, ratio = compute_acceleration_deceleration(rr_intervals)
                result["accel_capacity"] = ac
                result["decel_capacity"] = dc
                result["dc_ac_ratio"] = ratio

                # Cross-system coherence
                rho_p, rho_s, p_p, p_s, _, _, _ = compute_cross_coherence(
                    eeg_grand, hrv_profile
                )
                result["rho_cross_pearson"] = rho_p
                result["rho_cross_spearman"] = rho_s
                result["rho_cross_p_pearson"] = p_p
                result["rho_cross_p_spearman"] = p_s

                # Surrogate test on EEG (grand average signal from Cz or first channel)
                cz_channel = "Cz" if "Cz" in eeg_per_channel else list(eeg_per_channel.keys())[0]
                cz_data = raw_clean.get_data(picks=[cz_channel]).flatten()
                surr = surrogate_test(cz_data, raw_clean.info["sfreq"], EEG_BANDS,
                                      n_surrogates=200)
                result["surrogate_test_eeg"] = {
                    band: {"p_value": v["p_value"], "z_score": v["z_score"]}
                    for band, v in surr.items()
                }

        print(f"  EEG A(τ): {', '.join(f'{b}={v[\"A_mean\"]:.3f}' for b, v in eeg_grand.items())}")
        if "hrv_profile" in result:
            print(f"  HRV A(τ): {', '.join(f'{b}={v[\"A\"]:.3f}' for b, v in result['hrv_profile'].items())}")
            print(f"  ρ_cross: {result.get('rho_cross_pearson', 'N/A'):.3f}")

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_full_pipeline(bids_root, output_dir, task="EC", n_subjects=None):
    """Run Phase 1 pipeline on all LEMON subjects.

    Parameters
    ----------
    bids_root : str
    output_dir : str
    task : str
    n_subjects : int or None
        Limit to first N subjects (for testing).
    """
    bids_root = Path(bids_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load demographics
    participants = load_participants(bids_root)
    print(f"Found {len(participants)} participants")

    if n_subjects is not None:
        participants = participants.head(n_subjects)

    # Process each subject
    all_results = []
    for _, row in tqdm(participants.iterrows(), total=len(participants),
                       desc="Processing subjects"):
        subject_id = row.get("participant_id", row.get("subject_id"))
        result = process_single_subject(subject_id, bids_root, task=task)
        if result is not None:
            result["age"] = row.get("age", np.nan)
            result["sex"] = row.get("sex", row.get("gender", "unknown"))
            all_results.append(result)

    print(f"\nSuccessfully processed {len(all_results)} / {len(participants)} subjects")

    # Save raw results
    with open(output_dir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    # Extract arrays for statistical tests
    ages = np.array([r["age"] for r in all_results])
    eeg_profiles = [r["eeg_profile"] for r in all_results]
    eeg_bands = list(EEG_BANDS.keys())

    # Prediction 1: A(τ) > 0
    print("\n" + "="*60)
    print("PREDICTION 1: A(τ) > 0")
    pred1 = test_prediction_1(eeg_profiles, eeg_bands)
    print(pred1.to_string(index=False))
    pred1.to_csv(output_dir / "prediction_1_eeg.csv", index=False)

    # Prediction 2: A(τ) ~ age
    print("\nPREDICTION 2: A(τ) ~ age")
    pred2 = test_prediction_2(eeg_profiles, ages, eeg_bands)
    print(pred2.to_string(index=False))
    pred2.to_csv(output_dir / "prediction_2_eeg.csv", index=False)

    # Predictions 3 & 4: only for subjects with HRV data
    hrv_subjects = [r for r in all_results if "hrv_profile" in r]
    if len(hrv_subjects) > 10:
        hrv_profiles = [r["hrv_profile"] for r in hrv_subjects]
        hrv_bands = list(HRV_BANDS.keys())

        # HRV Prediction 1
        pred1_hrv = test_prediction_1(hrv_profiles, hrv_bands)
        print("\nPREDICTION 1 (HRV): A(τ) > 0")
        print(pred1_hrv.to_string(index=False))
        pred1_hrv.to_csv(output_dir / "prediction_1_hrv.csv", index=False)

        # Prediction 3
        rho_values = np.array([r["rho_cross_pearson"] for r in hrv_subjects])
        hrv_ages = np.array([r["age"] for r in hrv_subjects])
        hrv_eeg_profiles = [r["eeg_profile"] for r in hrv_subjects]

        pred3 = test_prediction_3(
            rho_values, hrv_ages, hrv_eeg_profiles, hrv_profiles,
            eeg_bands, hrv_bands
        )
        print("\nPREDICTION 3: ρ_cross degrades faster than individual A")
        for key, val in pred3.items():
            print(f"  {key}: {val}")

        with open(output_dir / "prediction_3.json", "w") as f:
            json.dump(pred3, f, indent=2, default=str)

    # Save summary
    summary = {
        "n_subjects_total": len(all_results),
        "n_subjects_with_hrv": len(hrv_subjects) if 'hrv_subjects' in dir() else 0,
        "age_range": [float(np.nanmin(ages)), float(np.nanmax(ages))],
        "task": task,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: LEMON Asymmetry Analysis")
    parser.add_argument("--bids-root", required=True, help="Path to LEMON BIDS root")
    parser.add_argument("--output", default="results/phase1/", help="Output directory")
    parser.add_argument("--task", default="EC", choices=["EC", "EO"])
    parser.add_argument("--n-subjects", type=int, default=None, help="Limit subjects (for testing)")
    args = parser.parse_args()

    run_full_pipeline(args.bids_root, args.output, args.task, args.n_subjects)


if __name__ == "__main__":
    main()
