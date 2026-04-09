"""
Phase 1: Adapted pipeline for actual LEMON S3 data structure.

LEMON structure on S3:
  data/lemon/
    Participants_MPILMBB_LEMON.csv
    sub-XXXXXX/
      RSEEG/
        sub-XXXXXX.vhdr
        sub-XXXXXX.vmrk
        sub-XXXXXX.eeg

Usage:
    python3 -m src.phase1.run_lemon --data-dir data/lemon/ --output results/phase1/
    python3 -m src.phase1.run_lemon --data-dir data/lemon/ --output results/phase1/ --n-subjects 5
"""

import argparse
import json
import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from ..utils.asymmetry import (
    compute_asymmetry_profile, bandpass_decompose, surrogate_test,
)
from .preprocess_eeg import EEG_BANDS
from .preprocess_ecg import (
    HRV_BANDS, detect_r_peaks, compute_rr_intervals,
    rr_to_continuous, compute_acceleration_deceleration,
)
from .cross_system import compute_cross_coherence
from .statistics import (
    test_prediction_1, test_prediction_2, test_prediction_3, test_prediction_4,
)


def load_participants(data_dir):
    """Load LEMON participants CSV."""
    csv_path = Path(data_dir) / "Participants_MPILMBB_LEMON.csv"
    df = pd.read_csv(csv_path)
    df.columns = ["subject_id", "sex", "age_range"]

    # Convert age range to midpoint
    def age_midpoint(s):
        try:
            low, high = s.split("-")
            return (int(low) + int(high)) / 2
        except Exception:
            return np.nan

    df["age"] = df["age_range"].apply(age_midpoint)
    df["sex"] = df["sex"].map({1: "F", 2: "M"})
    return df


def find_eeg_file(data_dir, subject_id):
    """Find .vhdr file for a subject."""
    patterns = [
        Path(data_dir) / subject_id / "RSEEG" / f"{subject_id}.vhdr",
        Path(data_dir) / subject_id / "eeg" / f"{subject_id}.vhdr",
    ]
    for p in patterns:
        if p.exists():
            return p
    # Glob fallback
    matches = list(Path(data_dir).glob(f"{subject_id}/**/*.vhdr"))
    return matches[0] if matches else None


def preprocess_and_analyze(vhdr_path, subject_id):
    """Full single-subject analysis pipeline.

    Returns dict with all results or None on failure.
    """
    print(f"\n  Loading {vhdr_path}")
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    print(f"  Channels: {len(raw.ch_names)}, sfreq: {sfreq} Hz, "
          f"duration: {raw.times[-1]:.1f}s")

    # ── Extract ECG before dropping non-EEG channels ──
    ecg_signal = None
    ecg_ch_name = None
    for ch in raw.ch_names:
        if "ECG" in ch.upper() or "EKG" in ch.upper():
            ecg_ch_name = ch
            break

    if ecg_ch_name:
        ecg_signal = raw.get_data(picks=[ecg_ch_name]).flatten()
        print(f"  ECG channel found: {ecg_ch_name}")
    else:
        # Sometimes ECG is labeled differently — check for misc/bio channels
        misc_chs = [ch for ch in raw.ch_names
                    if raw.get_channel_types([ch])[0] in ("misc", "bio", "ecg")]
        if misc_chs:
            ecg_ch_name = misc_chs[0]
            ecg_signal = raw.get_data(picks=[ecg_ch_name]).flatten()
            print(f"  Using misc channel as ECG: {ecg_ch_name}")
        else:
            print("  WARNING: No ECG channel found")

    # ── EEG preprocessing ──
    # Pick only EEG channels
    eeg_ch_names = [ch for ch in raw.ch_names
                    if raw.get_channel_types([ch])[0] == "eeg"]
    if not eeg_ch_names:
        # Try picking all channels except known non-EEG
        non_eeg = {ecg_ch_name} if ecg_ch_name else set()
        for ch in raw.ch_names:
            ch_type = raw.get_channel_types([ch])[0]
            if ch_type in ("eog", "ecg", "emg", "misc", "stim", "bio"):
                non_eeg.add(ch)
        eeg_ch_names = [ch for ch in raw.ch_names if ch not in non_eeg]

    print(f"  Using {len(eeg_ch_names)} EEG channels")
    raw_eeg = raw.copy().pick(eeg_ch_names)

    # Filter
    raw_eeg.filter(l_freq=0.5, h_freq=80.0, fir_design="firwin", verbose=False)
    raw_eeg.notch_filter(freqs=[50], fir_design="firwin", verbose=False)

    # Downsample if very high sfreq (LEMON is 2500 Hz — way more than we need)
    target_sfreq = 250.0
    if sfreq > target_sfreq * 1.5:
        print(f"  Resampling {sfreq} → {target_sfreq} Hz")
        raw_eeg.resample(target_sfreq, verbose=False)
        if ecg_signal is not None:
            # Also resample ECG
            from scipy.signal import resample
            n_new = int(len(ecg_signal) * target_sfreq / sfreq)
            ecg_signal = resample(ecg_signal, n_new)
        sfreq = target_sfreq

    # ICA artifact removal
    try:
        ica = mne.preprocessing.ICA(
            n_components=20, method="fastica", random_state=42,
            max_iter=500, verbose=False,
        )
        ica.fit(raw_eeg, verbose=False)

        # Remove EOG components
        eog_idx = []
        try:
            eog_idx, _ = ica.find_bads_eog(raw_eeg, verbose=False)
        except Exception:
            pass

        # Remove ECG components
        ecg_idx = []
        if ecg_ch_name and ecg_ch_name in raw.ch_names:
            try:
                ecg_idx, _ = ica.find_bads_ecg(raw, ch_name=ecg_ch_name, verbose=False)
            except Exception:
                pass

        ica.exclude = list(set(eog_idx + ecg_idx))
        print(f"  ICA: removing {len(eog_idx)} EOG + {len(ecg_idx)} ECG components")
        raw_eeg = ica.apply(raw_eeg, verbose=False)
    except Exception as e:
        print(f"  ICA failed ({e}), continuing without ICA cleaning")

    # ── Compute EEG asymmetry ──
    data = raw_eeg.get_data()
    eeg_sfreq = raw_eeg.info["sfreq"]
    n_channels = data.shape[0]

    channel_profiles = {}
    for i in range(n_channels):
        try:
            profile = compute_asymmetry_profile(
                data[i], eeg_sfreq, EEG_BANDS, method="bandpass"
            )
            channel_profiles[raw_eeg.ch_names[i]] = profile
        except Exception:
            pass

    # Grand average EEG profile
    eeg_grand = {}
    for band in EEG_BANDS:
        A_vals = [channel_profiles[ch][band]["A"]
                  for ch in channel_profiles
                  if not np.isnan(channel_profiles[ch][band]["A"])]
        eeg_grand[band] = {
            "A": np.median(A_vals) if A_vals else np.nan,
            "A_mean": np.mean(A_vals) if A_vals else np.nan,
            "A_std": np.std(A_vals) if A_vals else np.nan,
            "n_channels": len(A_vals),
            "center_freq": (EEG_BANDS[band][0] + EEG_BANDS[band][1]) / 2,
        }

    print(f"  EEG A(τ): " + ", ".join(
        f"{b}={v['A']:.3f}" for b, v in eeg_grand.items()
        if not np.isnan(v['A'])
    ))

    result = {
        "subject_id": subject_id,
        "eeg_profile": eeg_grand,
        "n_eeg_channels": n_channels,
        "eeg_sfreq": eeg_sfreq,
    }

    # ── ECG/HRV analysis ──
    if ecg_signal is not None:
        try:
            r_peaks = detect_r_peaks(ecg_signal, sfreq, method="neurokit")
            if len(r_peaks) < 30:
                r_peaks = detect_r_peaks(ecg_signal, sfreq, method="scipy")

            rr_times, rr_intervals = compute_rr_intervals(r_peaks, sfreq)
            print(f"  HRV: {len(rr_intervals)} RR intervals, "
                  f"mean HR: {60/np.mean(rr_intervals):.1f} bpm")

            hrv_signal, t_hrv = rr_to_continuous(rr_times, rr_intervals, target_sfreq=4.0)

            if len(hrv_signal) > 120:  # at least 30s at 4Hz
                hrv_profile = compute_asymmetry_profile(
                    hrv_signal, 4.0, HRV_BANDS, method="bandpass"
                )
                result["hrv_profile"] = hrv_profile
                result["n_rr"] = len(rr_intervals)
                result["mean_hr"] = 60.0 / np.mean(rr_intervals)

                # Acceleration/deceleration
                ac, dc, ratio = compute_acceleration_deceleration(rr_intervals)
                result["accel_capacity"] = ac
                result["decel_capacity"] = dc
                result["dc_ac_ratio"] = ratio

                print(f"  HRV A(τ): " + ", ".join(
                    f"{b}={v['A']:.3f}" for b, v in hrv_profile.items()
                    if not np.isnan(v['A'])
                ))

                # Cross-coherence
                rho_p, rho_s, p_p, p_s, _, _, _ = compute_cross_coherence(
                    eeg_grand, hrv_profile
                )
                result["rho_cross_pearson"] = rho_p
                result["rho_cross_spearman"] = rho_s
                print(f"  ρ_cross (Pearson): {rho_p:.3f}")

        except Exception as e:
            print(f"  HRV analysis failed: {e}")

    # ── Surrogate test (on one representative channel) ──
    try:
        rep_ch = "Cz" if "Cz" in channel_profiles else list(channel_profiles.keys())[0]
        rep_idx = raw_eeg.ch_names.index(rep_ch)
        surr = surrogate_test(
            data[rep_idx], eeg_sfreq, EEG_BANDS, n_surrogates=100
        )
        result["surrogate_eeg"] = {
            band: {"p": v["p_value"], "z": v["z_score"]}
            for band, v in surr.items()
        }
        sig_bands = [b for b, v in surr.items() if v["p_value"] < 0.05]
        print(f"  Surrogate test ({rep_ch}): {len(sig_bands)}/{len(EEG_BANDS)} bands significant")
    except Exception as e:
        print(f"  Surrogate test failed: {e}")

    return result


def run(data_dir, output_dir, n_subjects=None):
    """Run full Phase 1 pipeline."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load participants
    participants = load_participants(data_dir)
    print(f"Participants: {len(participants)}")
    print(f"Age distribution: {participants['age'].describe()}")

    # Find subjects with downloaded EEG data
    available = []
    for _, row in participants.iterrows():
        vhdr = find_eeg_file(data_dir, row["subject_id"])
        if vhdr is not None:
            available.append((row, vhdr))

    print(f"Available EEG files: {len(available)}")

    if n_subjects:
        available = available[:n_subjects]

    # Process subjects
    all_results = []
    for row, vhdr_path in available:
        print(f"\n{'='*60}")
        print(f"Subject: {row['subject_id']} | Age: {row['age_range']} ({row['age']}) | Sex: {row['sex']}")
        print(f"{'='*60}")

        try:
            result = preprocess_and_analyze(vhdr_path, row["subject_id"])
            if result is not None:
                result["age"] = row["age"]
                result["sex"] = row["sex"]
                result["age_range"] = row["age_range"]
                all_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print(f"\n\n{'='*60}")
    print(f"PROCESSED: {len(all_results)} / {len(available)} subjects")
    print(f"{'='*60}")

    # Save raw results
    with open(output_dir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    if len(all_results) < 2:
        print("Not enough subjects for statistics")
        return all_results

    # ── Statistical tests ──
    ages = np.array([r["age"] for r in all_results])
    eeg_profiles = [r["eeg_profile"] for r in all_results]
    eeg_bands = list(EEG_BANDS.keys())

    print("\n" + "="*60)
    print("PREDICTION 1: A(τ) > 0 at all EEG scales")
    print("="*60)
    pred1 = test_prediction_1(eeg_profiles, eeg_bands)
    print(pred1.to_string(index=False))
    pred1.to_csv(output_dir / "prediction_1_eeg.csv", index=False)

    # HRV prediction 1
    hrv_subjects = [r for r in all_results if "hrv_profile" in r]
    if hrv_subjects:
        hrv_profiles = [r["hrv_profile"] for r in hrv_subjects]
        hrv_bands = list(HRV_BANDS.keys())

        print("\nPREDICTION 1 (HRV): A(τ) > 0")
        pred1_hrv = test_prediction_1(hrv_profiles, hrv_bands)
        print(pred1_hrv.to_string(index=False))
        pred1_hrv.to_csv(output_dir / "prediction_1_hrv.csv", index=False)

    if len(all_results) >= 5:
        print("\nPREDICTION 2: A(τ) ~ age")
        pred2 = test_prediction_2(eeg_profiles, ages, eeg_bands)
        print(pred2.to_string(index=False))
        pred2.to_csv(output_dir / "prediction_2_eeg.csv", index=False)

    # Summary JSON
    summary = {
        "n_subjects": len(all_results),
        "n_with_hrv": len(hrv_subjects),
        "ages": ages.tolist(),
        "eeg_bands": eeg_bands,
        "per_subject": [
            {
                "id": r["subject_id"],
                "age": r["age"],
                "sex": r["sex"],
                "eeg_A": {b: float(r["eeg_profile"][b]["A"]) for b in eeg_bands},
                "hrv_A": {b: float(r["hrv_profile"][b]["A"]) for b in hrv_bands}
                        if "hrv_profile" in r else None,
                "rho_cross": float(r.get("rho_cross_pearson", np.nan)),
            }
            for r in all_results
        ],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll results saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: LEMON Asymmetry Analysis")
    parser.add_argument("--data-dir", default="data/lemon/")
    parser.add_argument("--output", default="results/phase1/")
    parser.add_argument("--n-subjects", type=int, default=None)
    args = parser.parse_args()
    run(args.data_dir, args.output, args.n_subjects)


if __name__ == "__main__":
    main()
