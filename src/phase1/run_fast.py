"""
Phase 1: Fast pipeline — no ICA, resample early, focus on asymmetry computation.

Usage:
    python3 -m src.phase1.run_fast --data-dir data/lemon/ --output results/phase1/
"""

import argparse
import json
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from ..utils.asymmetry import compute_asymmetry_profile, surrogate_test
from .preprocess_ecg import HRV_BANDS
from .cross_system import compute_cross_coherence
from .statistics import test_prediction_1, test_prediction_2

# EEG frequency bands
EEG_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 50),
}
# Skip high_gamma — 50-80 Hz too noisy without ICA


def load_participants(data_dir):
    csv_path = Path(data_dir) / "Participants_MPILMBB_LEMON.csv"
    df = pd.read_csv(csv_path)
    df.columns = ["subject_id", "sex", "age_range"]
    df["age"] = df["age_range"].apply(
        lambda s: (int(s.split("-")[0]) + int(s.split("-")[1])) / 2
        if "-" in str(s) else np.nan
    )
    df["sex"] = df["sex"].map({1: "F", 2: "M"})
    return df


def process_subject(vhdr_path, subject_id):
    """Fast single-subject processing: load → resample → filter → asymmetry."""
    print(f"  Loading {vhdr_path.name}...", end=" ", flush=True)

    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    sfreq_orig = raw.info["sfreq"]

    # Drop non-EEG channels
    non_eeg = [ch for ch in raw.ch_names
               if ch in ("VEOG", "HEOG", "ECG", "EKG", "EMG")]
    if non_eeg:
        raw.drop_channels(non_eeg)

    n_ch = len(raw.ch_names)
    duration = raw.times[-1]
    print(f"{n_ch} ch, {sfreq_orig}Hz, {duration:.0f}s", flush=True)

    # Resample FIRST to speed everything up
    target_sfreq = 250.0
    if sfreq_orig > target_sfreq * 1.5:
        print(f"  Resampling {sfreq_orig:.0f} → {target_sfreq:.0f} Hz...", end=" ", flush=True)
        raw.resample(target_sfreq, verbose=False)
        print("done", flush=True)

    sfreq = raw.info["sfreq"]

    # Basic filtering
    print(f"  Filtering...", end=" ", flush=True)
    raw.filter(l_freq=0.5, h_freq=50.0, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=[50], fir_design="firwin", verbose=False)
    print("done", flush=True)

    # Average reference
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj(verbose=False)

    # Compute asymmetry for each channel
    print(f"  Computing A(τ) across {n_ch} channels...", end=" ", flush=True)
    data = raw.get_data()

    channel_A = {}
    for i, ch in enumerate(raw.ch_names):
        try:
            profile = compute_asymmetry_profile(data[i], sfreq, EEG_BANDS,
                                                 method="bandpass")
            channel_A[ch] = profile
        except Exception:
            pass

    # Grand average across channels
    eeg_grand = {}
    for band in EEG_BANDS:
        vals = [channel_A[ch][band]["A"] for ch in channel_A
                if not np.isnan(channel_A[ch][band]["A"])]
        eeg_grand[band] = {
            "A": np.median(vals) if vals else np.nan,
            "A_mean": np.mean(vals) if vals else np.nan,
            "A_std": np.std(vals) if vals else np.nan,
            "n_channels": len(vals),
            "center_freq": (EEG_BANDS[band][0] + EEG_BANDS[band][1]) / 2,
        }

    print("done", flush=True)
    print(f"  A(τ): " + ", ".join(
        f"{b}={v['A']:.3f}" for b, v in eeg_grand.items()
        if not np.isnan(v['A'])
    ))

    # Surrogate test on Cz
    surr_result = None
    rep_ch = "Cz" if "Cz" in channel_A else list(channel_A.keys())[0]
    rep_idx = raw.ch_names.index(rep_ch)
    print(f"  Surrogate test ({rep_ch}, 200 surrogates)...", end=" ", flush=True)
    try:
        surr = surrogate_test(data[rep_idx], sfreq, EEG_BANDS, n_surrogates=200)
        surr_result = {
            band: {"p": v["p_value"], "z": v["z_score"]}
            for band, v in surr.items()
        }
        sig = sum(1 for v in surr.values() if v["p_value"] < 0.05)
        print(f"{sig}/{len(EEG_BANDS)} significant", flush=True)
    except Exception as e:
        print(f"failed: {e}", flush=True)

    return {
        "subject_id": subject_id,
        "eeg_profile": eeg_grand,
        "n_channels": n_ch,
        "sfreq": sfreq,
        "duration_s": duration,
        "surrogate": surr_result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/lemon/")
    parser.add_argument("--output", default="results/phase1/")
    parser.add_argument("--n-subjects", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load participants
    participants = load_participants(data_dir)
    print(f"Participants in CSV: {len(participants)}")

    # Find available EEG
    available = []
    for _, row in participants.iterrows():
        vhdr = data_dir / row["subject_id"] / "RSEEG" / f"{row['subject_id']}.vhdr"
        if vhdr.exists():
            available.append((row, vhdr))

    print(f"Available EEG: {len(available)}")
    if args.n_subjects:
        available = available[:args.n_subjects]

    # Process
    results = []
    for row, vhdr in available:
        print(f"\n{'='*60}")
        print(f"{row['subject_id']} | age={row['age_range']} ({row['age']}) | sex={row['sex']}")
        print(f"{'='*60}")
        try:
            r = process_subject(vhdr, row["subject_id"])
            r["age"] = row["age"]
            r["sex"] = row["sex"]
            r["age_range"] = row["age_range"]
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print(f"\n\n{'#'*60}")
    print(f"PROCESSED: {len(results)} / {len(available)}")
    print(f"{'#'*60}")

    if len(results) == 0:
        print("No results!")
        return

    # Save
    with open(output_dir / "all_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # ── Statistics ──
    eeg_bands = list(EEG_BANDS.keys())
    ages = np.array([r["age"] for r in results])
    eeg_profiles = [r["eeg_profile"] for r in results]

    print(f"\n{'='*60}")
    print("PREDICTION 1: A(τ) > 0 at all EEG scales")
    print(f"{'='*60}")
    pred1 = test_prediction_1(eeg_profiles, eeg_bands)
    print(pred1.to_string(index=False))
    pred1.to_csv(output_dir / "prediction_1_eeg.csv", index=False)

    if len(results) >= 5:
        print(f"\n{'='*60}")
        print("PREDICTION 2: A(τ) ~ age")
        print(f"{'='*60}")
        pred2 = test_prediction_2(eeg_profiles, ages, eeg_bands)
        print(pred2.to_string(index=False))
        pred2.to_csv(output_dir / "prediction_2_eeg.csv", index=False)

    # Summary per subject
    print(f"\n{'='*60}")
    print("PER-SUBJECT SUMMARY")
    print(f"{'='*60}")
    for r in results:
        bands_str = " | ".join(
            f"{b}={r['eeg_profile'][b]['A']:+.3f}"
            for b in eeg_bands if not np.isnan(r['eeg_profile'][b]['A'])
        )
        print(f"  {r['subject_id']} (age={r['age']:.0f}): {bands_str}")

    # Save summary JSON
    summary = {
        "n_subjects": len(results),
        "eeg_bands": eeg_bands,
        "subjects": [
            {
                "id": r["subject_id"],
                "age": float(r["age"]),
                "sex": r["sex"],
                "A": {b: float(r["eeg_profile"][b]["A"]) for b in eeg_bands},
            }
            for r in results
        ],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
