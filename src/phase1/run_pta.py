"""
Phase 1: Pipeline v2 — Broadband PTA + spatial variability.

Key changes from run_fast.py:
1. Broadband PTA (Cole-Voytek): filter 1-45 Hz, find cycles, classify by frequency
2. σ(A) inter-channel variability as age-sensitive metric
3. Per-region analysis (frontal, central, temporal, parietal, occipital)

Usage:
    python3 -m src.phase1.run_pta --data-dir data/lemon/ --output results/phase1_pta/
"""

import argparse
import json
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from ..utils.asymmetry import (
    find_broadband_cycles, broadband_asymmetry_by_band,
    compute_spatial_asymmetry, EEG_REGIONS,
)


BAND_EDGES = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "low_gamma": (30, 45),
}


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


def extract_ecg_from_eeg(raw_eeg, sfreq):
    """Extract cardiac rhythm from EEG via ICA.

    Finds the ICA component most correlated with typical cardiac
    frequency range (1-2 Hz), reconstructs it as a pseudo-ECG signal.

    Returns None if extraction fails.
    """
    try:
        ica = mne.preprocessing.ICA(
            n_components=15, method="fastica", random_state=42,
            max_iter=300, verbose=False,
        )
        ica.fit(raw_eeg, verbose=False)

        # Get ICA source time courses
        sources = ica.get_sources(raw_eeg).get_data()

        # Find component with most power in cardiac range (0.8-2 Hz)
        from scipy.signal import welch
        best_idx = None
        best_cardiac_ratio = 0
        for i in range(sources.shape[0]):
            freqs, psd = welch(sources[i], fs=sfreq, nperseg=int(4 * sfreq))
            total = np.trapz(psd, freqs)
            cardiac_mask = (freqs >= 0.8) & (freqs <= 2.0)
            cardiac_power = np.trapz(psd[cardiac_mask], freqs[cardiac_mask])
            ratio = cardiac_power / (total + 1e-20)
            if ratio > best_cardiac_ratio:
                best_cardiac_ratio = ratio
                best_idx = i

        if best_idx is not None and best_cardiac_ratio > 0.15:
            ecg_proxy = sources[best_idx]
            print(f"ICA-ECG comp={best_idx} cardiac_ratio={best_cardiac_ratio:.2f}", end=" ", flush=True)
            return ecg_proxy
        else:
            print(f"no cardiac ICA (best ratio={best_cardiac_ratio:.2f})", end=" ", flush=True)
            return None
    except Exception as e:
        print(f"ICA-ECG failed: {e}", end=" ", flush=True)
        return None


def process_subject(vhdr_path, subject_id):
    """Process single subject: PTA + spatial + ECG/HRV."""
    print(f"  Loading...", end=" ", flush=True)
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    sfreq_orig = raw.info["sfreq"]

    # Drop non-EEG
    non_eeg = [ch for ch in raw.ch_names if ch in ("VEOG", "HEOG", "ECG", "EKG", "EMG", "AFz")]
    if non_eeg:
        raw.drop_channels(non_eeg)

    n_ch = len(raw.ch_names)
    dur = raw.times[-1]
    print(f"{n_ch}ch, {sfreq_orig:.0f}Hz, {dur:.0f}s", end=" ", flush=True)

    # Resample
    target_sfreq = 250.0
    if sfreq_orig > target_sfreq * 1.5:
        raw.resample(target_sfreq, verbose=False)
    sfreq = raw.info["sfreq"]

    # Minimal filtering — just notch for line noise
    raw.notch_filter(freqs=[50], fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj(verbose=False)
    print("preprocessed", flush=True)

    data = raw.get_data()  # (n_channels, n_samples)
    ch_names = list(raw.ch_names)

    # ── 1. Broadband PTA: per-channel + per-region ──
    print(f"  Broadband PTA ({n_ch} ch)...", end=" ", flush=True)
    per_channel, per_region, spatial_stats = compute_spatial_asymmetry(
        data, ch_names, sfreq, regions=EEG_REGIONS, f_low=1.0, f_high=45.0
    )
    print("done", flush=True)

    # ── 2. Build per-channel A matrix (raw, unfiltered) ──
    per_channel_A = {}
    for band in BAND_EDGES:
        per_channel_A[band] = {
            ch: float(per_channel[ch][band]["A_mean"])
            for ch in per_channel
            if not np.isnan(per_channel[ch][band]["A_mean"])
        }

    # ── 3. ECG/HRV via ICA ──
    hrv_result = None
    rho_cross = None
    print(f"  ECG extraction: ", end="", flush=True)
    ecg_proxy = extract_ecg_from_eeg(raw, sfreq)
    if ecg_proxy is not None:
        try:
            from .preprocess_ecg import (
                detect_r_peaks, compute_rr_intervals, rr_to_continuous,
                compute_acceleration_deceleration, HRV_BANDS,
            )
            from ..utils.asymmetry import compute_asymmetry_profile

            r_peaks = detect_r_peaks(ecg_proxy, sfreq, method="scipy")
            if len(r_peaks) >= 30:
                rr_times, rr_intervals = compute_rr_intervals(r_peaks, sfreq)
                hrv_signal, t_hrv = rr_to_continuous(rr_times, rr_intervals, 4.0)

                if len(hrv_signal) > 120:
                    hrv_profile = compute_asymmetry_profile(
                        hrv_signal, 4.0, HRV_BANDS, cycle_method="hilbert"
                    )
                    ac, dc, ratio = compute_acceleration_deceleration(rr_intervals)
                    hrv_result = {
                        "profile": {b: {"A": float(v["A"]), "n_cycles": v["n_cycles"]}
                                    for b, v in hrv_profile.items()},
                        "mean_hr": float(60.0 / np.mean(rr_intervals)),
                        "n_rr": len(rr_intervals),
                        "dc_ac_ratio": float(ratio),
                    }
                    print(f"HR={hrv_result['mean_hr']:.0f}bpm, {hrv_result['n_rr']} RR", flush=True)

                    # ρ_cross: correlate EEG A profile with HRV A profile
                    from .cross_system import compute_cross_coherence
                    # Build EEG grand profile in format expected by cross_coherence
                    eeg_grand_for_cc = {}
                    for band in BAND_EDGES:
                        tm = spatial_stats["trimmed_mean_A_per_band"].get(band, np.nan)
                        eeg_grand_for_cc[band] = {
                            "A": tm,
                            "center_freq": (BAND_EDGES[band][0] + BAND_EDGES[band][1]) / 2,
                        }
                    rho_p, rho_s, _, _, _, _, _ = compute_cross_coherence(
                        eeg_grand_for_cc, hrv_profile
                    )
                    rho_cross = {"pearson": float(rho_p), "spearman": float(rho_s)}
                    print(f"  ρ_cross={rho_p:.3f}", flush=True)
                else:
                    print("HRV signal too short", flush=True)
            else:
                print(f"only {len(r_peaks)} R-peaks", flush=True)
        except Exception as e:
            print(f"HRV analysis failed: {e}", flush=True)
    else:
        print("", flush=True)

    # ── Print summary ──
    ss = spatial_stats
    print(f"  Power-filtered per band:")
    for band in BAND_EDGES:
        n_inc = ss["n_included_per_band"].get(band, 0)
        tm = ss["trimmed_mean_A_per_band"].get(band, np.nan)
        pp = ss["prop_positive_per_band"].get(band, np.nan)
        print(f"    {band:10s}: tm={tm:+.4f}  pp={pp:.3f}  n={n_inc}")
    print(f"  σ(A)={ss['sigma_A']:.4f}")

    return {
        "subject_id": subject_id,
        "per_region": per_region,
        "spatial_stats": spatial_stats,
        "per_channel_A": per_channel_A,
        "n_channels": n_ch,
        "n_valid_per_band": dict(ss["n_included_per_band"]),
        "duration_s": dur,
        "sfreq": sfreq,
        "hrv": hrv_result,
        "rho_cross": rho_cross,
    }


def run(data_dir, output_dir, n_subjects=None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants = load_participants(data_dir)
    available = []
    for _, row in participants.iterrows():
        vhdr = data_dir / row["subject_id"] / "RSEEG" / f"{row['subject_id']}.vhdr"
        if vhdr.exists():
            available.append((row, vhdr))
    print(f"Available: {len(available)}")
    if n_subjects:
        available = available[:n_subjects]

    results = []
    for row, vhdr in available:
        print(f"\n{'='*60}")
        print(f"{row['subject_id']} | age={row['age_range']} ({row['age']}) | sex={row['sex']}")
        print(f"{'='*60}")
        try:
            r = process_subject(vhdr, row["subject_id"])
            r["age"] = float(row["age"])
            r["sex"] = row["sex"]
            r["age_range"] = row["age_range"]
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print(f"\n\n{'#'*60}")
    print(f"PROCESSED: {len(results)} / {len(available)}")
    print(f"{'#'*60}")

    if not results:
        return

    with open(output_dir / "all_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # ── Analysis ──
    bands = list(BAND_EDGES.keys())
    ages = np.array([r["age"] for r in results])
    ss_key = "spatial_stats"

    # 1. Prediction 1: trimmed_mean A > 0 (power-filtered channels)
    print(f"\n{'='*60}")
    print("PREDICTION 1: A(τ) > 0 [power-filtered, trimmed mean]")
    print(f"{'='*60}")
    from scipy.stats import ttest_1samp
    for band in bands:
        vals = [r[ss_key]["trimmed_mean_A_per_band"].get(band, np.nan) for r in results]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) >= 3:
            t, p = ttest_1samp(vals, 0)
            p_one = p / 2 if t > 0 else 1 - p / 2
            print(f"  {band:12s}: tm_A={np.mean(vals):+.4f} ± {np.std(vals):.4f}  "
                  f"t={t:.2f}  p={p_one:.4e}  {'*' if p_one < 0.05 else ''}")

    # 2. prop_positive table
    print(f"\n{'='*60}")
    print("PROP_POSITIVE (fraction of power-filtered channels with A > 0)")
    print(f"{'='*60}")
    header = f"  {'subject':>12s} {'age':>4s} " + " ".join(f"{b:>12s}" for b in bands)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        pp = r[ss_key]["prop_positive_per_band"]
        vals_str = " ".join(f"{pp.get(b, np.nan):>12.3f}" for b in bands)
        print(f"  {r['subject_id']:>12s} {r['age']:>4.0f} {vals_str}")

    # 3. σ(A) per subject
    print(f"\n{'='*60}")
    print("σ(A) SPATIAL VARIABILITY")
    print(f"{'='*60}")
    header = f"  {'subject':>12s} {'age':>4s} {'σ_global':>8s} " + " ".join(f"{b:>12s}" for b in bands)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        sg = r[ss_key]["sigma_A"]
        sb = r[ss_key]["sigma_A_per_band"]
        vals_str = " ".join(f"{sb.get(b, np.nan):>12.4f}" for b in bands)
        print(f"  {r['subject_id']:>12s} {r['age']:>4.0f} {sg:>8.4f} {vals_str}")

    # 4. Per-region
    print(f"\n{'='*60}")
    print("PER-REGION (power-filtered): A_mean and prop_positive")
    print(f"{'='*60}")
    for r in results:
        parts = []
        for reg in ["frontal", "central", "temporal", "parietal", "occipital"]:
            if reg not in r["per_region"]:
                continue
            bd = r["per_region"][reg]["per_band"]
            for b in ["alpha", "beta"]:
                if b in bd:
                    a = bd[b].get("A_mean", np.nan)
                    pp = bd[b].get("prop_positive", np.nan)
                    n = bd[b].get("n_channels", 0)
                    if not np.isnan(a):
                        parts.append(f"{reg[:3]}.{b[:2]}={a:+.3f}(pp={pp:.2f},n={n})")
        print(f"  {r['subject_id']} ({r['age']:.0f}): {' | '.join(parts)}")

    # 5. Age correlations
    if len(results) >= 5:
        print(f"\n{'='*60}")
        print("AGE CORRELATIONS (N={})".format(len(results)))
        print(f"{'='*60}")
        from scipy.stats import pearsonr
        for band in bands:
            for metric, key in [("trimmed_mean", "trimmed_mean_A_per_band"),
                                ("prop_positive", "prop_positive_per_band"),
                                ("σ_A", "sigma_A_per_band")]:
                vals = np.array([r[ss_key][key].get(band, np.nan) for r in results])
                valid = ~np.isnan(vals)
                if valid.sum() >= 4:
                    rv, pv = pearsonr(ages[valid], vals[valid])
                    sig = "*" if pv < 0.05 else ""
                    print(f"  {metric:14s}({band:10s}) ~ age: r={rv:+.3f}  p={pv:.4f} {sig}")

        # Global σ
        sigma_global = np.array([r[ss_key]["sigma_A"] for r in results])
        valid = ~np.isnan(sigma_global)
        if valid.sum() >= 4:
            rv, pv = pearsonr(ages[valid], sigma_global[valid])
            print(f"  {'σ_global':14s}{'':10s}  ~ age: r={rv:+.3f}  p={pv:.4f} {'*' if pv < 0.05 else ''}")

        # n_valid per band ~ age
        for band in bands:
            nv = np.array([r["n_valid_per_band"].get(band, 0) for r in results], dtype=float)
            if np.std(nv) > 0:
                rv, pv = pearsonr(ages, nv)
                print(f"  {'n_valid':14s}({band:10s}) ~ age: r={rv:+.3f}  p={pv:.4f} {'*' if pv < 0.05 else ''}")

        # ρ_cross ~ age (Prediction 3)
        hrv_subs = [r for r in results if r.get("rho_cross") is not None]
        if len(hrv_subs) >= 4:
            rho_vals = np.array([r["rho_cross"]["pearson"] for r in hrv_subs])
            hrv_ages = np.array([r["age"] for r in hrv_subs])
            rv, pv = pearsonr(hrv_ages, rho_vals)
            print(f"\n  PREDICTION 3: ρ_cross ~ age (N={len(hrv_subs)})")
            print(f"    r={rv:+.3f}  p={pv:.4f} {'*' if pv < 0.05 else ''}")

    # 6. HRV summary
    hrv_subs = [r for r in results if r.get("hrv") is not None]
    if hrv_subs:
        print(f"\n{'='*60}")
        print(f"HRV RESULTS ({len(hrv_subs)}/{len(results)} subjects with ECG)")
        print(f"{'='*60}")
        for r in hrv_subs:
            h = r["hrv"]
            rho = r.get("rho_cross", {})
            rho_str = f"ρ={rho.get('pearson', np.nan):.3f}" if rho else "ρ=N/A"
            print(f"  {r['subject_id']} ({r['age']:.0f}): HR={h['mean_hr']:.0f}bpm  "
                  f"DC/AC={h['dc_ac_ratio']:.2f}  {rho_str}")

    # 7. n_valid per band table
    print(f"\n{'='*60}")
    print("N_VALID CHANNELS PER BAND (power-threshold + min_cycles)")
    print(f"{'='*60}")
    header = f"  {'subject':>12s} {'age':>4s} " + " ".join(f"{b:>10s}" for b in bands)
    print(header)
    for r in results:
        nv = r["n_valid_per_band"]
        vals_str = " ".join(f"{nv.get(b, 0):>10d}" for b in bands)
        print(f"  {r['subject_id']:>12s} {r['age']:>4.0f} {vals_str}")

    # Save
    def safe_float(x):
        if isinstance(x, float) and np.isnan(x):
            return None
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.integer):
            return int(x)
        return x

    summary = {
        "n_subjects": len(results),
        "n_with_hrv": len(hrv_subs),
        "bands": bands,
        "subjects": [
            {
                "id": r["subject_id"], "age": r["age"], "sex": r["sex"],
                "n_valid_per_band": r["n_valid_per_band"],
                **{f"{m}_{b}": r[ss_key][f"{m}_per_band"].get(b)
                   for m in ["trimmed_mean_A", "prop_positive", "sigma_A", "mean_A"]
                   for b in bands},
                "sigma_global": r[ss_key]["sigma_A"],
                "per_channel_A": r["per_channel_A"],
                "hrv": r.get("hrv"),
                "rho_cross": r.get("rho_cross"),
                "regions": {
                    reg: {b: r["per_region"][reg]["per_band"].get(b, {})
                          for b in bands}
                    for reg in r["per_region"]
                },
            }
            for r in results
        ],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=safe_float)

    print(f"\nSaved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/lemon/")
    parser.add_argument("--output", default="results/phase1_pta/")
    parser.add_argument("--n-subjects", type=int, default=None)
    args = parser.parse_args()
    run(args.data_dir, args.output, args.n_subjects)


if __name__ == "__main__":
    main()
