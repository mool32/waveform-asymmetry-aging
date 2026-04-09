"""
EC vs EO analysis: H(sign A) in alpha band.

Tests:
1. H(EO) < H(EC)? (visual input constrains posterior alpha -> lower entropy)
2. DeltaH = H(EO) - H(EC) ~ age? (older subjects have smaller |DeltaH|)

Marker scheme in LEMON .vmrk files:
  S  1  = block boundary
  S210  = eyes-closed (EC) tick markers (every ~5s within EC block)
  S200  = eyes-open (EO) tick markers (every ~5s within EO block)

We extract EC segments = all data during S210 tick periods,
and EO segments = all data during S200 tick periods.

For each condition:
  1. Broadband filter 1-45 Hz, find cycles, classify alpha (8-13 Hz)
  2. Per channel: PTA = rise_time / (rise_time + fall_time)
  3. Across channels: A_sign = PTA > 0.5 (i.e. rise fraction > 0.5)
     p_pos = fraction of channels with A > 0
     H = -p*log2(p) - (1-p)*log2(1-p)   (binary entropy)

Usage:
    python3 -m src.phase1.run_eceo_entropy --data-dir data/lemon/ --output results/eceo_entropy/
"""

import argparse
import json
import traceback
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("ERROR")


# ---------------------------------------------------------------------------
# Marker parsing
# ---------------------------------------------------------------------------

def parse_vmrk(vmrk_path):
    """Parse BrainVision .vmrk file, return list of (description, position)."""
    markers = []
    with open(vmrk_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Mk") and "=" in line:
                parts = line.split("=", 1)[1].split(",")
                if len(parts) >= 3:
                    desc = parts[1].strip()
                    try:
                        pos = int(parts[2].strip())
                    except ValueError:
                        continue
                    markers.append((desc, pos))
    return markers


def get_ec_eo_segments(markers, n_samples, sfreq):
    """Extract EC and EO time segments from LEMON markers.

    Strategy:
    - S210 markers indicate eyes-closed ticks (~5s apart)
    - S200 markers indicate eyes-open ticks (~5s apart)
    - Each block starts with S  1 followed by a run of S210 or S200

    We collect segments: for each consecutive pair of same-type ticks,
    the segment is [tick_i, tick_{i+1}].  This gives us clean within-block data.

    Returns (ec_segments, eo_segments) as lists of (start_sample, end_sample).
    """
    # Collect positions of EC and EO tick markers
    ec_positions = sorted([pos for desc, pos in markers if desc == "S210"])
    eo_positions = sorted([pos for desc, pos in markers if desc == "S200"])

    def positions_to_segments(positions, max_gap_samples):
        """Convert tick positions to segments between consecutive ticks."""
        segments = []
        for i in range(len(positions) - 1):
            start = positions[i]
            end = positions[i + 1]
            # Only include if gap is reasonable (< max_gap to avoid cross-block)
            if end - start <= max_gap_samples and end <= n_samples:
                segments.append((start, end))
        return segments

    # Max gap: 6 seconds worth of samples (ticks are ~5s apart)
    max_gap = int(6.0 * sfreq)

    ec_segments = positions_to_segments(ec_positions, max_gap)
    eo_segments = positions_to_segments(eo_positions, max_gap)

    return ec_segments, eo_segments


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def binary_entropy(p):
    """H = -p*log2(p) - (1-p)*log2(1-p), handles edge cases."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def compute_alpha_pta_per_channel(data_segment, sfreq, ch_names,
                                   f_low=1.0, f_high=45.0,
                                   alpha_lo=8.0, alpha_hi=13.0,
                                   min_cycles=10):
    """Compute broadband PTA for alpha cycles per channel.

    Returns dict: ch_name -> A_mean (asymmetry index for alpha cycles)
    Only channels with >= min_cycles alpha cycles are returned.
    """
    from scipy.signal import butter, filtfilt, find_peaks

    nyq = sfreq / 2.0
    b, a = butter(2, [f_low / nyq, min(f_high / nyq, 0.99)], btype="band")

    channel_A = {}
    for i, ch in enumerate(ch_names):
        sig = data_segment[i]
        if len(sig) < int(sfreq * 0.5):
            continue

        try:
            filtered = filtfilt(b, a, sig)
        except Exception:
            continue

        std = np.std(filtered)
        if std < 1e-10:
            continue

        peaks, _ = find_peaks(filtered, prominence=0.1 * std,
                              distance=int(sfreq / f_high / 2))
        troughs, _ = find_peaks(-filtered, prominence=0.1 * std,
                                distance=int(sfreq / f_high / 2))

        if len(peaks) < 1 or len(troughs) < 2:
            continue

        # Build trough-peak-trough cycles, classify by frequency
        alpha_A_vals = []
        for pi in range(len(peaks)):
            peak_idx = peaks[pi]
            pre = troughs[troughs < peak_idx]
            post = troughs[troughs > peak_idx]
            if len(pre) == 0 or len(post) == 0:
                continue
            t1 = pre[-1]
            t2 = post[0]
            t_rise = peak_idx - t1
            t_fall = t2 - peak_idx
            t_total = t_rise + t_fall
            if t_rise < 2 or t_fall < 2:
                continue
            freq = sfreq / t_total
            if alpha_lo <= freq < alpha_hi:
                A = (t_fall - t_rise) / t_total
                alpha_A_vals.append(A)

        if len(alpha_A_vals) >= min_cycles:
            channel_A[ch] = np.mean(alpha_A_vals)

    return channel_A


def compute_h_sign_a(channel_A_dict, min_channels=10):
    """Compute H(sign A) from per-channel asymmetry values.

    p_pos = fraction of channels with A > 0
    H = binary_entropy(p_pos)

    Returns (H, p_pos, n_channels) or (nan, nan, n) if too few channels.
    """
    vals = list(channel_A_dict.values())
    n = len(vals)
    if n < min_channels:
        return np.nan, np.nan, n

    p_pos = np.mean(np.array(vals) > 0)
    H = binary_entropy(p_pos)
    return H, p_pos, n


# ---------------------------------------------------------------------------
# Single-subject processing
# ---------------------------------------------------------------------------

def process_subject(args):
    """Process one subject. args = (vhdr_path, subject_id, age, sex, age_range)."""
    vhdr_path, subject_id, age, sex, age_range = args

    try:
        # Load
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
        sfreq_orig = raw.info["sfreq"]
        n_samples_orig = raw.n_times

        # Parse markers from .vmrk
        vmrk_path = str(vhdr_path).replace(".vhdr", ".vmrk")
        markers = parse_vmrk(vmrk_path)

        # Drop non-EEG channels
        non_eeg = [ch for ch in raw.ch_names if ch in ("VEOG", "HEOG", "ECG", "EKG", "EMG", "AFz")]
        if non_eeg:
            raw.drop_channels(non_eeg)

        n_ch = len(raw.ch_names)

        # Resample
        target_sfreq = 250.0
        if sfreq_orig > target_sfreq * 1.5:
            # Scale marker positions
            scale_factor = target_sfreq / sfreq_orig
            markers = [(desc, int(pos * scale_factor)) for desc, pos in markers]
            raw.resample(target_sfreq, verbose=False)

        sfreq = raw.info["sfreq"]
        n_samples = raw.n_times

        # Minimal preprocessing
        raw.notch_filter(freqs=[50], fir_design="firwin", verbose=False)
        raw.set_eeg_reference("average", projection=True, verbose=False)
        raw.apply_proj(verbose=False)

        data = raw.get_data()  # (n_channels, n_samples)
        ch_names = list(raw.ch_names)

        # Get EC/EO segments
        ec_segs, eo_segs = get_ec_eo_segments(markers, n_samples, sfreq)

        if len(ec_segs) < 3 or len(eo_segs) < 3:
            return {"subject_id": subject_id, "error": f"too few segments EC={len(ec_segs)} EO={len(eo_segs)}"}

        # Concatenate segments for each condition
        def concat_segments(segments):
            pieces = []
            for s, e in segments:
                if e <= data.shape[1]:
                    pieces.append(data[:, s:e])
            if pieces:
                return np.concatenate(pieces, axis=1)
            return None

        ec_data = concat_segments(ec_segs)
        eo_data = concat_segments(eo_segs)

        if ec_data is None or eo_data is None:
            return {"subject_id": subject_id, "error": "no valid segment data"}

        ec_dur = ec_data.shape[1] / sfreq
        eo_dur = eo_data.shape[1] / sfreq

        # Compute alpha PTA per channel for each condition
        ec_channel_A = compute_alpha_pta_per_channel(ec_data, sfreq, ch_names)
        eo_channel_A = compute_alpha_pta_per_channel(eo_data, sfreq, ch_names)

        # Compute H(sign A)
        H_ec, p_pos_ec, n_ch_ec = compute_h_sign_a(ec_channel_A)
        H_eo, p_pos_eo, n_ch_eo = compute_h_sign_a(eo_channel_A)

        # Also compute mean A across channels for each condition
        mean_A_ec = np.mean(list(ec_channel_A.values())) if ec_channel_A else np.nan
        mean_A_eo = np.mean(list(eo_channel_A.values())) if eo_channel_A else np.nan

        delta_H = H_eo - H_ec if not (np.isnan(H_eo) or np.isnan(H_ec)) else np.nan

        return {
            "subject_id": subject_id,
            "age": age,
            "sex": sex,
            "age_range": age_range,
            "H_ec": H_ec,
            "H_eo": H_eo,
            "delta_H": delta_H,
            "p_pos_ec": p_pos_ec,
            "p_pos_eo": p_pos_eo,
            "n_ch_ec": n_ch_ec,
            "n_ch_eo": n_ch_eo,
            "mean_A_ec": mean_A_ec,
            "mean_A_eo": mean_A_eo,
            "ec_dur_s": ec_dur,
            "eo_dur_s": eo_dur,
            "n_ec_segs": len(ec_segs),
            "n_eo_segs": len(eo_segs),
            "n_channels": n_ch,
        }

    except Exception as e:
        return {"subject_id": subject_id, "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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


def run(data_dir, output_dir, n_subjects=None, n_workers=None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants = load_participants(data_dir)

    # Find available subjects
    jobs = []
    for _, row in participants.iterrows():
        vhdr = data_dir / row["subject_id"] / "RSEEG" / f"{row['subject_id']}.vhdr"
        if vhdr.exists():
            jobs.append((
                str(vhdr), row["subject_id"],
                float(row["age"]), row["sex"], row["age_range"]
            ))

    print(f"Available subjects: {len(jobs)}")
    if n_subjects:
        jobs = jobs[:n_subjects]
    print(f"Processing: {len(jobs)}")

    # Process with multiprocessing
    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    print(f"Using {n_workers} workers")

    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(process_subject, jobs)):
                if "error" in r:
                    print(f"  [{i+1}/{len(jobs)}] {r['subject_id']}: FAILED - {r['error']}")
                else:
                    print(f"  [{i+1}/{len(jobs)}] {r['subject_id']}: "
                          f"H_ec={r['H_ec']:.3f} H_eo={r['H_eo']:.3f} "
                          f"dH={r['delta_H']:+.3f} "
                          f"(n_ch={r['n_ch_ec']}/{r['n_ch_eo']})")
                results.append(r)
    else:
        results = []
        for i, job in enumerate(jobs):
            r = process_subject(job)
            if "error" in r:
                print(f"  [{i+1}/{len(jobs)}] {r['subject_id']}: FAILED - {r['error']}")
            else:
                print(f"  [{i+1}/{len(jobs)}] {r['subject_id']}: "
                      f"H_ec={r['H_ec']:.3f} H_eo={r['H_eo']:.3f} "
                      f"dH={r['delta_H']:+.3f}")
            results.append(r)

    # Filter successful results
    valid = [r for r in results if "error" not in r and not np.isnan(r.get("delta_H", np.nan))]
    failed = [r for r in results if "error" in r]
    excluded = [r for r in results if "error" not in r and np.isnan(r.get("delta_H", np.nan))]

    print(f"\n{'='*70}")
    print(f"RESULTS: {len(valid)} valid / {len(excluded)} excluded / {len(failed)} failed "
          f"(of {len(results)} total)")
    print(f"{'='*70}")

    if len(valid) < 5:
        print("Too few valid subjects for statistics.")
        return

    # Build arrays
    ages = np.array([r["age"] for r in valid])
    H_ec = np.array([r["H_ec"] for r in valid])
    H_eo = np.array([r["H_eo"] for r in valid])
    delta_H = np.array([r["delta_H"] for r in valid])
    p_pos_ec = np.array([r["p_pos_ec"] for r in valid])
    p_pos_eo = np.array([r["p_pos_eo"] for r in valid])
    mean_A_ec = np.array([r["mean_A_ec"] for r in valid])
    mean_A_eo = np.array([r["mean_A_eo"] for r in valid])

    # ================================================================
    # TEST 1: Descriptive statistics
    # ================================================================
    print(f"\n{'='*70}")
    print("DESCRIPTIVE STATISTICS")
    print(f"{'='*70}")
    print(f"  N subjects: {len(valid)}")
    print(f"  Age range:  {ages.min():.0f} - {ages.max():.0f} (mean={ages.mean():.1f}, sd={ages.std():.1f})")
    print(f"")
    print(f"  H(sign A) Eyes-Closed: mean={H_ec.mean():.4f} sd={H_ec.std():.4f} "
          f"median={np.median(H_ec):.4f}")
    print(f"  H(sign A) Eyes-Open:   mean={H_eo.mean():.4f} sd={H_eo.std():.4f} "
          f"median={np.median(H_eo):.4f}")
    print(f"  DeltaH (EO-EC):        mean={delta_H.mean():.4f} sd={delta_H.std():.4f} "
          f"median={np.median(delta_H):.4f}")
    print(f"")
    print(f"  p_pos EC: mean={p_pos_ec.mean():.4f} sd={p_pos_ec.std():.4f}")
    print(f"  p_pos EO: mean={p_pos_eo.mean():.4f} sd={p_pos_eo.std():.4f}")
    print(f"")
    print(f"  Mean A(alpha) EC: mean={mean_A_ec.mean():.4f} sd={mean_A_ec.std():.4f}")
    print(f"  Mean A(alpha) EO: mean={mean_A_eo.mean():.4f} sd={mean_A_eo.std():.4f}")

    # ================================================================
    # TEST 2: H(EO) < H(EC)? Paired test
    # ================================================================
    from scipy.stats import ttest_rel, wilcoxon, pearsonr, spearmanr

    print(f"\n{'='*70}")
    print("TEST 1: H(EO) < H(EC)?  (prediction: visual input constrains alpha)")
    print(f"{'='*70}")

    t_stat, p_two = ttest_rel(H_eo, H_ec)
    p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2  # one-sided: EO < EC
    cohen_d = (H_eo.mean() - H_ec.mean()) / np.sqrt((H_eo.std()**2 + H_ec.std()**2) / 2)

    print(f"  Paired t-test: t({len(valid)-1}) = {t_stat:.4f}, "
          f"p(two-sided) = {p_two:.4e}, p(one-sided EO<EC) = {p_one:.4e}")
    print(f"  Cohen's d = {cohen_d:.4f}")
    print(f"  Mean difference (EO-EC) = {(H_eo - H_ec).mean():.4f}")

    try:
        w_stat, p_wilcox = wilcoxon(H_eo, H_ec, alternative="less")
        print(f"  Wilcoxon signed-rank: W = {w_stat:.1f}, p(EO<EC) = {p_wilcox:.4e}")
    except Exception as e:
        print(f"  Wilcoxon failed: {e}")

    # Also test: mean_A differs between conditions
    print(f"\n  --- Supplementary: Mean A(alpha) EC vs EO ---")
    t_a, p_a = ttest_rel(mean_A_ec, mean_A_eo)
    d_a = (mean_A_ec.mean() - mean_A_eo.mean()) / np.sqrt((mean_A_ec.std()**2 + mean_A_eo.std()**2) / 2)
    print(f"  Paired t-test on mean A: t = {t_a:.4f}, p = {p_a:.4e}, d = {d_a:.4f}")

    # ================================================================
    # TEST 3: DeltaH ~ age
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 2: DeltaH ~ age  (prediction: older -> smaller |DeltaH|)")
    print(f"{'='*70}")

    # Pearson correlation: delta_H ~ age
    r_dh, p_dh = pearsonr(ages, delta_H)
    rho_dh, p_rho_dh = spearmanr(ages, delta_H)
    print(f"  DeltaH ~ age:")
    print(f"    Pearson:  r = {r_dh:+.4f}, p = {p_dh:.4e}")
    print(f"    Spearman: rho = {rho_dh:+.4f}, p = {p_rho_dh:.4e}")

    # |DeltaH| ~ age (prediction: negative correlation)
    abs_dh = np.abs(delta_H)
    r_abs, p_abs = pearsonr(ages, abs_dh)
    rho_abs, p_rho_abs = spearmanr(ages, abs_dh)
    print(f"  |DeltaH| ~ age:")
    print(f"    Pearson:  r = {r_abs:+.4f}, p = {p_abs:.4e}")
    print(f"    Spearman: rho = {rho_abs:+.4f}, p = {p_rho_abs:.4e}")

    # H_ec ~ age, H_eo ~ age
    r_ec, p_ec = pearsonr(ages, H_ec)
    r_eo, p_eo = pearsonr(ages, H_eo)
    print(f"  H_ec ~ age:    r = {r_ec:+.4f}, p = {p_ec:.4e}")
    print(f"  H_eo ~ age:    r = {r_eo:+.4f}, p = {p_eo:.4e}")

    # mean_A ~ age per condition
    r_aec, p_aec = pearsonr(ages, mean_A_ec)
    r_aeo, p_aeo = pearsonr(ages, mean_A_eo)
    print(f"  mean_A_ec ~ age: r = {r_aec:+.4f}, p = {p_aec:.4e}")
    print(f"  mean_A_eo ~ age: r = {r_aeo:+.4f}, p = {p_aeo:.4e}")

    # ================================================================
    # TEST 4: Linear regression DeltaH ~ age (with effect size)
    # ================================================================
    from scipy.stats import linregress

    print(f"\n{'='*70}")
    print("LINEAR REGRESSION: DeltaH ~ age")
    print(f"{'='*70}")

    slope, intercept, r_val, p_val, se = linregress(ages, delta_H)
    print(f"  DeltaH = {slope:.6f} * age + {intercept:.4f}")
    print(f"  R^2 = {r_val**2:.4f}, p = {p_val:.4e}, SE(slope) = {se:.6f}")

    slope2, intercept2, r2, p2, se2 = linregress(ages, abs_dh)
    print(f"  |DeltaH| = {slope2:.6f} * age + {intercept2:.4f}")
    print(f"  R^2 = {r2**2:.4f}, p = {p2:.4e}, SE(slope) = {se2:.6f}")

    # ================================================================
    # Age-group analysis
    # ================================================================
    print(f"\n{'='*70}")
    print("AGE GROUP ANALYSIS")
    print(f"{'='*70}")

    young = np.array([r for r in valid if r["age"] <= 35])
    old = np.array([r for r in valid if r["age"] >= 55])

    if len(young) >= 5 and len(old) >= 5:
        from scipy.stats import ttest_ind, mannwhitneyu

        y_dh = np.array([r["delta_H"] for r in young])
        o_dh = np.array([r["delta_H"] for r in old])
        y_hec = np.array([r["H_ec"] for r in young])
        o_hec = np.array([r["H_ec"] for r in old])
        y_heo = np.array([r["H_eo"] for r in young])
        o_heo = np.array([r["H_eo"] for r in old])
        y_abs = np.abs(y_dh)
        o_abs = np.abs(o_dh)

        print(f"  Young (age<=35): N={len(young)}, DeltaH mean={y_dh.mean():.4f} sd={y_dh.std():.4f}")
        print(f"  Old   (age>=55): N={len(old)},   DeltaH mean={o_dh.mean():.4f} sd={o_dh.std():.4f}")

        t_grp, p_grp = ttest_ind(y_dh, o_dh)
        d_grp = (y_dh.mean() - o_dh.mean()) / np.sqrt((y_dh.std()**2 + o_dh.std()**2) / 2)
        print(f"  DeltaH young vs old: t = {t_grp:.4f}, p = {p_grp:.4e}, d = {d_grp:.4f}")

        t_abs, p_abs_grp = ttest_ind(y_abs, o_abs)
        d_abs = (y_abs.mean() - o_abs.mean()) / np.sqrt((y_abs.std()**2 + o_abs.std()**2) / 2)
        print(f"  |DeltaH| young vs old: t = {t_abs:.4f}, p = {p_abs_grp:.4e}, d = {d_abs:.4f}")

        # H per condition per group
        print(f"\n  Young: H_ec={y_hec.mean():.4f}  H_eo={y_heo.mean():.4f}")
        print(f"  Old:   H_ec={o_hec.mean():.4f}  H_eo={o_heo.mean():.4f}")

        t_ec_grp, p_ec_grp = ttest_ind(y_hec, o_hec)
        t_eo_grp, p_eo_grp = ttest_ind(y_heo, o_heo)
        print(f"  H_ec young vs old: t={t_ec_grp:.4f}, p={p_ec_grp:.4e}")
        print(f"  H_eo young vs old: t={t_eo_grp:.4f}, p={p_eo_grp:.4e}")
    else:
        print(f"  Not enough subjects in age groups (young={len(young)}, old={len(old)})")

    # ================================================================
    # p_pos analysis (complementary)
    # ================================================================
    print(f"\n{'='*70}")
    print("SUPPLEMENTARY: p_pos analysis")
    print(f"{'='*70}")

    t_pp, p_pp = ttest_rel(p_pos_eo, p_pos_ec)
    print(f"  p_pos EC vs EO: t = {t_pp:.4f}, p = {p_pp:.4e}")
    print(f"  p_pos EC mean={p_pos_ec.mean():.4f}, EO mean={p_pos_eo.mean():.4f}")

    r_pp_ec, p_pp_ec = pearsonr(ages, p_pos_ec)
    r_pp_eo, p_pp_eo = pearsonr(ages, p_pos_eo)
    print(f"  p_pos_ec ~ age: r={r_pp_ec:+.4f}, p={p_pp_ec:.4e}")
    print(f"  p_pos_eo ~ age: r={r_pp_eo:+.4f}, p={p_pp_eo:.4e}")

    # ================================================================
    # Save results
    # ================================================================
    df_out = pd.DataFrame(valid)
    df_out.to_csv(output_dir / "eceo_entropy_results.csv", index=False)

    summary = {
        "n_valid": len(valid),
        "n_failed": len(failed),
        "n_excluded": len(excluded),
        "H_ec_mean": float(H_ec.mean()),
        "H_eo_mean": float(H_eo.mean()),
        "delta_H_mean": float(delta_H.mean()),
        "test1_t": float(t_stat),
        "test1_p_onesided": float(p_one),
        "test1_cohen_d": float(cohen_d),
        "test2_r_deltaH_age": float(r_dh),
        "test2_p_deltaH_age": float(p_dh),
        "test2_r_absDeltaH_age": float(r_abs),
        "test2_p_absDeltaH_age": float(p_abs),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"CSV: {output_dir / 'eceo_entropy_results.csv'}")


def main():
    parser = argparse.ArgumentParser(description="EC vs EO H(sign A) entropy analysis")
    parser.add_argument("--data-dir", default="data/lemon/")
    parser.add_argument("--output", default="results/eceo_entropy/")
    parser.add_argument("--n-subjects", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    args = parser.parse_args()
    run(args.data_dir, args.output, args.n_subjects, args.n_workers)


if __name__ == "__main__":
    main()
