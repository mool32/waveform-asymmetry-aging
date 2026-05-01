#!/usr/bin/env python3
"""
Age prediction from EEG features: PTA vs spectral power vs aperiodic slope.

Compares ridge regression models predicting chronological age from:
  1. PTA features only (waveform asymmetry)
  2. Spectral power only (relative band power)
  3. Aperiodic slope only (1/f exponent)
  4. All spectral features combined
  5. PTA + spectral combined
  6. Sex only (baseline)

Cross-validated on LEMON (N=215), generalized to Dortmund (N=608).

Usage:
    python scripts/run_age_prediction.py [--skip-extraction]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
LEMON_SUMMARY = ROOT / "results" / "phase1_full" / "summary.json"
DORT_SUMMARY = ROOT / "results" / "dortmund_replication" / "summary.json"
LEMON_DATA = ROOT / "data" / "lemon"
SPECTRAL_CACHE = ROOT / "results" / "age_prediction" / "spectral_features.json"
RESULTS_DIR = ROOT / "results" / "age_prediction"
FIG_PATH = ROOT / "paper" / "figures" / "fig6_age_prediction.pdf"

# ── Band definitions (same as PTA pipeline) ──
BAND_EDGES = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "low_gamma": (30, 45),
}

EEG_REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8",
                "AF3", "AF4", "AF7", "AF8", "F1", "F2", "F5", "F6"],
    "central": ["C3", "C4", "Cz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
                "C1", "C2", "C5", "C6"],
    "temporal": ["T7", "T8", "FT7", "FT8", "TP7", "TP8"],
    "parietal": ["P3", "P4", "Pz", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
                 "CPz", "P1", "P2", "P5", "P6", "P7", "P8"],
    "occipital": ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8",
                  "PO9", "PO10", "POz"],
}

# ── Figure style (match paper) ──
COLOR_LEMON = "#377eb8"
COLOR_DORT = "#e41a1c"
COLOR_PTA = "#377eb8"
COLOR_POWER = "#e41a1c"
COLOR_COMBINED = "#984ea3"
COLOR_BASELINE = "#999999"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ═══════════════════════════════════════════════════════════════════
# PHASE A: Spectral Feature Extraction
# ═══════════════════════════════════════════════════════════════════

def extract_spectral_features_all(data_dir, summary):
    """Extract spectral features from all LEMON raw EEG files."""
    import mne
    mne.set_log_level("ERROR")

    subjects = summary["subjects"]
    results = []
    n_total = len(subjects)

    for idx, subj in enumerate(subjects):
        sid = subj["id"]
        vhdr = data_dir / sid / "RSEEG" / f"{sid}.vhdr"

        if not vhdr.exists():
            print(f"  [{idx+1}/{n_total}] {sid}: SKIP (no file)")
            results.append({"id": sid, "status": "missing"})
            continue

        try:
            feats = extract_spectral_single(vhdr, sid, mne)
            feats["id"] = sid
            feats["status"] = "ok"
            results.append(feats)
            print(f"  [{idx+1}/{n_total}] {sid}: OK "
                  f"(slope={feats['aperiodic_slope']:.2f}, "
                  f"alpha_peak={feats['alpha_peak_freq']:.1f} Hz)")
        except Exception as e:
            print(f"  [{idx+1}/{n_total}] {sid}: FAIL ({e})")
            results.append({"id": sid, "status": "error", "error": str(e)})

    return results


def extract_spectral_single(vhdr_path, sid, mne):
    """Extract spectral features from a single subject's raw EEG."""
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

    # Drop non-EEG channels (same as run_pta.py)
    non_eeg = [ch for ch in raw.ch_names if ch in ("VEOG", "HEOG", "ECG", "EKG", "EMG", "AFz")]
    if non_eeg:
        raw.drop_channels(non_eeg)

    # Resample to 250 Hz
    if raw.info["sfreq"] > 375:
        raw.resample(250.0, verbose=False)
    sfreq = raw.info["sfreq"]

    # Notch filter + average reference (same as run_pta.py)
    raw.notch_filter(freqs=[50], fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj(verbose=False)

    data = raw.get_data()  # (n_channels, n_samples)
    ch_names = list(raw.ch_names)
    n_ch = len(ch_names)

    # Compute Welch PSD per channel
    nperseg = min(int(4 * sfreq), data.shape[1])
    all_psd = []
    all_freqs = None
    for i in range(n_ch):
        freqs, psd = welch(data[i], fs=sfreq, nperseg=nperseg)
        all_psd.append(psd)
        if all_freqs is None:
            all_freqs = freqs
    all_psd = np.array(all_psd)  # (n_ch, n_freqs)

    # ── Relative band power per channel ──
    freq_mask_total = (all_freqs >= 1) & (all_freqs <= 45)
    total_power_per_ch = np.trapz(all_psd[:, freq_mask_total], all_freqs[freq_mask_total], axis=1)

    band_power_per_ch = {}  # band -> array(n_ch,)
    for band, (fl, fh) in BAND_EDGES.items():
        mask = (all_freqs >= fl) & (all_freqs <= fh)
        bp = np.trapz(all_psd[:, mask], all_freqs[mask], axis=1)
        band_power_per_ch[band] = bp / (total_power_per_ch + 1e-20)

    # ── Aggregate: global and regional band power ──
    global_power = {}
    regional_power = {}
    for band in BAND_EDGES:
        global_power[band] = float(np.mean(band_power_per_ch[band]))
        for region, region_chs in EEG_REGIONS.items():
            ch_idx = [j for j, ch in enumerate(ch_names) if ch in region_chs]
            if ch_idx:
                key = f"{band}_{region}"
                regional_power[key] = float(np.mean(band_power_per_ch[band][ch_idx]))

    # ── Aperiodic slope (log-log linear fit, 2-40 Hz) ──
    mean_psd = np.mean(all_psd, axis=0)
    slope_mask = (all_freqs >= 2) & (all_freqs <= 40)
    log_f = np.log10(all_freqs[slope_mask])
    log_p = np.log10(mean_psd[slope_mask] + 1e-30)
    coeffs = np.polyfit(log_f, log_p, 1)
    aperiodic_slope = -coeffs[0]  # positive = steeper slope

    # ── Alpha peak frequency ──
    alpha_mask = (all_freqs >= 7) & (all_freqs <= 14)
    alpha_freqs = all_freqs[alpha_mask]
    alpha_psd = mean_psd[alpha_mask]
    alpha_peak_freq = float(alpha_freqs[np.argmax(alpha_psd)])

    return {
        "global_power": global_power,
        "regional_power": regional_power,
        "aperiodic_slope": float(aperiodic_slope),
        "alpha_peak_freq": alpha_peak_freq,
    }


# ═══════════════════════════════════════════════════════════════════
# PHASE B: Feature Matrix Construction
# ═══════════════════════════════════════════════════════════════════

BANDS = ["delta", "theta", "alpha", "beta", "low_gamma"]
REGIONS = ["frontal", "central", "temporal", "parietal", "occipital"]


def build_pta_features(summary):
    """Extract PTA feature matrix from summary.json."""
    rows = []
    for s in summary["subjects"]:
        row = {"id": s["id"], "age": s["age"], "sex": 1 if s["sex"] == "M" else 0}
        for band in BANDS:
            row[f"pta_tmean_{band}"] = s.get(f"trimmed_mean_A_{band}", np.nan)
            row[f"pta_pp_{band}"] = s.get(f"prop_positive_{band}", np.nan)
            row[f"pta_sigma_{band}"] = s.get(f"sigma_A_{band}", np.nan)
        row["pta_sigma_global"] = s.get("sigma_global", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_spectral_features(spectral_data, summary):
    """Build spectral feature matrix from cached spectral extraction."""
    # Map by subject ID
    spec_by_id = {s["id"]: s for s in spectral_data if s.get("status") == "ok"}

    rows = []
    for s in summary["subjects"]:
        sid = s["id"]
        spec = spec_by_id.get(sid)
        if spec is None:
            continue
        row = {"id": sid}
        # Global band power
        for band in BANDS:
            row[f"power_global_{band}"] = spec["global_power"].get(band, np.nan)
        # Regional band power
        for band in BANDS:
            for region in REGIONS:
                key = f"{band}_{region}"
                row[f"power_{key}"] = spec["regional_power"].get(key, np.nan)
        # Slope and alpha peak
        row["aperiodic_slope"] = spec["aperiodic_slope"]
        row["alpha_peak_freq"] = spec["alpha_peak_freq"]
        rows.append(row)
    return pd.DataFrame(rows)


def get_feature_sets(pta_df, spec_df):
    """Define feature sets for comparison."""
    # Merge on ID
    merged = pta_df.merge(spec_df, on="id", how="inner")
    y = merged["age"].values
    sex = merged["sex"].values

    # PTA features (16)
    pta_cols = [c for c in merged.columns if c.startswith("pta_")]
    X_pta = merged[pta_cols].values

    # Power regional (25) + alpha peak (1) = 26
    power_cols = [c for c in merged.columns if c.startswith("power_")]
    X_power = merged[power_cols + ["alpha_peak_freq"]].values

    # Slope only (1)
    X_slope = merged[["aperiodic_slope"]].values

    # Spectral all = power + slope + alpha peak (27)
    spectral_cols = power_cols + ["aperiodic_slope", "alpha_peak_freq"]
    X_spectral = merged[spectral_cols].values

    # PTA + Spectral (43)
    X_combined = merged[pta_cols + spectral_cols].values

    # Sex only (1)
    X_sex = sex.reshape(-1, 1)

    feature_sets = {
        "PTA": X_pta,
        "Power": X_power,
        "Slope": X_slope,
        "Spectral": X_spectral,
        "PTA+Spectral": X_combined,
        "Sex": X_sex,
    }

    feature_names = {
        "PTA": pta_cols,
        "Power": power_cols + ["alpha_peak_freq"],
        "Slope": ["aperiodic_slope"],
        "Spectral": spectral_cols,
        "PTA+Spectral": pta_cols + spectral_cols,
        "Sex": ["sex"],
    }

    return feature_sets, feature_names, y, merged


# ═══════════════════════════════════════════════════════════════════
# PHASE C: Model Training and Evaluation
# ═══════════════════════════════════════════════════════════════════

def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])),
    ])


def age_to_bins(y, n_bins=4):
    """Convert continuous age to quartile bins for stratified CV."""
    bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1
    bins[-1] += 1
    return np.digitize(y, bins)


def _cv_predict_stratified(pipe, X, y, y_bins, cv):
    """Cross-val predict using y_bins for stratified splitting, y for fitting."""
    y_pred = np.full(len(y), np.nan)
    for train_idx, test_idx in cv.split(X, y_bins):
        p = clone(pipe)
        p.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = p.predict(X[test_idx])
    # For repeated CV, average predictions across repeats
    return y_pred


def evaluate_model(X, y, n_splits=10, n_repeats=10, n_perm=1000):
    """Run repeated stratified CV + permutation test."""
    y_bins = age_to_bins(y)

    # Out-of-fold predictions (accumulate across all repeats, then average)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    pred_accum = np.zeros(len(y))
    pred_count = np.zeros(len(y))
    mae_per_repeat = []
    r2_per_repeat = []

    fold_idx = 0
    for train_idx, test_idx in cv.split(X, y_bins):
        pipe = make_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        pred_accum[test_idx] += preds
        pred_count[test_idx] += 1

        fold_idx += 1
        # End of one repeat
        if fold_idx % n_splits == 0:
            # Compute per-repeat metrics from this repeat's folds
            repeat_start = fold_idx - n_splits
            # Collect this repeat's out-of-fold predictions
            cv_single = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1,
                                                 random_state=42 + (fold_idx // n_splits - 1))
            y_pred_single = _cv_predict_stratified(make_pipeline(), X, y, y_bins, cv_single)
            mae_per_repeat.append(mean_absolute_error(y, y_pred_single))
            r2_per_repeat.append(r2_score(y, y_pred_single))

    y_pred = pred_accum / pred_count
    overall_mae = mean_absolute_error(y, y_pred)
    overall_r2 = r2_score(y, y_pred)
    overall_r, _ = pearsonr(y, y_pred)

    # Permutation test (use fewer splits for speed)
    print(f"    Permutation test ({n_perm} perms)...", end=" ", flush=True)
    cv_perm = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    # Manual permutation test with stratified bins
    rng = np.random.RandomState(42)
    real_score = -overall_mae  # neg MAE (higher is better)
    perm_scores = []
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        y_shuf_bins = age_to_bins(y_shuf)
        y_pred_perm = _cv_predict_stratified(make_pipeline(), X, y_shuf, y_shuf_bins, cv_perm)
        perm_scores.append(-mean_absolute_error(y_shuf, y_pred_perm))
    perm_scores = np.array(perm_scores)
    perm_p = (np.sum(perm_scores >= real_score) + 1) / (n_perm + 1)
    print(f"p={perm_p:.4f}")

    return {
        "mae": float(overall_mae),
        "r2": float(overall_r2),
        "r": float(overall_r),
        "mae_per_repeat": [float(x) for x in mae_per_repeat],
        "r2_per_repeat": [float(x) for x in r2_per_repeat],
        "mae_ci95": (float(np.percentile(mae_per_repeat, 2.5)),
                     float(np.percentile(mae_per_repeat, 97.5))),
        "perm_p": float(perm_p),
        "y_pred": y_pred.tolist(),
    }


def cross_dataset_generalize(X_train, y_train, X_test, y_test):
    """Train on LEMON, test on Dortmund."""
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    r, _ = pearsonr(y_test, y_pred)
    return {
        "mae": float(mae),
        "r2": float(r2),
        "r": float(r),
        "y_pred": y_pred.tolist(),
        "y_true": y_test.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════
# PHASE D: Figure Generation
# ═══════════════════════════════════════════════════════════════════

def generate_figure(results, y_true_lemon, dort_gen, chance_mae):
    """Generate publication figure: bar chart + scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.5), gridspec_kw={"wspace": 0.35})

    # ── Panel A: Bar chart of MAE ──
    ax = axes[0]
    model_order = ["PTA", "Power", "Slope", "Spectral", "PTA+Spectral", "Sex"]
    colors = [COLOR_PTA, COLOR_POWER, COLOR_POWER, COLOR_POWER, COLOR_COMBINED, COLOR_BASELINE]
    maes = [results[m]["mae"] for m in model_order]
    ci_low = [results[m]["mae_ci95"][0] for m in model_order]
    ci_high = [results[m]["mae_ci95"][1] for m in model_order]
    yerr_low = [maes[i] - ci_low[i] for i in range(len(maes))]
    yerr_high = [ci_high[i] - maes[i] for i in range(len(maes))]

    x_pos = np.arange(len(model_order))
    bars = ax.bar(x_pos, maes, color=colors, edgecolor="black", linewidth=0.5, width=0.7)
    ax.errorbar(x_pos, maes, yerr=[yerr_low, yerr_high],
                fmt="none", ecolor="black", capsize=3, linewidth=1)

    # Chance level
    ax.axhline(chance_mae, color="gray", linestyle="--", linewidth=1, label=f"Chance ({chance_mae:.1f} yr)")

    # Significance stars
    for i, m in enumerate(model_order):
        p = results[m]["perm_p"]
        if p < 0.001:
            ax.text(i, maes[i] - yerr_low[i] - 0.8, "***", ha="center", fontsize=9)
        elif p < 0.01:
            ax.text(i, maes[i] - yerr_low[i] - 0.8, "**", ha="center", fontsize=9)
        elif p < 0.05:
            ax.text(i, maes[i] - yerr_low[i] - 0.8, "*", ha="center", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("MAE (years)")
    ax.set_title("A. Age prediction: LEMON CV", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, chance_mae + 3)

    # ── Panel B: Scatter — PTA model (the one that generalizes cross-dataset) ──
    ax = axes[1]

    # Use PTA model for both datasets (consistent comparison)
    y_pred_lemon = np.array(results["PTA"]["y_pred"])

    ax.scatter(y_true_lemon, y_pred_lemon, s=15, alpha=0.5,
               c=COLOR_LEMON, edgecolors="none", label="LEMON (CV)")

    if dort_gen is not None:
        y_true_dort = np.array(dort_gen["y_true"])
        y_pred_dort = np.array(dort_gen["y_pred"])
        ax.scatter(y_true_dort, y_pred_dort, s=8, alpha=0.3,
                   c=COLOR_DORT, edgecolors="none", label="Dortmund")

    # Identity line
    lims = [15, 85]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual age (years)")
    ax.set_ylabel("Predicted age (years)")
    ax.set_title("B. PTA model generalization", fontsize=11)
    ax.set_aspect("equal")

    # Annotate
    pta_mae = results["PTA"]["mae"]
    pta_r2 = results["PTA"]["r2"]
    txt = f"LEMON: MAE={pta_mae:.1f}, R²={pta_r2:.2f}"
    if dort_gen is not None:
        txt += f"\nDortmund: MAE={dort_gen['mae']:.1f}, R²={dort_gen['r2']:.2f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG_PATH), dpi=300, bbox_inches="tight")
    print(f"\nFigure saved: {FIG_PATH}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Age prediction: PTA vs spectral")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip spectral extraction (use cached)")
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="Number of permutations (default 1000)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load LEMON PTA data ──
    print("Loading LEMON PTA summary...")
    with open(LEMON_SUMMARY) as f:
        lemon_summary = json.load(f)
    print(f"  N = {lemon_summary['n_subjects']} subjects")

    # ── Phase A: Spectral extraction ──
    if args.skip_extraction and SPECTRAL_CACHE.exists():
        print(f"Loading cached spectral features from {SPECTRAL_CACHE}")
        with open(SPECTRAL_CACHE) as f:
            spectral_data = json.load(f)
        n_ok = sum(1 for s in spectral_data if s.get("status") == "ok")
        print(f"  {n_ok} subjects with spectral features")
    else:
        print("\nExtracting spectral features from raw EEG...")
        spectral_data = extract_spectral_features_all(LEMON_DATA, lemon_summary)
        n_ok = sum(1 for s in spectral_data if s.get("status") == "ok")
        print(f"\n  {n_ok}/{len(spectral_data)} subjects extracted successfully")
        with open(SPECTRAL_CACHE, "w") as f:
            json.dump(spectral_data, f, indent=2)
        print(f"  Cached to {SPECTRAL_CACHE}")

    # ── Phase B: Build feature matrices ──
    print("\nBuilding feature matrices...")
    pta_df = build_pta_features(lemon_summary)
    spec_df = build_spectral_features(spectral_data, lemon_summary)
    feature_sets, feature_names, y_lemon, merged_df = get_feature_sets(pta_df, spec_df)

    print(f"  N subjects (with both PTA + spectral): {len(y_lemon)}")
    for name, X in feature_sets.items():
        print(f"  {name}: {X.shape[1]} features")

    # Chance-level MAE
    chance_mae = float(np.mean(np.abs(y_lemon - np.mean(y_lemon))))
    print(f"  Chance MAE (predict mean): {chance_mae:.1f} years")

    # ── Phase C: Cross-validated age prediction ──
    print("\n" + "="*60)
    print("CROSS-VALIDATED AGE PREDICTION (LEMON)")
    print("="*60)

    results = {}
    for name in ["PTA", "Power", "Slope", "Spectral", "PTA+Spectral", "Sex"]:
        X = feature_sets[name]
        print(f"\n  Model: {name} ({X.shape[1]} features)")
        res = evaluate_model(X, y_lemon, n_perm=args.n_perm)
        results[name] = res
        print(f"    MAE = {res['mae']:.2f} years "
              f"(95% CI: [{res['mae_ci95'][0]:.2f}, {res['mae_ci95'][1]:.2f}])")
        print(f"    R²  = {res['r2']:.3f}, r = {res['r']:.3f}")
        print(f"    Permutation p = {res['perm_p']:.4f}")

    # ── Phase D: Cross-dataset generalization (PTA only) ──
    print("\n" + "="*60)
    print("CROSS-DATASET GENERALIZATION (LEMON → DORTMUND)")
    print("="*60)

    dort_gen = None
    if DORT_SUMMARY.exists():
        print("Loading Dortmund PTA summary...")
        with open(DORT_SUMMARY) as f:
            dort_summary = json.load(f)
        dort_pta_df = build_pta_features(dort_summary)
        dort_pta_df = dort_pta_df.dropna(subset=["age"])

        # PTA columns (same as LEMON)
        pta_cols = [c for c in merged_df.columns if c.startswith("pta_")]
        X_train = merged_df[pta_cols].values
        y_train = y_lemon
        X_test = dort_pta_df[pta_cols].values
        y_test = dort_pta_df["age"].values

        print(f"  Train: LEMON N={len(y_train)}, Test: Dortmund N={len(y_test)}")
        dort_gen = cross_dataset_generalize(X_train, y_train, X_test, y_test)
        print(f"  Dortmund MAE = {dort_gen['mae']:.2f} years")
        print(f"  Dortmund R²  = {dort_gen['r2']:.3f}, r = {dort_gen['r']:.3f}")
    else:
        print("  Dortmund summary not found, skipping generalization")

    # ── Generate figure ──
    print("\nGenerating figure...")
    generate_figure(results, y_lemon, dort_gen, chance_mae)

    # ── Save results ──
    output = {
        "n_lemon": int(len(y_lemon)),
        "chance_mae": float(chance_mae),
        "models": {},
        "cross_dataset": dort_gen,
    }
    for name, res in results.items():
        output["models"][name] = {
            "mae": res["mae"],
            "r2": res["r2"],
            "r": res["r"],
            "mae_ci95": res["mae_ci95"],
            "perm_p": res["perm_p"],
            "n_features": feature_sets[name].shape[1],
        }

    results_path = RESULTS_DIR / "age_prediction_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── Summary table ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<18} {'N feat':>6} {'MAE':>8} {'R²':>8} {'r':>8} {'p_perm':>8}")
    print("-" * 60)
    for name in ["PTA", "Power", "Slope", "Spectral", "PTA+Spectral", "Sex"]:
        r = results[name]
        print(f"{name:<18} {feature_sets[name].shape[1]:>6} "
              f"{r['mae']:>8.2f} {r['r2']:>8.3f} {r['r']:>8.3f} {r['perm_p']:>8.4f}")
    if dort_gen:
        print(f"\nDortmund generalization (PTA): MAE={dort_gen['mae']:.2f}, "
              f"R²={dort_gen['r2']:.3f}, r={dort_gen['r']:.3f}")


if __name__ == "__main__":
    main()
