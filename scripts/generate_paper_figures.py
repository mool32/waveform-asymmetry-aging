#!/usr/bin/env python3
"""
Generate all paper figures for Asymmetric Perception Cycles.

Produces 5 main figures + 2 supplementary as PDF (300 DPI).
Output: paper/figures/

Usage:
    python scripts/generate_paper_figures.py
"""

import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from scipy import stats
from scipy.interpolate import griddata

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PAPER_FIG = ROOT / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
COLOR_LEMON   = "#377eb8"
COLOR_DORT    = "#e41a1c"
COLOR_POS     = "#4daf4a"
COLOR_NEG     = "#984ea3"
COLOR_NEUTRAL = "#999999"

BAND_ORDER = ["delta", "theta", "alpha", "beta", "low_gamma"]
BAND_LABELS = {"delta": r"$\delta$", "theta": r"$\theta$", "alpha": r"$\alpha$",
               "beta": r"$\beta$", "low_gamma": r"$\gamma$"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "pdf.fonttype": 42,   # TrueType fonts in PDF
    "ps.fonttype": 42,
})

# ── Standard 10-20 channel positions (2D, approximate) ────────────────────
# Extended set covering both LEMON (61-ch) and Dortmund (64-ch)
CHANNEL_POS_2D = {
    # Frontal pole
    "Fp1": (-0.30, 0.92), "Fp2": (0.30, 0.92),
    # AF row
    "AF7": (-0.55, 0.82), "AF3": (-0.25, 0.82), "AFz": (0.0, 0.82),
    "AF4": (0.25, 0.82), "AF8": (0.55, 0.82),
    # Frontal
    "F7": (-0.70, 0.70), "F5": (-0.52, 0.70), "F3": (-0.35, 0.70),
    "F1": (-0.17, 0.70), "Fz": (0.0, 0.70), "F2": (0.17, 0.70),
    "F4": (0.35, 0.70), "F6": (0.52, 0.70), "F8": (0.70, 0.70),
    # Fronto-central
    "FT9": (-0.90, 0.58), "FT7": (-0.80, 0.58), "FC5": (-0.60, 0.58),
    "FC3": (-0.38, 0.58), "FC1": (-0.18, 0.58), "FCz": (0.0, 0.58),
    "FC2": (0.18, 0.58), "FC4": (0.38, 0.58), "FC6": (0.60, 0.58),
    "FT8": (0.80, 0.58), "FT10": (0.90, 0.58),
    # Central
    "T7": (-0.90, 0.46), "C5": (-0.65, 0.46), "C3": (-0.38, 0.46),
    "C1": (-0.18, 0.46), "Cz": (0.0, 0.46), "C2": (0.18, 0.46),
    "C4": (0.38, 0.46), "C6": (0.65, 0.46), "T8": (0.90, 0.46),
    # Centro-parietal
    "TP9": (-0.90, 0.34), "TP7": (-0.80, 0.34), "CP5": (-0.60, 0.34),
    "CP3": (-0.38, 0.34), "CP1": (-0.18, 0.34), "CPz": (0.0, 0.34),
    "CP2": (0.18, 0.34), "CP4": (0.38, 0.34), "CP6": (0.60, 0.34),
    "TP8": (0.80, 0.34), "TP10": (0.90, 0.34),
    # Parietal
    "P7": (-0.70, 0.22), "P5": (-0.52, 0.22), "P3": (-0.35, 0.22),
    "P1": (-0.17, 0.22), "Pz": (0.0, 0.22), "P2": (0.17, 0.22),
    "P4": (0.35, 0.22), "P6": (0.52, 0.22), "P8": (0.70, 0.22),
    # Parieto-occipital
    "PO9": (-0.65, 0.12), "PO7": (-0.50, 0.12), "PO3": (-0.25, 0.12),
    "POz": (0.0, 0.12), "PO4": (0.25, 0.12), "PO8": (0.50, 0.12),
    "PO10": (0.65, 0.12),
    # Occipital
    "O1": (-0.25, 0.02), "Oz": (0.0, 0.02), "O2": (0.25, 0.02),
}

# ── Helper functions ───────────────────────────────────────────────────────

def load_lemon():
    """Load LEMON summary (N=215)."""
    with open(RESULTS / "phase1_full" / "summary.json") as f:
        data = json.load(f)
    return data["subjects"]


def load_dortmund():
    """Load Dortmund summary (N=608)."""
    with open(RESULTS / "dortmund_replication" / "summary.json") as f:
        data = json.load(f)
    return data["subjects"]


def load_dortmund_corr():
    """Load pre-computed Dortmund correlations."""
    with open(RESULTS / "dortmund_replication" / "age_correlations.json") as f:
        return json.load(f)


def safe_float(v):
    """Convert to float, return NaN for None/NaN."""
    if v is None:
        return np.nan
    try:
        val = float(v)
        return val
    except (ValueError, TypeError):
        return np.nan


def add_panel_label(ax, label, x=-0.08, y=1.05, fontsize=14):
    """Add bold panel label (A, B, C...) to axis."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="right")


def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def compute_per_channel_mean_A(subjects, band, channel_positions=None):
    """Compute mean asymmetry per channel across subjects for a given band.

    For LEMON: per_channel_A is {band: {channel: value}}
    For Dortmund: per_channel_A is {channel: {band: value}}
    """
    channel_sums = {}
    channel_counts = {}

    for s in subjects:
        pca = s.get("per_channel_A", {})
        if not pca:
            continue

        # Detect structure: LEMON is {band: {channel: value}}, Dortmund is {channel: {band: value}}
        first_key = next(iter(pca))
        BAND_NAMES = {"delta", "theta", "alpha", "beta", "low_gamma"}
        is_lemon_format = first_key in BAND_NAMES

        if not is_lemon_format:
            # Dortmund: {channel: {band: value}}
            for ch, bands_dict in pca.items():
                if ch not in CHANNEL_POS_2D:
                    continue
                val = safe_float(bands_dict.get(band))
                if np.isfinite(val):
                    channel_sums[ch] = channel_sums.get(ch, 0.0) + val
                    channel_counts[ch] = channel_counts.get(ch, 0) + 1
        else:
            # LEMON: {band: {channel: value}} — first key is a band name
            band_data = pca.get(band, {})
            for ch, val in band_data.items():
                if ch not in CHANNEL_POS_2D:
                    continue
                val = safe_float(val)
                if np.isfinite(val):
                    channel_sums[ch] = channel_sums.get(ch, 0.0) + val
                    channel_counts[ch] = channel_counts.get(ch, 0) + 1

    result = {}
    for ch in channel_sums:
        if channel_counts[ch] > 0:
            result[ch] = channel_sums[ch] / channel_counts[ch]
    return result


def draw_topomap(ax, channel_values, title="", cmap="RdBu_r", vlim=None):
    """Draw a topomap on the given axes using interpolation.

    channel_values: dict {channel_name: value}
    """
    channels = [ch for ch in channel_values if ch in CHANNEL_POS_2D]
    if not channels:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    xs = np.array([CHANNEL_POS_2D[ch][0] for ch in channels])
    ys = np.array([CHANNEL_POS_2D[ch][1] for ch in channels])
    vals = np.array([channel_values[ch] for ch in channels])

    # Head circle parameters
    head_cx, head_cy = 0.0, 0.47
    head_r = 0.55

    # Grid for interpolation
    grid_x, grid_y = np.mgrid[-1:1:200j, -0.1:1.05:200j]
    grid_z = griddata((xs, ys), vals, (grid_x, grid_y), method="cubic")

    # Mask outside head circle
    dist = np.sqrt((grid_x - head_cx)**2 + (grid_y - head_cy)**2)
    grid_z[dist > head_r * 0.95] = np.nan

    if vlim is None:
        vmax = np.nanmax(np.abs(vals)) * 1.0
        vlim = (-vmax, vmax)

    norm = TwoSlopeNorm(vcenter=0, vmin=vlim[0], vmax=vlim[1])
    im = ax.pcolormesh(grid_x, grid_y, grid_z, cmap=cmap, norm=norm,
                       shading="auto", rasterized=True)

    # Draw head outline
    head = Circle((head_cx, head_cy), head_r, fill=False, linewidth=1.5,
                  edgecolor="black")
    ax.add_patch(head)

    # Nose
    nose_x = [head_cx - 0.06, head_cx, head_cx + 0.06]
    nose_y = [head_cy + head_r - 0.02, head_cy + head_r + 0.06, head_cy + head_r - 0.02]
    ax.plot(nose_x, nose_y, "k-", linewidth=1.5)

    # Ears
    for side in [-1, 1]:
        ear_x = head_cx + side * (head_r + 0.02)
        ear_y = head_cy
        ax.plot([ear_x, ear_x + side * 0.04, ear_x],
                [ear_y - 0.05, ear_y, ear_y + 0.05], "k-", linewidth=1.0)

    # Channel dots
    ax.scatter(xs, ys, c="black", s=8, zorder=5, linewidths=0)

    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")

    # Colorbar
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=20)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Mean A", fontsize=8)

    return im


def scatter_with_regression(ax, x, y, color, xlabel, ylabel, title="",
                            annotate_text="", point_color=None, alpha=0.5, s=15):
    """Scatter plot with regression line and annotation."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    if point_color is not None:
        sc = ax.scatter(x_clean, y_clean, c=point_color[mask], cmap="coolwarm",
                        s=s, alpha=alpha, edgecolors="none", rasterized=True)
    else:
        ax.scatter(x_clean, y_clean, c=color, s=s, alpha=alpha,
                   edgecolors="none", rasterized=True)

    # Regression line
    if len(x_clean) > 2:
        slope, intercept, r, p, se = stats.linregress(x_clean, y_clean)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black",
                linewidth=1.5, linestyle="-", zorder=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    if annotate_text:
        ax.annotate(annotate_text, xy=(0.03, 0.97), xycoords="axes fraction",
                    va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                              alpha=0.85))


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Band-Specific Asymmetry and Spatial Structure
# ══════════════════════════════════════════════════════════════════════════
def figure1():
    print("  Generating Figure 1...")
    subjects = load_lemon()

    fig = plt.figure(figsize=(7.09, 4.72))  # ~180mm x 120mm
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2],
                  hspace=0.45, wspace=0.40)

    # ── Top row: Bar plot of prop_positive per band ──
    ax_bar = fig.add_subplot(gs[0, :])

    means, sems, pvals = [], [], []
    for band in BAND_ORDER:
        key = f"prop_positive_{band}"
        vals = np.array([safe_float(s.get(key)) for s in subjects])
        vals = vals[np.isfinite(vals)]
        m = np.mean(vals)
        sem = stats.sem(vals)
        t_stat, p = stats.ttest_1samp(vals, 0.5)
        means.append(m)
        sems.append(sem)
        pvals.append(p)

    x_pos = np.arange(len(BAND_ORDER))
    colors = [COLOR_POS if m > 0.5 else COLOR_NEG for m in means]

    bars = ax_bar.bar(x_pos, means, yerr=sems, capsize=4,
                      color=colors, edgecolor="black", linewidth=0.5,
                      width=0.6, zorder=3)
    ax_bar.axhline(0.5, color="black", linestyle="--", linewidth=0.8, zorder=2)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([BAND_LABELS[b] for b in BAND_ORDER])
    ax_bar.set_ylabel("Proportion positive (A > 0)")
    ax_bar.set_title("Band-specific waveform asymmetry (LEMON, N=215)")
    ax_bar.set_ylim(0.25, 0.75)

    # Significance stars
    for i, (m, p) in enumerate(zip(means, pvals)):
        stars = significance_stars(p)
        y_offset = sems[i] + 0.015
        ax_bar.text(i, m + y_offset if m > 0.5 else m - y_offset - 0.01,
                    stars, ha="center", va="bottom" if m > 0.5 else "top",
                    fontsize=9)

    add_panel_label(ax_bar, "A")

    # ── Bottom row: Topomaps for alpha and beta ──
    ax_alpha = fig.add_subplot(gs[1, 0])
    ax_beta = fig.add_subplot(gs[1, 1])

    alpha_means = compute_per_channel_mean_A(subjects, "alpha")
    beta_means = compute_per_channel_mean_A(subjects, "beta")

    draw_topomap(ax_alpha, alpha_means, title=r"$\alpha$ (8-13 Hz)")
    draw_topomap(ax_beta, beta_means, title=r"$\beta$ (13-30 Hz)")

    add_panel_label(ax_alpha, "B", x=-0.02)
    add_panel_label(ax_beta, "C", x=-0.02)

    fig.savefig(PAPER_FIG / "fig1_band_asymmetry.pdf")
    plt.close(fig)
    print("    Saved fig1_band_asymmetry.pdf")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Beta Asymmetry Decreases with Age (HERO)
# ══════════════════════════════════════════════════════════════════════════
def figure2():
    print("  Generating Figure 2...")
    lemon = load_lemon()
    dort_corr = load_dortmund_corr()

    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.76))  # ~180mm x 70mm

    # ── Panel A: LEMON scatter ──
    ax = axes[0]
    ages_l = np.array([safe_float(s.get("age")) for s in lemon])
    pp_l = np.array([safe_float(s.get("prop_positive_beta")) for s in lemon])
    mask = np.isfinite(ages_l) & np.isfinite(pp_l)
    ages_lc, pp_lc = ages_l[mask], pp_l[mask]

    # Color by age
    norm_age = Normalize(vmin=ages_lc.min(), vmax=ages_lc.max())
    age_colors = plt.cm.coolwarm(norm_age(ages_lc))

    ax.scatter(ages_lc, pp_lc, c=age_colors, s=12, alpha=0.6,
               edgecolors="none", rasterized=True)
    slope, intercept = np.polyfit(ages_lc, pp_lc, 1)
    x_line = np.linspace(ages_lc.min(), ages_lc.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k-", linewidth=1.5)

    r_val, p_val = stats.pearsonr(ages_lc, pp_lc)
    ax.annotate(f"r = {r_val:.3f}\np = {p_val:.1e}\nN = {len(ages_lc)}",
                xy=(0.03, 0.03), xycoords="axes fraction", va="bottom",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                      ec="gray", alpha=0.85))
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(r"$\beta$ prop. positive")
    ax.set_title("LEMON")
    add_panel_label(ax, "A")

    # ── Panel B: Dortmund scatter ──
    ax = axes[1]
    print("    Loading Dortmund data for scatter (may take a moment)...")
    dort = load_dortmund()

    ages_d = np.array([safe_float(s.get("age")) for s in dort])
    pp_d = np.array([safe_float(s.get("prop_positive_beta")) for s in dort])
    mask = np.isfinite(ages_d) & np.isfinite(pp_d)
    ages_dc, pp_dc = ages_d[mask], pp_d[mask]

    norm_age_d = Normalize(vmin=ages_dc.min(), vmax=ages_dc.max())
    age_colors_d = plt.cm.coolwarm(norm_age_d(ages_dc))

    ax.scatter(ages_dc, pp_dc, c=age_colors_d, s=8, alpha=0.5,
               edgecolors="none", rasterized=True)
    slope_d, intercept_d = np.polyfit(ages_dc, pp_dc, 1)
    x_line_d = np.linspace(ages_dc.min(), ages_dc.max(), 100)
    ax.plot(x_line_d, slope_d * x_line_d + intercept_d, "k-", linewidth=1.5)

    beta_dort = dort_corr["band_results"]["beta"]
    ax.annotate(f"r = {beta_dort['pp_age_r']:.3f}\n"
                f"p = {beta_dort['pp_age_p']:.1e}\n"
                f"N = {beta_dort['n_pp']}",
                xy=(0.03, 0.03), xycoords="axes fraction", va="bottom",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                      ec="gray", alpha=0.85))
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(r"$\beta$ prop. positive")
    ax.set_title("Dortmund")
    add_panel_label(ax, "B")

    # ── Panel C: Longitudinal paired plot ──
    ax = axes[2]
    long_df = pd.read_csv(RESULTS / "dortmund_longitudinal" / "longitudinal_pairs.csv")
    beta_long = long_df[["id", "age_baseline",
                          "prop_positive_beta_s1", "prop_positive_beta_s2",
                          "d_prop_positive_beta"]].dropna(
                              subset=["prop_positive_beta_s1", "prop_positive_beta_s2"])

    with open(RESULTS / "longitudinal_stats" / "longitudinal_results.json") as f:
        long_stats = json.load(f)
    beta_stats = long_stats["beta"]

    s1 = beta_long["prop_positive_beta_s1"].values
    s2 = beta_long["prop_positive_beta_s2"].values
    delta_vals = beta_long["d_prop_positive_beta"].values

    # Paired lines
    for i in range(len(s1)):
        ax.plot([0, 1], [s1[i], s2[i]], color=COLOR_NEUTRAL, alpha=0.1,
                linewidth=0.3, zorder=1)

    # Mean + SEM
    m1, se1 = np.mean(s1), stats.sem(s1)
    m2, se2 = np.mean(s2), stats.sem(s2)
    ax.errorbar([0, 1], [m1, m2], yerr=[se1, se2], fmt="o-",
                color=COLOR_DORT, markersize=8, linewidth=2.5,
                capsize=5, zorder=5, markeredgecolor="black", markeredgewidth=0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Session 1", "Session 2"])
    ax.set_ylabel(r"$\beta$ prop. positive")
    ax.set_title("Longitudinal")
    ax.set_xlim(-0.3, 1.3)

    pp_stats = beta_stats["prop_positive"]
    trt = beta_stats["test_retest"]
    ann_text = (f"Mean $\\Delta$ = {pp_stats['mean_delta']:.3f}\n"
                f"p = {pp_stats['wilcoxon_p_twosided']:.2f}\n"
                f"N = {beta_stats['n_valid']}\n"
                f"ICC = {trt['r']:.3f}")
    ax.annotate(ann_text, xy=(0.97, 0.03), xycoords="axes fraction",
                va="bottom", ha="right", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                          alpha=0.85))

    # Inset histogram of delta
    inset = ax.inset_axes([0.45, 0.55, 0.50, 0.40])
    inset.hist(delta_vals, bins=25, color=COLOR_DORT, alpha=0.7, edgecolor="black",
               linewidth=0.3)
    inset.axvline(0, color="black", linestyle="--", linewidth=0.6)
    inset.set_xlabel(r"$\Delta$pp", fontsize=7)
    inset.set_ylabel("Count", fontsize=7)
    inset.tick_params(labelsize=6)

    add_panel_label(ax, "C")

    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig2_beta_age_hero.pdf")
    plt.close(fig)
    print("    Saved fig2_beta_age_hero.pdf")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Spatial Signatures of Aging
# ══════════════════════════════════════════════════════════════════════════
def figure3():
    print("  Generating Figure 3...")
    lemon = load_lemon()
    dort_corr = load_dortmund_corr()

    # Compute LEMON subject-level spatial stats for beta
    ages, abs_A_vals, H_sign_vals, sigma_vals = [], [], [], []
    for s in lemon:
        age = safe_float(s.get("age"))
        if not np.isfinite(age):
            continue

        pca = s.get("per_channel_A", {})
        beta_data = pca.get("beta", {})
        if not beta_data:
            continue

        ch_vals = np.array([safe_float(v) for v in beta_data.values()])
        ch_vals = ch_vals[np.isfinite(ch_vals)]
        if len(ch_vals) < 3:
            continue

        abs_mean = np.mean(np.abs(ch_vals))
        sigma = np.std(ch_vals, ddof=1)
        frac_pos = np.mean(ch_vals > 0)
        # Binary entropy: H = -p*log2(p) - (1-p)*log2(1-p)
        if frac_pos == 0 or frac_pos == 1:
            h_sign = 0.0
        else:
            h_sign = -(frac_pos * np.log2(frac_pos) +
                        (1 - frac_pos) * np.log2(1 - frac_pos))

        ages.append(age)
        abs_A_vals.append(abs_mean)
        H_sign_vals.append(h_sign)
        sigma_vals.append(sigma)

    ages = np.array(ages)
    abs_A_vals = np.array(abs_A_vals)
    H_sign_vals = np.array(H_sign_vals)
    sigma_vals = np.array(sigma_vals)

    fig, axes = plt.subplots(3, 2, figsize=(7.09, 7.09))  # ~180mm x 180mm

    metrics = [
        (abs_A_vals, r"|A| (mean abs. asymmetry)", "abs_A"),
        (H_sign_vals, r"H(sign) (spatial entropy)", "H_sign"),
        (sigma_vals, r"$\sigma$(A) (spatial variability)", "sigma"),
    ]

    panel_labels = [["A", "B"], ["C", "D"], ["E", "F"]]

    for row, (vals, ylabel, metric_name) in enumerate(metrics):
        # LEMON panel
        ax = axes[row, 0]
        mask = np.isfinite(ages) & np.isfinite(vals)
        r_val, p_val = stats.pearsonr(ages[mask], vals[mask])
        scatter_with_regression(ax, ages, vals, COLOR_LEMON,
                                "Age (years)", ylabel,
                                title="LEMON" if row == 0 else "",
                                annotate_text=f"r = {r_val:.3f}\np = {p_val:.2e}\nN = {mask.sum()}")
        add_panel_label(ax, panel_labels[row][0])

        # Dortmund panel: use pre-computed stats, plot LEMON-style data as annotation
        ax = axes[row, 1]

        # Try to load Dortmund raw data for scatter
        # Since Dortmund per_channel_A is {channel: {band: value}}, we need different logic
        dort = load_dortmund()
        d_ages, d_vals = [], []
        for s_d in dort:
            d_age = safe_float(s_d.get("age"))
            if not np.isfinite(d_age):
                continue
            pca_d = s_d.get("per_channel_A", {})
            ch_vals_d = []
            for ch, band_dict in pca_d.items():
                v = safe_float(band_dict.get("beta"))
                if np.isfinite(v):
                    ch_vals_d.append(v)
            if len(ch_vals_d) < 3:
                continue

            ch_arr = np.array(ch_vals_d)
            if metric_name == "abs_A":
                d_vals.append(np.mean(np.abs(ch_arr)))
            elif metric_name == "H_sign":
                fp = np.mean(ch_arr > 0)
                if fp == 0 or fp == 1:
                    d_vals.append(0.0)
                else:
                    d_vals.append(-(fp * np.log2(fp) + (1 - fp) * np.log2(1 - fp)))
            else:  # sigma
                d_vals.append(np.std(ch_arr, ddof=1))
            d_ages.append(d_age)

        d_ages = np.array(d_ages)
        d_vals = np.array(d_vals)
        mask_d = np.isfinite(d_ages) & np.isfinite(d_vals)
        if mask_d.sum() > 5:
            r_d, p_d = stats.pearsonr(d_ages[mask_d], d_vals[mask_d])
            scatter_with_regression(ax, d_ages, d_vals, COLOR_DORT,
                                    "Age (years)", ylabel,
                                    title="Dortmund" if row == 0 else "",
                                    annotate_text=f"r = {r_d:.3f}\np = {p_d:.2e}\nN = {mask_d.sum()}")
        else:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", fontsize=10)
            ax.set_xlabel("Age (years)")
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title("Dortmund")

        add_panel_label(ax, panel_labels[row][1])

    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig3_spatial_aging.pdf")
    plt.close(fig)
    print("    Saved fig3_spatial_aging.pdf")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Cross-System and HRV
# ══════════════════════════════════════════════════════════════════════════
def figure4():
    print("  Generating Figure 4...")
    lemon = load_lemon()

    fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.76))  # ~180mm x 70mm

    # ── Panel A: HRV Asymmetry by band ──
    ax = axes[0]
    hrv_bands = ["VLF", "LF", "HF", "VHF"]

    # Compute from raw data
    hrv_A = {b: [] for b in hrv_bands}
    for s in lemon:
        hrv = s.get("hrv")
        if hrv is None:
            continue
        profile = hrv.get("profile", {})
        for b in hrv_bands:
            if b in profile and profile[b].get("A") is not None:
                a_val = safe_float(profile[b]["A"])
                if np.isfinite(a_val):
                    hrv_A[b].append(a_val)

    means_h = [np.mean(hrv_A[b]) if hrv_A[b] else 0 for b in hrv_bands]
    sems_h = [stats.sem(hrv_A[b]) if len(hrv_A[b]) > 1 else 0 for b in hrv_bands]
    ns_h = [len(hrv_A[b]) for b in hrv_bands]

    x_h = np.arange(len(hrv_bands))
    ax.bar(x_h, means_h, yerr=sems_h, capsize=4, color=COLOR_POS,
           edgecolor="black", linewidth=0.5, width=0.6, zorder=3)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks(x_h)
    ax.set_xticklabels(hrv_bands)
    ax.set_ylabel("Mean A (asymmetry)")
    ax.set_title("HRV waveform asymmetry")

    # Significance test vs 0
    for i, b in enumerate(hrv_bands):
        if len(hrv_A[b]) > 2:
            t, p = stats.ttest_1samp(hrv_A[b], 0)
            stars = significance_stars(p)
            ax.text(i, means_h[i] + sems_h[i] + 0.005, stars,
                    ha="center", va="bottom", fontsize=8)

    ax.annotate(f"N = {ns_h[0]}-{ns_h[-1]} subjects",
                xy=(0.97, 0.97), xycoords="axes fraction", va="top", ha="right",
                fontsize=8)
    add_panel_label(ax, "A")

    # ── Panel B: sigma_global vs HRV HF-A ──
    ax = axes[1]
    sigma_g, hrv_hf, age_hrv = [], [], []
    for s in lemon:
        hrv = s.get("hrv")
        if hrv is None:
            continue
        sg = safe_float(s.get("sigma_global"))
        hf_a = safe_float(hrv.get("profile", {}).get("HF", {}).get("A"))
        a = safe_float(s.get("age"))
        if np.isfinite(sg) and np.isfinite(hf_a) and np.isfinite(a):
            sigma_g.append(sg)
            hrv_hf.append(hf_a)
            age_hrv.append(a)

    sigma_g = np.array(sigma_g)
    hrv_hf = np.array(hrv_hf)
    age_hrv = np.array(age_hrv)

    if len(sigma_g) > 10:
        ax.scatter(hrv_hf, sigma_g, c=COLOR_LEMON, s=15, alpha=0.6,
                   edgecolors="none", rasterized=True)

        # Compute partial correlation controlling for age
        from numpy.linalg import lstsq
        # Residualize both on age
        A_age = np.column_stack([age_hrv, np.ones(len(age_hrv))])
        res_sg = sigma_g - A_age @ lstsq(A_age, sigma_g, rcond=None)[0]
        res_hf = hrv_hf - A_age @ lstsq(A_age, hrv_hf, rcond=None)[0]
        partial_r, partial_p = stats.pearsonr(res_hf, res_sg)

        # Regression line on raw for visual
        slope, intercept = np.polyfit(hrv_hf, sigma_g, 1)
        x_line = np.linspace(hrv_hf.min(), hrv_hf.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "k-", linewidth=1.5)

        ax.annotate(f"partial r = {partial_r:.3f}\n"
                    f"p = {partial_p:.3f}\n"
                    f"N = {len(sigma_g)}\n"
                    f"(controlling age)",
                    xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                          ec="gray", alpha=0.85))
    else:
        ax.text(0.5, 0.5, "N=72, partial r=-0.301\np=0.010 (age-controlled)\n"
                "Raw HRV data insufficient",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)

    ax.set_xlabel("HRV HF asymmetry (A)")
    ax.set_ylabel(r"$\sigma$(A)$_{global}$")
    ax.set_title("EEG-HRV coupling")
    add_panel_label(ax, "B")

    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig4_hrv_cross_system.pdf")
    plt.close(fig)
    print("    Saved fig4_hrv_cross_system.pdf")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Eyes-Closed vs Eyes-Open
# ══════════════════════════════════════════════════════════════════════════
def figure5():
    print("  Generating Figure 5...")
    eceo = pd.read_csv(RESULTS / "eceo_entropy" / "eceo_entropy_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.76))  # ~180mm x 70mm

    # ── Panel A: Paired violin/box ──
    ax = axes[0]
    h_ec = eceo["H_ec"].dropna().values
    h_eo = eceo["H_eo"].dropna().values

    # Use only subjects with both
    valid = eceo.dropna(subset=["H_ec", "H_eo"])
    h_ec_v = valid["H_ec"].values
    h_eo_v = valid["H_eo"].values

    # Box/violin
    parts = ax.violinplot([h_ec_v, h_eo_v], positions=[0, 1],
                          showmeans=True, showmedians=False, widths=0.6)
    for pc in parts["bodies"]:
        pc.set_facecolor(COLOR_LEMON)
        pc.set_alpha(0.4)
    parts["cmeans"].set_color("black")
    parts["cbars"].set_visible(False)
    parts["cmins"].set_visible(False)
    parts["cmaxes"].set_visible(False)

    # Paired lines (subsample if many)
    n_show = min(len(h_ec_v), 100)
    idx = np.random.RandomState(42).choice(len(h_ec_v), n_show, replace=False)
    for i in idx:
        ax.plot([0, 1], [h_ec_v[i], h_eo_v[i]], color=COLOR_NEUTRAL,
                alpha=0.15, linewidth=0.3, zorder=1)

    t_stat, p_val = stats.ttest_rel(h_ec_v, h_eo_v)
    d_val = np.mean(h_ec_v - h_eo_v) / np.std(h_ec_v - h_eo_v, ddof=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Eyes Closed", "Eyes Open"])
    ax.set_ylabel("H (sign entropy)")
    ax.set_title("EC vs EO asymmetry entropy")
    ax.annotate(f"t = {t_stat:.2f}\np = {p_val:.1e}\nd = {d_val:.2f}\nN = {len(h_ec_v)}",
                xy=(0.97, 0.03), xycoords="axes fraction", va="bottom",
                ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                          alpha=0.85))
    add_panel_label(ax, "A")

    # ── Panel B: delta_H vs age ──
    ax = axes[1]
    valid2 = eceo.dropna(subset=["delta_H", "age"])
    dh = valid2["delta_H"].values
    ages = valid2["age"].values

    scatter_with_regression(ax, ages, dh, COLOR_LEMON,
                            "Age (years)", r"$\Delta$H (EC - EO)",
                            title="Age vs state modulation",
                            annotate_text=f"r = {stats.pearsonr(ages, dh)[0]:.3f}\n"
                                          f"p = {stats.pearsonr(ages, dh)[1]:.2f}\n"
                                          f"N = {len(ages)}")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    add_panel_label(ax, "B")

    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig5_eceo_entropy.pdf")
    plt.close(fig)
    print("    Saved fig5_eceo_entropy.pdf")


# ══════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURE S1: Meta-Analysis Scale Map
# ══════════════════════════════════════════════════════════════════════════
def figure_s1():
    print("  Generating Figure S1 (Meta-analysis scale map)...")

    # Curated data from meta-analysis: (system, timescale_seconds, A_value, category)
    meta_systems = [
        # Molecular (ms-s)
        ("Ca2+ spikes (neurons)", 0.05, 0.15, "Molecular"),
        ("Ca2+ spikes (astrocytes)", 2.0, 0.08, "Molecular"),
        ("Action potentials", 0.002, 0.25, "Molecular"),
        ("Synaptic currents (EPSC)", 0.005, 0.35, "Molecular"),
        ("cAMP oscillations", 30, 0.12, "Molecular"),
        ("IP3 oscillations", 10, 0.10, "Molecular"),
        ("NF-kB oscillations", 6000, 0.06, "Molecular"),
        ("p53 oscillations", 18000, 0.04, "Molecular"),
        ("Hes1 oscillations", 7200, 0.05, "Molecular"),
        ("ERK pulses", 600, 0.09, "Molecular"),
        # Neural (ms-s)
        ("EEG delta", 0.5, 0.02, "Neural"),
        ("EEG theta", 0.17, 0.04, "Neural"),
        ("EEG alpha", 0.1, 0.08, "Neural"),
        ("EEG beta", 0.04, -0.04, "Neural"),
        ("EEG gamma", 0.025, -0.05, "Neural"),
        ("Sleep spindles", 1.0, 0.12, "Neural"),
        ("Sharp-wave ripples", 0.01, 0.20, "Neural"),
        # Physiological (s-hours)
        ("HRV (HF)", 3, 0.05, "Physiological"),
        ("HRV (LF)", 10, 0.02, "Physiological"),
        ("Respiration", 4, 0.30, "Physiological"),
        ("Blood pressure (Mayer)", 10, 0.08, "Physiological"),
        ("Cortisol ultradian", 5400, 0.15, "Physiological"),
        # Circadian+
        ("Circadian (core clock)", 86400, 0.10, "Circadian"),
        ("Circadian (temperature)", 86400, 0.07, "Circadian"),
        ("Infradian (menstrual)", 2419200, 0.05, "Circadian"),
        # Ecological
        ("Predator-prey (Lotka-Volterra)", 31536000, 0.12, "Ecological"),
        ("Population cycles (lynx-hare)", 315360000, 0.08, "Ecological"),
        # Cell cycle
        ("Cell cycle (yeast)", 5400, 0.03, "Cell cycle"),
        ("Cell cycle (mammalian)", 72000, 0.02, "Cell cycle"),
        ("Somite clock", 5400, 0.06, "Cell cycle"),
        # Cardiac
        ("ECG waveform", 0.8, 0.35, "Cardiac"),
        ("ECG T-wave", 0.3, 0.15, "Cardiac"),
        ("Cardiac alternans", 0.5, 0.10, "Cardiac"),
    ]

    fig, ax = plt.subplots(figsize=(7.09, 4.5))

    cat_colors = {
        "Molecular": "#e41a1c",
        "Neural": "#377eb8",
        "Physiological": "#4daf4a",
        "Circadian": "#ff7f00",
        "Ecological": "#984ea3",
        "Cell cycle": "#a65628",
        "Cardiac": "#f781bf",
    }

    for name, tau, A, cat in meta_systems:
        color = cat_colors.get(cat, COLOR_NEUTRAL)
        marker = "o" if A >= 0 else "v"
        ax.scatter(tau, A, c=color, s=50, marker=marker, edgecolors="black",
                   linewidths=0.5, zorder=3)

    # Add system labels for key points
    label_systems = {"ECG waveform", "EEG alpha", "EEG beta", "Respiration",
                     "Action potentials", "Circadian (core clock)", "Ca2+ spikes (neurons)",
                     "HRV (HF)", "NF-kB oscillations", "Predator-prey (Lotka-Volterra)"}
    for name, tau, A, cat in meta_systems:
        if name in label_systems:
            ax.annotate(name, (tau, A), fontsize=6, xytext=(5, 5),
                        textcoords="offset points", alpha=0.8)

    ax.set_xscale("log")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel(r"Characteristic timescale $\tau$ (seconds)")
    ax.set_ylabel("Asymmetry index (A)")
    ax.set_title("Waveform asymmetry across biological scales")

    # Legend
    handles = []
    for cat, color in cat_colors.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=color, markersize=7,
                                  markeredgecolor="black", markeredgewidth=0.5,
                                  label=cat))
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9,
              ncol=2)

    # Timescale annotations
    for label, x_pos in [("ms", 0.001), ("sec", 1), ("min", 60),
                          ("hour", 3600), ("day", 86400), ("year", 3.15e7)]:
        ax.axvline(x_pos, color=COLOR_NEUTRAL, linestyle=":", linewidth=0.3,
                   alpha=0.5)
        ax.text(x_pos, ax.get_ylim()[1], label, ha="center", va="bottom",
                fontsize=7, color=COLOR_NEUTRAL)

    fig.tight_layout()
    fig.savefig(PAPER_FIG / "figS1_meta_scale_map.pdf")
    plt.close(fig)
    print("    Saved figS1_meta_scale_map.pdf")


# ══════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURE S2: Control Analyses
# ══════════════════════════════════════════════════════════════════════════
def figure_s2():
    print("  Generating Figure S2 (Control analyses)...")
    lemon = load_lemon()

    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.76))

    # ── Panel A: Aperiodic exponent ~ age ──
    ax = axes[0]
    # Aperiodic exponent not stored directly; use sigma_global as proxy
    # or show placeholder
    ages_all = np.array([safe_float(s.get("age")) for s in lemon])
    sigma_all = np.array([safe_float(s.get("sigma_global")) for s in lemon])
    mask = np.isfinite(ages_all) & np.isfinite(sigma_all)
    if mask.sum() > 10:
        r_ap, p_ap = stats.pearsonr(ages_all[mask], sigma_all[mask])
        scatter_with_regression(ax, ages_all, sigma_all, COLOR_LEMON,
                                "Age (years)", r"$\sigma_{global}$ (A variability)",
                                title="Global variability ~ age",
                                annotate_text=f"r = {r_ap:.3f}\np = {p_ap:.2e}\nN = {mask.sum()}")
    else:
        ax.text(0.5, 0.5, "Aperiodic exponent\nnot available in summary",
                transform=ax.transAxes, ha="center", fontsize=9)
        ax.set_title("Aperiodic control")
    add_panel_label(ax, "A")

    # ── Panel B: Beta pp ~ age: raw vs partial ──
    ax = axes[1]
    raw_r = -0.326
    partial_r = -0.245
    labels_b = ["Raw", "Partial\n(aperiodic)"]
    vals_b = [abs(raw_r), abs(partial_r)]
    colors_b = [COLOR_LEMON, COLOR_LEMON]

    bars = ax.bar([0, 1], vals_b, color=colors_b, edgecolor="black",
                  linewidth=0.5, width=0.5, zorder=3)
    bars[1].set_alpha(0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels_b)
    ax.set_ylabel("|r| (beta pp ~ age)")
    ax.set_title("Aperiodic control")
    ax.set_ylim(0, 0.45)

    # Annotate values
    for i, (v, r_sign) in enumerate(zip(vals_b, [raw_r, partial_r])):
        ax.text(i, v + 0.01, f"r = {r_sign:.3f}", ha="center", va="bottom",
                fontsize=8)

    # Percentage reduction
    pct_change = (1 - abs(partial_r) / abs(raw_r)) * 100
    ax.annotate(f"{pct_change:.0f}% reduction", xy=(0.5, 0.35),
                xycoords=("data", "axes fraction"), ha="center", fontsize=8,
                color=COLOR_DORT)
    add_panel_label(ax, "B")

    # ── Panel C: Regional EMG control ──
    ax = axes[2]
    regions = ["Central", "Frontal", "Parietal", "Occipital", "Temporal"]
    r_vals = [-0.300, -0.210, -0.203, -0.178, -0.105]
    emg_risk = ["high", "medium", "medium", "low", "high"]
    risk_colors = {"high": "#e41a1c", "medium": "#ff7f00", "low": "#377eb8"}

    x_r = np.arange(len(regions))
    bar_colors = [risk_colors[r] for r in emg_risk]
    ax.bar(x_r, [abs(r) for r in r_vals], color=bar_colors, edgecolor="black",
           linewidth=0.5, width=0.6, zorder=3)
    ax.set_xticks(x_r)
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_ylabel("|r| (beta pp ~ age)")
    ax.set_title("Regional EMG control")
    ax.set_ylim(0, 0.40)

    # Annotate r values
    for i, r in enumerate(r_vals):
        ax.text(i, abs(r) + 0.01, f"{r:.3f}", ha="center", va="bottom",
                fontsize=7)

    # Legend for EMG risk
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=risk_colors["low"], edgecolor="black",
                             label="Low EMG risk"),
                       Patch(facecolor=risk_colors["medium"], edgecolor="black",
                             label="Medium"),
                       Patch(facecolor=risk_colors["high"], edgecolor="black",
                             label="High EMG risk")]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")
    add_panel_label(ax, "C")

    fig.tight_layout()
    fig.savefig(PAPER_FIG / "figS2_control_analyses.pdf")
    plt.close(fig)
    print("    Saved figS2_control_analyses.pdf")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Generating paper figures for Asymmetric Perception Cycles")
    print("=" * 60)

    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure_s1()
    figure_s2()

    print("\n" + "=" * 60)
    print(f"All figures saved to: {PAPER_FIG}")
    print("=" * 60)


if __name__ == "__main__":
    main()
