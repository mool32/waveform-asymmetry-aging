"""
Phase 1: Visualization for LEMON analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 11


def plot_asymmetry_profile(eeg_profile, hrv_profile=None, subject_id="",
                            output_path=None):
    """Plot A(τ) profile for one subject."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # EEG
    eeg_bands = sorted(eeg_profile.keys(),
                        key=lambda b: eeg_profile[b].get("center_freq", 0))
    eeg_freqs = [eeg_profile[b]["center_freq"] for b in eeg_bands]
    eeg_A = [eeg_profile[b].get("A_mean", eeg_profile[b].get("A", np.nan))
             for b in eeg_bands]

    ax.plot(eeg_freqs, eeg_A, "o-", color="#377eb8", markersize=8,
            linewidth=2, label="EEG A(τ)")

    # HRV
    if hrv_profile:
        hrv_bands = sorted(hrv_profile.keys(),
                            key=lambda b: hrv_profile[b].get("center_freq", 0))
        hrv_freqs = [hrv_profile[b]["center_freq"] for b in hrv_bands]
        hrv_A = [hrv_profile[b]["A"] for b in hrv_bands]
        ax.plot(hrv_freqs, hrv_A, "s-", color="#e41a1c", markersize=8,
                linewidth=2, label="HRV A(τ)")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("A(τ)")
    ax.set_xscale("log")
    ax.legend()
    ax.set_title(f"Asymmetry Profile — {subject_id}" if subject_id else "Asymmetry Profile")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_age_regression(ages, values, band_name, value_label="A(τ)",
                         output_path=None):
    """Scatter + regression of a metric vs age."""
    fig, ax = plt.subplots(figsize=(8, 5))

    valid = ~np.isnan(values) & ~np.isnan(ages)
    x, y = np.array(ages)[valid], np.array(values)[valid]

    ax.scatter(x, y, alpha=0.5, s=40, color="#377eb8", edgecolors="white",
               linewidths=0.5)

    # Regression line
    if len(x) > 5:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2, alpha=0.7)

        from scipy.stats import pearsonr
        r, pval = pearsonr(x, y)
        ax.text(0.05, 0.95, f"r = {r:.3f}, p = {pval:.1e}\nslope = {z[0]:.4f}/year",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Age (years)")
    ax.set_ylabel(value_label)
    ax.set_title(f"{value_label} vs Age — {band_name}")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_prediction_3_comparison(rho_values, mean_eeg_A, mean_hrv_A, ages,
                                  output_path=None):
    """Plot comparing age slopes of ρ_cross vs individual A values."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, values, label, color in [
        (axes[0], mean_eeg_A, "Mean EEG A(τ)", "#377eb8"),
        (axes[1], mean_hrv_A, "Mean HRV A(τ)", "#e41a1c"),
        (axes[2], rho_values, "ρ_cross (EEG-HRV)", "#4daf4a"),
    ]:
        valid = ~np.isnan(values) & ~np.isnan(ages)
        x, y = np.array(ages)[valid], np.array(values)[valid]
        ax.scatter(x, y, alpha=0.5, s=30, color=color, edgecolors="white")

        if len(x) > 5:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "k-", linewidth=2, alpha=0.7)

            from scipy.stats import pearsonr
            r, pval = pearsonr(x, y)
            ax.set_title(f"{label}\nslope={z[0]:.4f}/yr, r={r:.3f}", fontsize=10)
        else:
            ax.set_title(label)

        ax.set_xlabel("Age")
        ax.set_ylabel(label)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)

    plt.suptitle("Prediction 3: ρ_cross should degrade faster than individual A",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_surrogate_comparison(surrogate_results, output_path=None):
    """Plot observed A vs surrogate distribution for each band."""
    n_bands = len(surrogate_results)
    fig, axes = plt.subplots(1, n_bands, figsize=(3 * n_bands, 4))
    if n_bands == 1:
        axes = [axes]

    for ax, (band, res) in zip(axes, surrogate_results.items()):
        surr = res["A_surrogates"]
        obs = res["A_observed"]

        if len(surr) > 0 and not np.isnan(obs):
            ax.hist(surr, bins=30, alpha=0.7, color="#aaaaaa", edgecolor="white")
            ax.axvline(obs, color="red", linewidth=2, label=f"Observed: {obs:.3f}")
            ax.set_title(f"{band}\np={res['p_value']:.3f}, z={res['z_score']:.1f}",
                          fontsize=9)
        else:
            ax.set_title(f"{band}\nN/A")

        ax.set_xlabel("A(τ)")
        ax.legend(fontsize=7)

    plt.suptitle("Surrogate Test: Observed A(τ) vs Phase-Randomized", fontsize=11)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
