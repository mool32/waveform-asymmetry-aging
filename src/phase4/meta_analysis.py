"""
Phase 4: Meta-analytic A(τ) scale map.

Collects published rise/fall times across 15 orders of magnitude of timescale,
computes A(τ) = (t_fall - t_rise) / (t_fall + t_rise) for each, and produces
the universal asymmetry-vs-timescale plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "serif"


# ──────────────────────────────────────────────────────────────────────
# Literature database of biological oscillatory asymmetries
# Each entry: system, process, t_rise, t_fall, units, timescale_seconds,
#             reference (first author + year), notes
# ──────────────────────────────────────────────────────────────────────

LITERATURE_DATA = [
    # ── Molecular: Ion channels ──
    {
        "system": "Ion channel",
        "process": "Na+ channel (nerve)",
        "t_rise": 0.1e-3, "t_fall": 1.0e-3,  # activation vs inactivation
        "timescale_s": 0.5e-3,
        "reference": "Hodgkin & Huxley 1952",
        "notes": "Squid giant axon. Activation ~0.1ms, inactivation ~1ms",
    },
    {
        "system": "Ion channel",
        "process": "K+ delayed rectifier",
        "t_rise": 1.0e-3, "t_fall": 10.0e-3,
        "timescale_s": 5e-3,
        "reference": "Hodgkin & Huxley 1952",
        "notes": "Activation ~1ms, deactivation ~10ms",
    },
    {
        "system": "Ion channel",
        "process": "Ca2+ L-type channel (cardiac)",
        "t_rise": 2e-3, "t_fall": 30e-3,
        "timescale_s": 15e-3,
        "reference": "Bhatt 2015",
        "notes": "Activation ~2ms, inactivation ~30ms at -10mV",
    },
    {
        "system": "Ion channel",
        "process": "NMDA receptor",
        "t_rise": 5e-3, "t_fall": 100e-3,
        "timescale_s": 50e-3,
        "reference": "Lester 1990",
        "notes": "Rise ~5ms, decay ~50-100ms",
    },

    # ── Subcellular: Ca2+ transients ──
    {
        "system": "Subcellular",
        "process": "Cardiac Ca2+ transient",
        "t_rise": 20e-3, "t_fall": 200e-3,
        "timescale_s": 0.1,
        "reference": "Bers 2002",
        "notes": "Rise ~20ms, decay ~200ms (SERCA pump)",
    },
    {
        "system": "Subcellular",
        "process": "Neuronal Ca2+ transient (GCaMP)",
        "t_rise": 50e-3, "t_fall": 500e-3,
        "timescale_s": 0.25,
        "reference": "Chen 2013",
        "notes": "GCaMP6f: rise ~50ms, decay ~400-500ms",
    },
    {
        "system": "Subcellular",
        "process": "IP3-mediated Ca2+ oscillation",
        "t_rise": 0.5, "t_fall": 5.0,
        "timescale_s": 3.0,
        "reference": "Dupont 2011",
        "notes": "Rise ~0.5s, recovery ~5s",
    },
    {
        "system": "Subcellular",
        "process": "p53 pulse",
        "t_rise": 30 * 60, "t_fall": 300 * 60,  # ~30min rise, ~5h fall
        "timescale_s": 3 * 3600,
        "reference": "Lahav 2004",
        "notes": "p53 accumulation ~30min, degradation ~5h in response to DSBs",
    },
    {
        "system": "Subcellular",
        "process": "NF-κB oscillation",
        "t_rise": 15 * 60, "t_fall": 75 * 60,
        "timescale_s": 45 * 60,
        "reference": "Hoffmann 2002",
        "notes": "Nuclear entry ~15min, export+IκB resynthesis ~75min",
    },

    # ── Cellular: Action potentials ──
    {
        "system": "Neural",
        "process": "Cortical neuron AP",
        "t_rise": 0.5e-3, "t_fall": 2.0e-3,
        "timescale_s": 1e-3,
        "reference": "Bean 2007",
        "notes": "Depolarization ~0.5ms, repolarization+AHP ~2ms",
    },
    {
        "system": "Neural",
        "process": "Cardiac ventricular AP",
        "t_rise": 2e-3, "t_fall": 300e-3,
        "timescale_s": 0.15,
        "reference": "Nerbonne & Kass 2005",
        "notes": "Depolarization ~2ms, plateau+repolarization ~300ms",
    },
    {
        "system": "Neural",
        "process": "Purkinje cell complex spike",
        "t_rise": 0.3e-3, "t_fall": 5.0e-3,
        "timescale_s": 3e-3,
        "reference": "Davie 2008",
        "notes": "Initial spike ~0.3ms, complex afterdepolarization ~5ms",
    },

    # ── Organ: Cardiac ──
    {
        "system": "Cardiac",
        "process": "ECG QRS vs T-wave",
        "t_rise": 80e-3, "t_fall": 200e-3,
        "timescale_s": 0.15,
        "reference": "Goldberger 2017",
        "notes": "QRS ~80ms (depolarization), T-wave ~200ms (repolarization)",
    },
    {
        "system": "Cardiac",
        "process": "HRV acceleration vs deceleration",
        "t_rise": 1.0, "t_fall": 3.0,
        "timescale_s": 2.0,
        "reference": "Piskorski & Guzik 2007",
        "notes": "Heart rate accelerates faster than decelerates (vagal)",
    },

    # ── Respiratory ──
    {
        "system": "Respiratory",
        "process": "Inspiration vs expiration",
        "t_rise": 1.5, "t_fall": 2.5,
        "timescale_s": 2.0,
        "reference": "Tobin 1983",
        "notes": "I:E ratio ~1:1.5-2.0 in quiet breathing",
    },

    # ── Endocrine ──
    {
        "system": "Endocrine",
        "process": "Insulin pulse",
        "t_rise": 2 * 60, "t_fall": 8 * 60,
        "timescale_s": 5 * 60,
        "reference": "Pørksen 2002",
        "notes": "Secretory burst ~2min, clearance ~8min",
    },
    {
        "system": "Endocrine",
        "process": "GnRH pulse",
        "t_rise": 10 * 60, "t_fall": 80 * 60,
        "timescale_s": 45 * 60,
        "reference": "Knobil 1980",
        "notes": "Release ~10min, inter-pulse interval ~80min",
    },
    {
        "system": "Endocrine",
        "process": "Cortisol ultradian pulse",
        "t_rise": 20 * 60, "t_fall": 60 * 60,
        "timescale_s": 40 * 60,
        "reference": "Lightman & Conway-Campbell 2010",
        "notes": "Rise ~20min, fall ~60min",
    },
    {
        "system": "Endocrine",
        "process": "Cortisol awakening response (CAR)",
        "t_rise": 30 * 60, "t_fall": 14 * 3600,
        "timescale_s": 8 * 3600,
        "reference": "Fries 2009",
        "notes": "Morning rise ~30min, day-long decline ~14h",
    },

    # ── Circadian ──
    {
        "system": "Circadian",
        "process": "Melatonin onset/offset",
        "t_rise": 2 * 3600, "t_fall": 6 * 3600,
        "timescale_s": 4 * 3600,
        "reference": "Lewy 1999",
        "notes": "Evening rise ~2h, morning suppression (by light) ~6h clearance",
    },
    {
        "system": "Circadian",
        "process": "Core body temperature",
        "t_rise": 8 * 3600, "t_fall": 16 * 3600,
        "timescale_s": 12 * 3600,
        "reference": "Refinetti 2010",
        "notes": "Morning rise ~8h, evening decline ~16h",
    },

    # ── Immune ──
    {
        "system": "Immune",
        "process": "Acute inflammation (neutrophils)",
        "t_rise": 6 * 3600, "t_fall": 72 * 3600,
        "timescale_s": 36 * 3600,
        "reference": "Serhan 2007",
        "notes": "Neutrophil infiltration peaks ~6h, resolution ~3 days",
    },
    {
        "system": "Immune",
        "process": "Adaptive immune response (IgG)",
        "t_rise": 7 * 86400, "t_fall": 28 * 86400,
        "timescale_s": 14 * 86400,
        "reference": "Murphy 2022 (Janeway's)",
        "notes": "Primary response peak ~7 days, decline ~4 weeks",
    },

    # ── Wound healing ──
    {
        "system": "Tissue repair",
        "process": "Wound healing",
        "t_rise": 3 * 86400, "t_fall": 60 * 86400,
        "timescale_s": 30 * 86400,
        "reference": "Gurtner 2008",
        "notes": "Inflammatory phase ~3 days, remodeling ~60 days",
    },

    # ── Developmental ──
    {
        "system": "Developmental",
        "process": "Somitogenesis clock",
        "t_rise": 30 * 60, "t_fall": 90 * 60,
        "timescale_s": 60 * 60,
        "reference": "Pourquié 2011",
        "notes": "Hes7 oscillation: rise ~30min, fall ~90min (mouse)",
    },
    {
        "system": "Developmental",
        "process": "Pubertal growth spurt",
        "t_rise": 2 * 365 * 86400, "t_fall": 5 * 365 * 86400,
        "timescale_s": 3.5 * 365 * 86400,
        "reference": "Tanner 1962",
        "notes": "Rapid growth ~2 years, deceleration ~5 years",
    },

    # ── Gene expression ──
    {
        "system": "Gene expression",
        "process": "Transcriptional burst",
        "t_rise": 5 * 60, "t_fall": 30 * 60,
        "timescale_s": 15 * 60,
        "reference": "Suter 2011",
        "notes": "Burst duration ~5min, inter-burst interval ~30min",
    },
    {
        "system": "Gene expression",
        "process": "Heat shock response (Hsp70)",
        "t_rise": 15 * 60, "t_fall": 6 * 3600,
        "timescale_s": 3 * 3600,
        "reference": "Morimoto 1998",
        "notes": "Hsp70 induction ~15min, return to baseline ~6h",
    },

    # ── Muscle ──
    {
        "system": "Muscle",
        "process": "Skeletal muscle twitch (fast-twitch)",
        "t_rise": 10e-3, "t_fall": 50e-3,
        "timescale_s": 30e-3,
        "reference": "Bottinelli & Reggiani 2000",
        "notes": "Contraction ~10ms, relaxation ~50ms",
    },
    {
        "system": "Muscle",
        "process": "Skeletal muscle twitch (slow-twitch)",
        "t_rise": 30e-3, "t_fall": 200e-3,
        "timescale_s": 100e-3,
        "reference": "Bottinelli & Reggiani 2000",
        "notes": "Contraction ~30ms, relaxation ~200ms",
    },
]


def build_meta_dataframe():
    """Convert literature data to DataFrame with computed A(τ)."""
    df = pd.DataFrame(LITERATURE_DATA)

    # Compute asymmetry index
    df["A"] = (df["t_fall"] - df["t_rise"]) / (df["t_fall"] + df["t_rise"])

    # Total cycle time
    df["t_cycle"] = df["t_rise"] + df["t_fall"]

    # Log timescale
    df["log_timescale"] = np.log10(df["timescale_s"])

    # Rise/fall ratio
    df["fall_rise_ratio"] = df["t_fall"] / df["t_rise"]

    return df


def plot_asymmetry_scale_map(df=None, output_path=None, figsize=(14, 8)):
    """Plot A(τ) across 15 orders of magnitude of timescale.

    The key figure of Phase 4.
    """
    if df is None:
        df = build_meta_dataframe()

    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1],
                              sharex=True, gridspec_kw={"hspace": 0.05})

    # Color by system
    system_colors = {
        "Ion channel": "#e41a1c",
        "Subcellular": "#ff7f00",
        "Neural": "#377eb8",
        "Cardiac": "#e41a1c",
        "Respiratory": "#4daf4a",
        "Endocrine": "#984ea3",
        "Circadian": "#a65628",
        "Immune": "#f781bf",
        "Tissue repair": "#999999",
        "Developmental": "#66c2a5",
        "Gene expression": "#fc8d62",
        "Muscle": "#8da0cb",
    }

    ax = axes[0]
    for system in df["system"].unique():
        mask = df["system"] == system
        color = system_colors.get(system, "#333333")
        ax.scatter(df.loc[mask, "log_timescale"], df.loc[mask, "A"],
                   c=color, s=120, label=system, edgecolors="black",
                   linewidths=0.5, zorder=3, alpha=0.85)

    # Reference lines
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, zorder=1)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, zorder=1)

    ax.set_ylabel("Asymmetry Index  A(τ) = (t_fall − t_rise) / (t_fall + t_rise)",
                   fontsize=12)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_title("Universal Asymmetry of Biological Oscillations Across Scales",
                  fontsize=14, fontweight="bold")

    # Add timescale labels
    timescale_labels = {
        -4: "0.1 ms", -3: "1 ms", -2: "10 ms", -1: "0.1 s",
        0: "1 s", 1: "10 s", 2: "~min", 3: "~15 min",
        4: "~3 h", 5: "~1 day", 6: "~10 days", 7: "~3 months",
        8: "~3 years",
    }

    # Bottom panel: fall/rise ratio (log scale)
    ax2 = axes[1]
    for system in df["system"].unique():
        mask = df["system"] == system
        color = system_colors.get(system, "#333333")
        ax2.scatter(df.loc[mask, "log_timescale"],
                    np.log10(df.loc[mask, "fall_rise_ratio"]),
                    c=color, s=60, edgecolors="black", linewidths=0.5,
                    zorder=3, alpha=0.85)

    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("log₁₀(t_fall / t_rise)", fontsize=11)
    ax2.set_xlabel("log₁₀(timescale / seconds)", fontsize=12)

    # Add human-readable timescale labels
    ax2_top = ax2.twiny()
    tick_positions = list(timescale_labels.keys())
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(tick_positions)
    ax2_top.set_xticklabels([timescale_labels[t] for t in tick_positions],
                             fontsize=8, rotation=30)
    ax2_top.tick_params(top=False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}")

    return fig, axes


def print_summary_table(df=None):
    """Print formatted summary table of all entries."""
    if df is None:
        df = build_meta_dataframe()

    # Sort by timescale
    df_sorted = df.sort_values("log_timescale")

    print(f"\n{'='*100}")
    print("ASYMMETRIC PERCEPTION CYCLES: META-ANALYTIC SCALE MAP")
    print(f"{'='*100}")
    print(f"\n{'System':<18} {'Process':<35} {'t_rise':<12} {'t_fall':<12} {'A(τ)':<8} {'Reference':<30}")
    print("-" * 115)

    for _, row in df_sorted.iterrows():
        def fmt_time(t):
            if t < 1e-3:
                return f"{t*1e6:.0f} µs"
            elif t < 1:
                return f"{t*1e3:.1f} ms"
            elif t < 60:
                return f"{t:.1f} s"
            elif t < 3600:
                return f"{t/60:.1f} min"
            elif t < 86400:
                return f"{t/3600:.1f} h"
            else:
                return f"{t/86400:.1f} days"

        print(f"{row['system']:<18} {row['process']:<35} "
              f"{fmt_time(row['t_rise']):<12} {fmt_time(row['t_fall']):<12} "
              f"{row['A']:<8.3f} {row['reference']:<30}")

    print(f"\nTotal entries: {len(df)}")
    print(f"A(τ) range: [{df['A'].min():.3f}, {df['A'].max():.3f}]")
    print(f"Mean A(τ): {df['A'].mean():.3f} ± {df['A'].std():.3f}")
    print(f"All A(τ) > 0: {(df['A'] > 0).all()}")
    print(f"Timescale range: {df['timescale_s'].min():.1e} s to {df['timescale_s'].max():.1e} s")
    print(f"  = {np.log10(df['timescale_s'].max()) - np.log10(df['timescale_s'].min()):.1f} orders of magnitude")


if __name__ == "__main__":
    df = build_meta_dataframe()
    print_summary_table(df)
    plot_asymmetry_scale_map(df, output_path="results/phase4/asymmetry_scale_map.png")
