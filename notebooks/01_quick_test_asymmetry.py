"""
Quick test: verify A(τ) computation on synthetic signals.
Run this before touching real data to confirm the pipeline works.

Usage: python notebooks/01_quick_test_asymmetry.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
from src.utils.asymmetry import (
    find_cycles, asymmetry_index, compute_asymmetry_profile,
    bandpass_decompose, surrogate_test,
)


def make_asymmetric_oscillation(n_seconds=60, sfreq=250, freq=10.0,
                                 rise_fraction=0.3, noise_level=0.1):
    """Generate a signal with known asymmetry (fast rise, slow fall).

    rise_fraction: fraction of cycle spent rising (< 0.5 = fast rise, slow fall).
    """
    t = np.arange(0, n_seconds, 1.0 / sfreq)
    period = 1.0 / freq
    phase = (t % period) / period  # 0 to 1 within each cycle

    # Piecewise linear sawtooth with asymmetry
    signal = np.where(
        phase < rise_fraction,
        phase / rise_fraction,  # rise
        1.0 - (phase - rise_fraction) / (1.0 - rise_fraction),  # fall
    )
    signal = signal - 0.5  # center at zero

    # Add noise
    signal += noise_level * np.random.randn(len(signal))

    expected_A = (1.0 - rise_fraction - rise_fraction) / 1.0  # = 1 - 2*rise_fraction
    # More precisely: t_rise/T = rise_fraction, t_fall/T = 1-rise_fraction
    # A = (t_fall - t_rise)/(t_fall + t_rise) = (1-2*rise_fraction)/1 = 1-2*rise_fraction

    return t, signal, expected_A


def test_basic_asymmetry():
    """Test that A(τ) correctly measures known asymmetry."""
    print("="*60)
    print("TEST 1: Basic asymmetry measurement")
    print("="*60)

    for rise_frac in [0.2, 0.3, 0.5, 0.7]:
        t, signal, expected_A = make_asymmetric_oscillation(
            rise_fraction=rise_frac, noise_level=0.05
        )

        cycles = find_cycles(signal, min_prominence=0.2)
        A, A_values = asymmetry_index(cycles)

        print(f"  rise_fraction={rise_frac:.1f}: expected A={expected_A:.3f}, "
              f"measured A={A:.3f}, n_cycles={len(cycles)}, "
              f"{'PASS' if abs(A - expected_A) < 0.15 else 'FAIL'}")


def test_multiband_profile():
    """Test asymmetry profile across multiple frequency bands."""
    print("\n" + "="*60)
    print("TEST 2: Multi-band asymmetry profile")
    print("="*60)

    sfreq = 250
    t = np.arange(0, 120, 1.0 / sfreq)

    # Create composite signal: asymmetric at all bands
    signal = np.zeros_like(t)
    bands_info = {
        "theta": (5, 0.25),   # 5 Hz, fast rise
        "alpha": (10, 0.30),  # 10 Hz, moderately fast rise
        "beta": (20, 0.35),   # 20 Hz, slightly asymmetric
    }

    for freq, rise_frac in bands_info.values():
        period = 1.0 / freq
        phase = (t % period) / period
        component = np.where(
            phase < rise_frac,
            phase / rise_frac,
            1.0 - (phase - rise_frac) / (1.0 - rise_frac),
        ) - 0.5
        signal += component

    signal += 0.1 * np.random.randn(len(signal))

    # Compute profile
    test_bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
    }
    profile = compute_asymmetry_profile(signal, sfreq, test_bands, method="bandpass")

    for name, info in profile.items():
        print(f"  {name}: A={info['A']:.3f}, n_cycles={info['n_cycles']}")

    # All should be positive
    all_positive = all(info["A"] > 0 for info in profile.values()
                       if not np.isnan(info["A"]))
    print(f"  All A > 0: {all_positive} {'PASS' if all_positive else 'FAIL'}")


def test_surrogate():
    """Test that surrogate testing works (real signal > surrogates)."""
    print("\n" + "="*60)
    print("TEST 3: Surrogate test")
    print("="*60)

    t, signal, _ = make_asymmetric_oscillation(
        n_seconds=60, sfreq=250, freq=10, rise_fraction=0.25, noise_level=0.1
    )

    test_bands = {"alpha": (8, 13)}
    results = surrogate_test(signal, 250, test_bands, n_surrogates=100)

    for band, res in results.items():
        print(f"  {band}: A_obs={res['A_observed']:.3f}, "
              f"p={res['p_value']:.4f}, z={res['z_score']:.1f}")
        print(f"  Surrogate mean={np.mean(res['A_surrogates']):.3f} ± "
              f"{np.std(res['A_surrogates']):.3f}")
        print(f"  {'PASS' if res['p_value'] < 0.05 else 'FAIL'}")


def test_symmetric_signal():
    """Test that symmetric signal gives A ≈ 0."""
    print("\n" + "="*60)
    print("TEST 4: Symmetric signal (sine wave) should give A ≈ 0")
    print("="*60)

    sfreq = 250
    t = np.arange(0, 60, 1.0 / sfreq)
    signal = np.sin(2 * np.pi * 10 * t) + 0.05 * np.random.randn(len(t))

    test_bands = {"alpha": (8, 13)}
    profile = compute_asymmetry_profile(signal, sfreq, test_bands)

    A = profile["alpha"]["A"]
    print(f"  Sine wave A(alpha) = {A:.4f}")
    print(f"  {'PASS' if abs(A) < 0.1 else 'FAIL'} (expected ≈ 0)")


def test_phase4_meta():
    """Quick test of Phase 4 meta-analysis data."""
    print("\n" + "="*60)
    print("TEST 5: Phase 4 meta-analysis data")
    print("="*60)

    from src.phase4.meta_analysis import build_meta_dataframe, print_summary_table

    df = build_meta_dataframe()
    print(f"  Entries: {len(df)}")
    print(f"  All A > 0: {(df['A'] > 0).all()}")
    print(f"  A range: [{df['A'].min():.3f}, {df['A'].max():.3f}]")
    print(f"  Timescale range: {df['timescale_s'].min():.1e} to {df['timescale_s'].max():.1e} s")
    print(f"  Orders of magnitude: {np.log10(df['timescale_s'].max()) - np.log10(df['timescale_s'].min()):.1f}")

    print_summary_table(df)


if __name__ == "__main__":
    np.random.seed(42)

    test_basic_asymmetry()
    test_multiband_profile()
    test_surrogate()
    test_symmetric_signal()
    test_phase4_meta()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
