"""
Core asymmetry index A(τ) computation.

The asymmetry index measures the temporal asymmetry of oscillatory cycles:
    A(τ) = (t_fall - t_rise) / (t_fall + t_rise)

where t_rise = trough-to-peak time, t_fall = peak-to-trough time.

A > 0: "fast up, slow down" — the healthy biological pattern
A = 0: symmetric oscillation
A < 0: "slow up, fast down" — potentially pathological
"""

import numpy as np
from scipy.signal import find_peaks, hilbert
import pywt


def wavelet_decompose(signal, sfreq, scales_hz, wavelet="cmor1.5-1.0"):
    """Decompose signal into frequency bands using continuous wavelet transform.

    Parameters
    ----------
    signal : array, shape (n_samples,)
        Input time series.
    sfreq : float
        Sampling frequency in Hz.
    scales_hz : dict
        Mapping of band name -> (low_hz, high_hz).
        E.g. {"alpha": (8, 13), "beta": (13, 30)}
    wavelet : str
        Wavelet name for pywt.cwt.

    Returns
    -------
    bands : dict
        Mapping of band name -> filtered signal (real part of CWT).
    """
    bands = {}
    for name, (f_low, f_high) in scales_hz.items():
        f_center = (f_low + f_high) / 2
        # CWT scale = sfreq / (2 * pi * f_center) for cmor wavelet
        # More precisely: scale = central_freq * sfreq / f_center
        central_freq = pywt.central_frequency(wavelet)
        scale = central_freq * sfreq / f_center
        coeffs, _ = pywt.cwt(signal, [scale], wavelet, sampling_period=1.0 / sfreq)
        bands[name] = np.real(coeffs[0])
    return bands


def bandpass_decompose(signal, sfreq, scales_hz, order=4):
    """Decompose signal into frequency bands using Butterworth bandpass filters.

    More suitable for asymmetry measurement than CWT because it preserves
    waveform shape better (linear phase when using filtfilt).

    Parameters
    ----------
    signal : array, shape (n_samples,)
    sfreq : float
    scales_hz : dict
        Band name -> (low_hz, high_hz).
    order : int
        Butterworth filter order.

    Returns
    -------
    bands : dict
        Band name -> filtered signal.
    """
    from scipy.signal import butter, filtfilt

    bands = {}
    nyq = sfreq / 2.0
    for name, (f_low, f_high) in scales_hz.items():
        low = f_low / nyq
        high = min(f_high / nyq, 0.99)
        if low <= 0:
            # Lowpass only
            b, a = butter(order, high, btype="low")
        else:
            b, a = butter(order, [low, high], btype="band")
        bands[name] = filtfilt(b, a, signal)
    return bands


def find_cycles(signal, min_prominence=None):
    """Find oscillatory cycles (trough-peak-trough) in a signal.

    Parameters
    ----------
    signal : array, shape (n_samples,)
        Band-filtered signal.
    min_prominence : float or None
        Minimum peak/trough prominence. If None, set to 0.5 * std(signal).

    Returns
    -------
    cycles : list of dict
        Each dict has keys:
        - 'trough1_idx': index of first trough
        - 'peak_idx': index of peak
        - 'trough2_idx': index of second trough
        - 't_rise': samples from trough1 to peak
        - 't_fall': samples from peak to trough2
        - 'amplitude': peak value - mean of troughs
    """
    if min_prominence is None:
        min_prominence = 0.5 * np.std(signal)

    # Find peaks and troughs
    peaks, peak_props = find_peaks(signal, prominence=min_prominence)
    troughs, trough_props = find_peaks(-signal, prominence=min_prominence)

    if len(peaks) < 1 or len(troughs) < 2:
        return []

    cycles = []
    for i in range(len(peaks)):
        peak_idx = peaks[i]
        # Find nearest trough before peak
        pre_troughs = troughs[troughs < peak_idx]
        if len(pre_troughs) == 0:
            continue
        trough1_idx = pre_troughs[-1]

        # Find nearest trough after peak
        post_troughs = troughs[troughs > peak_idx]
        if len(post_troughs) == 0:
            continue
        trough2_idx = post_troughs[0]

        t_rise = peak_idx - trough1_idx
        t_fall = trough2_idx - peak_idx

        if t_rise == 0 or t_fall == 0:
            continue

        amplitude = signal[peak_idx] - 0.5 * (signal[trough1_idx] + signal[trough2_idx])

        cycles.append({
            "trough1_idx": trough1_idx,
            "peak_idx": peak_idx,
            "trough2_idx": trough2_idx,
            "t_rise": t_rise,
            "t_fall": t_fall,
            "amplitude": amplitude,
        })

    return cycles


def find_cycles_hilbert(signal, sfreq):
    """Find oscillatory cycles using Hilbert transform phase.

    Uses instantaneous phase to identify cycle boundaries (trough-to-trough),
    then measures rise time (trough→peak) and fall time (peak→trough) using
    the actual signal extrema within each cycle.

    Parameters
    ----------
    signal : array, shape (n_samples,)
        Band-filtered signal.
    sfreq : float
        Sampling frequency.

    Returns
    -------
    cycles : list of dict
        Each with 't_rise' and 't_fall' in samples.
    """
    analytic = hilbert(signal)
    phase = np.angle(analytic)

    # Find zero crossings of phase (wraps from π to -π = cycle boundary)
    phase_diff = np.diff(phase)
    # Phase wraps: large negative jump means crossing from π to -π
    cycle_starts = np.where(phase_diff < -np.pi)[0] + 1

    if len(cycle_starts) < 2:
        return []

    cycles = []
    for i in range(len(cycle_starts) - 1):
        start = cycle_starts[i]
        end = cycle_starts[i + 1]

        if end - start < 3:
            continue

        cycle_signal = signal[start:end]

        # Peak = max within cycle, trough = min
        peak_local = np.argmax(cycle_signal)
        trough_local = np.argmin(cycle_signal)

        # We want trough → peak → trough pattern
        # If trough before peak: t_rise = peak - trough_start
        # But cycle boundaries are at troughs (phase = -π)
        # So: rise = start → peak, fall = peak → end
        t_rise = peak_local
        t_fall = (end - start) - peak_local

        if t_rise < 1 or t_fall < 1:
            continue

        amplitude = np.max(cycle_signal) - np.min(cycle_signal)

        cycles.append({
            "t_rise": t_rise,
            "t_fall": t_fall,
            "amplitude": amplitude,
            "start_idx": start,
            "end_idx": end,
        })

    return cycles


def find_cycles_zerocross(signal):
    """Find cycles using zero-crossing method.

    Identifies cycles as zero-crossing-up → peak → zero-crossing-down → trough → next zero-up.
    Rise time = time from ascending zero-crossing to peak.
    Fall time = time from peak to next descending zero-crossing.

    More robust than Hilbert for real noisy signals.
    """
    # Find ascending zero crossings
    sign = np.sign(signal)
    sign_change = np.diff(sign)
    asc_zeros = np.where(sign_change > 0)[0]  # crossing from neg to pos
    desc_zeros = np.where(sign_change < 0)[0]  # crossing from pos to neg

    if len(asc_zeros) < 2 or len(desc_zeros) < 1:
        return []

    cycles = []
    for i in range(len(asc_zeros) - 1):
        rise_start = asc_zeros[i]
        next_rise = asc_zeros[i + 1]

        # Find descending zero-crossing between these two ascending ones
        desc_between = desc_zeros[(desc_zeros > rise_start) & (desc_zeros < next_rise)]
        if len(desc_between) == 0:
            continue
        fall_start = desc_between[0]

        # Peak = max between ascending zero and descending zero
        peak_idx = rise_start + np.argmax(signal[rise_start:fall_start + 1])

        # Rise time: ascending zero → peak
        t_rise = peak_idx - rise_start
        # Fall time: peak → descending zero
        t_fall = fall_start - peak_idx

        if t_rise < 1 or t_fall < 1:
            continue

        amplitude = signal[peak_idx] - np.min(signal[fall_start:next_rise + 1])

        cycles.append({
            "t_rise": t_rise,
            "t_fall": t_fall,
            "amplitude": amplitude,
            "peak_idx": peak_idx,
        })

    return cycles


def asymmetry_index(cycles):
    """Compute A(τ) from a list of cycles.

    Returns median of (t_fall - t_rise) / (t_fall + t_rise) across cycles.

    Parameters
    ----------
    cycles : list of dict
        Output of find_cycles().

    Returns
    -------
    A : float
        Asymmetry index. Positive = fast rise, slow fall (healthy).
    A_values : array
        Per-cycle asymmetry values.
    """
    if len(cycles) == 0:
        return np.nan, np.array([])

    A_values = np.array([
        (c["t_fall"] - c["t_rise"]) / (c["t_fall"] + c["t_rise"])
        for c in cycles
    ])

    return np.median(A_values), A_values


def compute_asymmetry_profile(signal, sfreq, scales_hz, method="bandpass",
                               min_prominence=None, cycle_method="hilbert"):
    """Compute A(τ) profile across multiple frequency bands.

    Parameters
    ----------
    signal : array, shape (n_samples,)
    sfreq : float
    scales_hz : dict
        Band name -> (low_hz, high_hz).
    method : str
        'bandpass' or 'wavelet' for signal decomposition.
    min_prominence : float or None
    cycle_method : str
        'hilbert' (robust, default) or 'peaks' (original peak/trough detection).

    Returns
    -------
    profile : dict
        Band name -> {
            'A': float (median asymmetry),
            'A_values': array (per-cycle values),
            'n_cycles': int,
            'center_freq': float
        }
    """
    if method == "bandpass":
        bands = bandpass_decompose(signal, sfreq, scales_hz)
    else:
        bands = wavelet_decompose(signal, sfreq, scales_hz)

    profile = {}
    for name, band_signal in bands.items():
        if cycle_method == "hilbert":
            cycles = find_cycles_hilbert(band_signal, sfreq)
        elif cycle_method == "zerocross":
            cycles = find_cycles_zerocross(band_signal)
        else:
            cycles = find_cycles(band_signal, min_prominence=min_prominence)
        A, A_values = asymmetry_index(cycles)
        f_low, f_high = scales_hz[name]
        profile[name] = {
            "A": A,
            "A_values": A_values,
            "n_cycles": len(cycles),
            "center_freq": (f_low + f_high) / 2,
        }

    return profile



# ──────────────────────────────────────────────────────────────────────
# Broadband Peak-Trough Asymmetry (Cole & Voytek 2017 style)
# ──────────────────────────────────────────────────────────────────────

def find_broadband_cycles(signal, sfreq, f_low=1.0, f_high=45.0,
                           min_amplitude_percentile=25):
    """Find oscillatory cycles in broadband-filtered signal.

    Cole-Voytek style: filter broadband to preserve waveform shape,
    detect cycles as trough→peak→trough using extrema, then classify
    each cycle by its instantaneous frequency.

    Parameters
    ----------
    signal : array, shape (n_samples,)
        Raw or minimally filtered signal.
    sfreq : float
        Sampling frequency.
    f_low, f_high : float
        Broadband filter bounds.
    min_amplitude_percentile : float
        Reject cycles with amplitude below this percentile.

    Returns
    -------
    cycles : list of dict
        Each dict has:
        - 't_rise': rise time in samples
        - 't_fall': fall time in samples
        - 'amplitude': peak - mean(troughs)
        - 'frequency': 1 / cycle_duration in Hz
        - 'rise_fraction': t_rise / (t_rise + t_fall)  — the PTA metric
        - 'A': (t_fall - t_rise) / (t_fall + t_rise)
        - 'peak_idx', 'trough1_idx', 'trough2_idx': absolute indices
        - 'peak_sharpness': sharpness of peak (2nd derivative)
        - 'trough_sharpness': sharpness of trough
    """
    from scipy.signal import butter, filtfilt

    # Broadband filter — preserves waveform shape
    nyq = sfreq / 2.0
    b, a = butter(2, [f_low / nyq, min(f_high / nyq, 0.99)], btype="band")
    filtered = filtfilt(b, a, signal)

    # Find all peaks and troughs with generous sensitivity
    std = np.std(filtered)
    peaks, _ = find_peaks(filtered, prominence=0.1 * std,
                          distance=int(sfreq / f_high / 2))
    troughs, _ = find_peaks(-filtered, prominence=0.1 * std,
                            distance=int(sfreq / f_high / 2))

    if len(peaks) < 1 or len(troughs) < 2:
        return []

    # Build trough-peak-trough cycles
    cycles = []
    for i in range(len(peaks)):
        peak_idx = peaks[i]

        pre_troughs = troughs[troughs < peak_idx]
        post_troughs = troughs[troughs > peak_idx]
        if len(pre_troughs) == 0 or len(post_troughs) == 0:
            continue

        trough1 = pre_troughs[-1]
        trough2 = post_troughs[0]

        t_rise = peak_idx - trough1
        t_fall = trough2 - peak_idx
        t_total = t_rise + t_fall

        if t_rise < 2 or t_fall < 2:
            continue

        freq = sfreq / t_total
        if freq < f_low or freq > f_high:
            continue

        amplitude = filtered[peak_idx] - 0.5 * (
            filtered[trough1] + filtered[trough2]
        )

        # Peak and trough sharpness (2nd derivative proxy)
        # Sharpness = how pointed the extremum is
        if peak_idx > 0 and peak_idx < len(filtered) - 1:
            peak_sharp = -(filtered[peak_idx - 1] - 2 * filtered[peak_idx]
                           + filtered[peak_idx + 1])
        else:
            peak_sharp = np.nan

        if trough1 > 0 and trough1 < len(filtered) - 1:
            trough_sharp = (filtered[trough1 - 1] - 2 * filtered[trough1]
                            + filtered[trough1 + 1])
        else:
            trough_sharp = np.nan

        cycles.append({
            "t_rise": t_rise,
            "t_fall": t_fall,
            "amplitude": amplitude,
            "frequency": freq,
            "rise_fraction": t_rise / t_total,
            "A": (t_fall - t_rise) / t_total,
            "peak_idx": peak_idx,
            "trough1_idx": trough1,
            "trough2_idx": trough2,
            "peak_sharpness": peak_sharp,
            "trough_sharpness": trough_sharp,
        })

    if not cycles:
        return []

    # Filter by amplitude percentile
    amplitudes = np.array([c["amplitude"] for c in cycles])
    amp_threshold = np.percentile(amplitudes, min_amplitude_percentile)
    cycles = [c for c in cycles if c["amplitude"] >= amp_threshold]

    return cycles


def broadband_asymmetry_by_band(cycles, band_edges=None):
    """Classify broadband cycles by frequency and compute A per band.

    Parameters
    ----------
    cycles : list of dict
        Output of find_broadband_cycles.
    band_edges : dict or None
        Band name -> (low_hz, high_hz). Default: standard EEG bands.

    Returns
    -------
    profile : dict
        Band name -> {
            'A_median': float,
            'A_mean': float,
            'A_std': float,
            'rise_fraction_mean': float,
            'peak_sharpness_mean': float,
            'trough_sharpness_mean': float,
            'n_cycles': int,
            'A_values': array,
        }
    """
    if band_edges is None:
        band_edges = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "low_gamma": (30, 45),
        }

    profile = {}
    for band_name, (f_lo, f_hi) in band_edges.items():
        band_cycles = [c for c in cycles if f_lo <= c["frequency"] < f_hi]

        if len(band_cycles) < 5:
            profile[band_name] = {
                "A_median": np.nan, "A_mean": np.nan, "A_std": np.nan,
                "rise_fraction_mean": np.nan,
                "peak_sharpness_mean": np.nan,
                "trough_sharpness_mean": np.nan,
                "n_cycles": len(band_cycles),
                "A_values": np.array([]),
                "center_freq": (f_lo + f_hi) / 2,
            }
            continue

        A_vals = np.array([c["A"] for c in band_cycles])
        rf_vals = np.array([c["rise_fraction"] for c in band_cycles])
        ps_vals = np.array([c["peak_sharpness"] for c in band_cycles
                            if not np.isnan(c["peak_sharpness"])])
        ts_vals = np.array([c["trough_sharpness"] for c in band_cycles
                            if not np.isnan(c["trough_sharpness"])])

        profile[band_name] = {
            "A_median": np.median(A_vals),
            "A_mean": np.mean(A_vals),
            "A_std": np.std(A_vals),
            "rise_fraction_mean": np.mean(rf_vals),
            "peak_sharpness_mean": np.mean(ps_vals) if len(ps_vals) > 0 else np.nan,
            "trough_sharpness_mean": np.mean(ts_vals) if len(ts_vals) > 0 else np.nan,
            "sharpness_ratio": (np.mean(ps_vals) / (np.mean(ts_vals) + 1e-10)
                                if len(ps_vals) > 0 and len(ts_vals) > 0 else np.nan),
            "n_cycles": len(band_cycles),
            "A_values": A_vals,
            "center_freq": (f_lo + f_hi) / 2,
        }

    return profile


# ──────────────────────────────────────────────────────────────────────
# Spatial variability metrics
# ──────────────────────────────────────────────────────────────────────

# Standard EEG region definitions (10-20 system)
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


def _relative_band_power(signal, sfreq, f_low, f_high):
    """Compute relative power in a frequency band via Welch PSD."""
    from scipy.signal import welch
    nperseg = min(int(4 * sfreq), len(signal))
    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)
    total_power = np.trapz(psd, freqs)
    band_mask = (freqs >= f_low) & (freqs <= f_high)
    band_power = np.trapz(psd[band_mask], freqs[band_mask])
    return band_power / (total_power + 1e-20)


def compute_spatial_asymmetry(channel_data, ch_names, sfreq,
                               regions=None, f_low=1.0, f_high=45.0,
                               min_cycles=30):
    """Compute per-channel broadband PTA with power-thresholding and smart aggregation.

    Key improvements over naive all-channel median:
    1. Power threshold: per band, only include channels where relative power
       in that band exceeds the median across channels (band-relevant channels only)
    2. Min cycles filter: exclude channels with < min_cycles in a band
    3. Multiple aggregation metrics: trimmed_mean, prop_positive, sigma_A

    Parameters
    ----------
    channel_data : array, shape (n_channels, n_samples)
    ch_names : list of str
    sfreq : float
    regions : dict or None
    f_low, f_high : float
    min_cycles : int
        Minimum cycles per channel per band to be included.

    Returns
    -------
    per_channel : dict
        ch_name -> broadband asymmetry profile (all cycles, unfiltered).
    per_region : dict
        region_name -> per_band -> aggregated stats (power-filtered).
    spatial_stats : dict
        Per-band: sigma_A, prop_positive, trimmed_mean_A, n_included,
        plus global sigma_A.
    """
    if regions is None:
        regions = EEG_REGIONS

    band_edges = {
        "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
        "beta": (13, 30), "low_gamma": (30, 45),
    }

    # ── Step 1: Per-channel broadband PTA ──
    per_channel = {}
    for i, ch in enumerate(ch_names):
        try:
            cycles = find_broadband_cycles(
                channel_data[i], sfreq, f_low=f_low, f_high=f_high
            )
            profile = broadband_asymmetry_by_band(cycles, band_edges)
            per_channel[ch] = profile
        except Exception:
            pass

    if not per_channel:
        return {}, {}, {}

    # ── Step 2: Compute relative band power per channel ──
    band_power = {}  # band -> {ch: relative_power}
    for band, (bl, bh) in band_edges.items():
        bp = {}
        for i, ch in enumerate(ch_names):
            if ch in per_channel:
                bp[ch] = _relative_band_power(channel_data[i], sfreq, bl, bh)
        band_power[band] = bp

    # ── Step 3: Power-thresholded channel selection per band ──
    # For each band, include only channels above median relative power
    included_channels = {}  # band -> list of ch_names
    for band in band_edges:
        bp = band_power[band]
        if not bp:
            included_channels[band] = []
            continue
        power_vals = np.array(list(bp.values()))
        threshold = np.median(power_vals)
        included = [ch for ch, pw in bp.items()
                    if pw >= threshold
                    and ch in per_channel
                    and per_channel[ch][band]["n_cycles"] >= min_cycles]
        included_channels[band] = included

    # ── Step 4: Smart aggregation per band ──
    from scipy.stats import trim_mean

    spatial_stats = {"sigma_A_per_band": {}, "range_A_per_band": {},
                     "prop_positive_per_band": {}, "trimmed_mean_A_per_band": {},
                     "mean_A_per_band": {}, "n_included_per_band": {}}

    for band in band_edges:
        chs = included_channels[band]
        A_vals = [per_channel[ch][band]["A_mean"] for ch in chs
                  if not np.isnan(per_channel[ch][band]["A_mean"])]
        A_arr = np.array(A_vals)

        n = len(A_arr)
        spatial_stats["n_included_per_band"][band] = n

        if n >= 3:
            spatial_stats["mean_A_per_band"][band] = float(np.mean(A_arr))
            spatial_stats["trimmed_mean_A_per_band"][band] = float(
                trim_mean(A_arr, proportiontocut=0.1)
            )
            spatial_stats["prop_positive_per_band"][band] = float(
                np.mean(A_arr > 0)
            )
            spatial_stats["sigma_A_per_band"][band] = float(np.std(A_arr))
            spatial_stats["range_A_per_band"][band] = float(
                np.max(A_arr) - np.min(A_arr)
            )
        else:
            for key in ["mean_A_per_band", "trimmed_mean_A_per_band",
                        "prop_positive_per_band", "sigma_A_per_band",
                        "range_A_per_band"]:
                spatial_stats[key][band] = np.nan

    # Global σ(A) across all included channels and bands
    all_included_A = []
    for band in band_edges:
        for ch in included_channels[band]:
            v = per_channel[ch][band]["A_mean"]
            if not np.isnan(v):
                all_included_A.append(v)
    spatial_stats["sigma_A"] = float(np.std(all_included_A)) if len(all_included_A) >= 3 else np.nan

    # ── Step 5: Per-region aggregation (power-filtered) ──
    per_region = {}
    for region_name, region_chs in regions.items():
        region_bands = {}
        for band in band_edges:
            # Channels in this region AND included for this band
            matched = [ch for ch in region_chs
                       if ch in included_channels.get(band, [])]
            A_vals = [per_channel[ch][band]["A_mean"] for ch in matched
                      if not np.isnan(per_channel[ch][band]["A_mean"])]
            n = len(A_vals)
            if n >= 2:
                A_arr = np.array(A_vals)
                region_bands[band] = {
                    "A_mean": float(np.mean(A_arr)),
                    "A_std": float(np.std(A_arr)),
                    "prop_positive": float(np.mean(A_arr > 0)),
                    "n_channels": n,
                }
            else:
                region_bands[band] = {
                    "A_mean": np.nan, "A_std": np.nan,
                    "prop_positive": np.nan, "n_channels": n,
                }

        per_region[region_name] = {
            "per_band": region_bands,
            "n_channels_total": len([ch for ch in region_chs if ch in per_channel]),
        }

    return per_channel, per_region, spatial_stats


def surrogate_test(signal, sfreq, scales_hz, n_surrogates=1000,
                   method="bandpass", min_prominence=None):
    """Test whether A(τ) is significantly different from phase-randomized surrogates.

    Phase randomization preserves the power spectrum but destroys waveform asymmetry.

    Parameters
    ----------
    signal : array
    sfreq : float
    scales_hz : dict
    n_surrogates : int
    method : str
    min_prominence : float or None

    Returns
    -------
    results : dict
        Band name -> {
            'A_observed': float,
            'A_surrogates': array of surrogate A values,
            'p_value': float (one-sided: A_obs > surrogates),
            'z_score': float
        }
    """
    # Compute observed profile
    observed = compute_asymmetry_profile(signal, sfreq, scales_hz, method,
                                         min_prominence)

    results = {name: {"A_observed": v["A"], "A_surrogates": []}
               for name, v in observed.items()}

    # Generate surrogates via phase randomization (IAAFT-like)
    n = len(signal)
    fft_signal = np.fft.rfft(signal)
    amplitudes = np.abs(fft_signal)

    for _ in range(n_surrogates):
        random_phases = np.exp(2j * np.pi * np.random.random(len(fft_signal)))
        surrogate_fft = amplitudes * random_phases
        # Preserve DC and Nyquist as real
        surrogate_fft[0] = fft_signal[0]
        if n % 2 == 0:
            surrogate_fft[-1] = np.abs(surrogate_fft[-1])
        surrogate = np.fft.irfft(surrogate_fft, n=n)

        surr_profile = compute_asymmetry_profile(surrogate, sfreq, scales_hz,
                                                  method, min_prominence)
        for name in results:
            results[name]["A_surrogates"].append(surr_profile[name]["A"])

    # Compute p-values
    for name in results:
        surr_array = np.array(results[name]["A_surrogates"])
        surr_array = surr_array[~np.isnan(surr_array)]
        A_obs = results[name]["A_observed"]

        if len(surr_array) == 0 or np.isnan(A_obs):
            results[name]["p_value"] = np.nan
            results[name]["z_score"] = np.nan
        else:
            results[name]["p_value"] = np.mean(surr_array >= A_obs)
            results[name]["z_score"] = (
                (A_obs - np.mean(surr_array)) / (np.std(surr_array) + 1e-10)
            )
        results[name]["A_surrogates"] = surr_array

    return results
