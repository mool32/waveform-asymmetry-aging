"""
Phase 1: LEMON ECG/HRV preprocessing pipeline.

Steps:
1. R-peak detection
2. RR intervals → HRV time series
3. Wavelet/bandpass decomposition into 4 HRV scales
4. Compute A(τ) for HRV bands
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


# HRV frequency bands for asymmetry analysis
HRV_BANDS = {
    "VLF": (0.003, 0.04),
    "LF": (0.04, 0.15),
    "HF": (0.15, 0.4),
    "VHF": (0.4, 1.0),
}


def detect_r_peaks(ecg_signal, sfreq, method="neurokit"):
    """Detect R-peaks in ECG signal.

    Parameters
    ----------
    ecg_signal : array, shape (n_samples,)
    sfreq : float
        Sampling frequency.
    method : str
        'neurokit' or 'scipy'.

    Returns
    -------
    r_peaks : array
        Indices of R-peaks.
    """
    if method == "neurokit":
        try:
            import neurokit2 as nk
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=int(sfreq))
            _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=int(sfreq))
            r_peaks = info["ECG_R_Peaks"]
            return np.array(r_peaks)
        except ImportError:
            method = "scipy"

    if method == "scipy":
        # Simple R-peak detection via thresholding
        from scipy.signal import butter, filtfilt

        # Bandpass 5-15 Hz to isolate QRS
        nyq = sfreq / 2
        b, a = butter(3, [5 / nyq, min(15 / nyq, 0.99)], btype="band")
        filtered = filtfilt(b, a, ecg_signal)

        # Square to enhance R-peaks
        squared = filtered ** 2

        # Find peaks with minimum distance ~0.5s (120 bpm max)
        min_dist = int(0.5 * sfreq)
        height_threshold = np.mean(squared) + 2 * np.std(squared)
        peaks, _ = find_peaks(squared, distance=min_dist, height=height_threshold)
        return peaks


def compute_rr_intervals(r_peaks, sfreq):
    """Compute RR intervals from R-peak indices.

    Parameters
    ----------
    r_peaks : array
        R-peak sample indices.
    sfreq : float

    Returns
    -------
    rr_times : array
        Times of RR intervals (midpoint between consecutive R-peaks) in seconds.
    rr_intervals : array
        RR intervals in seconds.
    """
    rr_intervals = np.diff(r_peaks) / sfreq
    rr_times = (r_peaks[:-1] + r_peaks[1:]) / (2 * sfreq)

    # Remove physiologically implausible intervals
    valid = (rr_intervals > 0.3) & (rr_intervals < 2.0)  # 30-200 bpm
    rr_times = rr_times[valid]
    rr_intervals = rr_intervals[valid]

    return rr_times, rr_intervals


def rr_to_continuous(rr_times, rr_intervals, target_sfreq=4.0):
    """Interpolate RR intervals to evenly-sampled time series.

    Parameters
    ----------
    rr_times : array
        Times of RR intervals.
    rr_intervals : array
        RR intervals in seconds.
    target_sfreq : float
        Target sampling frequency for interpolated HRV signal.

    Returns
    -------
    hrv_signal : array
        Evenly-sampled HRV time series.
    t_hrv : array
        Time points.
    """
    if len(rr_times) < 4:
        return np.array([]), np.array([])

    # Cubic spline interpolation
    interp_func = interp1d(rr_times, rr_intervals, kind="cubic",
                           fill_value="extrapolate")

    t_start = rr_times[0]
    t_end = rr_times[-1]
    t_hrv = np.arange(t_start, t_end, 1.0 / target_sfreq)
    hrv_signal = interp_func(t_hrv)

    # Detrend
    hrv_signal = hrv_signal - np.mean(hrv_signal)

    return hrv_signal, t_hrv


def compute_hrv_asymmetry(ecg_signal, ecg_sfreq, hrv_sfreq=4.0, bands=None):
    """Full ECG → HRV → A(τ) pipeline.

    Parameters
    ----------
    ecg_signal : array
    ecg_sfreq : float
        ECG sampling rate.
    hrv_sfreq : float
        Resampling rate for HRV.
    bands : dict or None

    Returns
    -------
    profile : dict
        HRV asymmetry profile (band -> A, n_cycles, etc.)
    rr_times : array
    rr_intervals : array
    hrv_signal : array
    """
    from ..utils.asymmetry import compute_asymmetry_profile

    if bands is None:
        bands = HRV_BANDS

    # R-peak detection
    r_peaks = detect_r_peaks(ecg_signal, ecg_sfreq)

    if len(r_peaks) < 10:
        return None, None, None, None

    # RR intervals
    rr_times, rr_intervals = compute_rr_intervals(r_peaks, ecg_sfreq)

    # Interpolate to continuous
    hrv_signal, t_hrv = rr_to_continuous(rr_times, rr_intervals, hrv_sfreq)

    if len(hrv_signal) < hrv_sfreq * 30:  # need at least 30s
        return None, rr_times, rr_intervals, None

    # Compute asymmetry profile
    profile = compute_asymmetry_profile(hrv_signal, hrv_sfreq, bands,
                                         method="bandpass")

    return profile, rr_times, rr_intervals, hrv_signal


def compute_acceleration_deceleration(rr_intervals):
    """Compute HRV acceleration and deceleration capacity.

    This is an additional asymmetry metric specific to cardiac dynamics:
    acceleration = RR shortening (faster beats), deceleration = RR lengthening.

    Parameters
    ----------
    rr_intervals : array

    Returns
    -------
    ac : float
        Acceleration capacity (negative value = shortening).
    dc : float
        Deceleration capacity (positive value = lengthening).
    ratio : float
        |DC| / |AC| — expected > 1 in healthy (deceleration dominates).
    """
    diffs = np.diff(rr_intervals)

    # Phase-rectified signal averaging (PRSA)
    # Acceleration anchors: where RR decreases
    acc_anchors = np.where(diffs < 0)[0] + 1
    # Deceleration anchors: where RR increases
    dec_anchors = np.where(diffs > 0)[0] + 1

    window = 5  # PRSA window

    def prsa(anchors, signal, w):
        segments = []
        for a in anchors:
            if a - w >= 0 and a + w < len(signal):
                segments.append(signal[a - w: a + w + 1])
        if not segments:
            return np.nan
        avg = np.mean(segments, axis=0)
        # Capacity = (avg[w] + avg[w+1] - avg[w-1] - avg[w-2]) / 4
        return (avg[w] + avg[w + 1] - avg[w - 1] - avg[w - 2]) / 4

    ac = prsa(acc_anchors, rr_intervals, window)
    dc = prsa(dec_anchors, rr_intervals, window)

    ratio = abs(dc) / (abs(ac) + 1e-10) if not (np.isnan(ac) or np.isnan(dc)) else np.nan

    return ac, dc, ratio
