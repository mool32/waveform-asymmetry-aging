"""
Peak-Trough Asymmetry for Calcium Imaging Traces.

Adapts the broadband PTA approach from EEG to Ca2+ fluorescence signals.
Ca2+ transients have known asymmetry: fast rise (channel opening) → slow decay (pump/buffer).

Key differences from EEG:
- Ca2+ transients are slower (0.1-5 Hz typical)
- Signal is always positive (fluorescence), not oscillatory around zero
- Rise = depolarization/Ca2+ influx, Fall = repolarization/Ca2+ clearance
- Expected: A > 0.5 (rise faster than fall) for all healthy cells
"""

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import uniform_filter1d


def preprocess_trace(trace, fs, lowcut=0.05, highcut=None, smooth_window=None):
    """
    Preprocess Ca2+ fluorescence trace.

    Parameters
    ----------
    trace : array
        Raw fluorescence trace (F or dF/F)
    fs : float
        Sampling frequency (Hz)
    lowcut : float
        High-pass cutoff to remove slow drift (Hz)
    highcut : float or None
        Low-pass cutoff (Hz). If None, set to fs/4
    smooth_window : int or None
        Smoothing window (samples). If None, no smoothing.

    Returns
    -------
    filtered : array
        Preprocessed trace
    """
    if highcut is None:
        highcut = min(fs / 4, 10.0)

    # Bandpass filter
    nyq = fs / 2
    if lowcut > 0 and highcut < nyq:
        b, a = butter(3, [lowcut / nyq, highcut / nyq], btype='band')
        filtered = filtfilt(b, a, trace)
    elif lowcut > 0:
        b, a = butter(3, lowcut / nyq, btype='high')
        filtered = filtfilt(b, a, trace)
    else:
        filtered = trace.copy()

    if smooth_window and smooth_window > 1:
        filtered = uniform_filter1d(filtered, smooth_window)

    return filtered


def detect_transients(trace, fs, min_prominence=None, min_distance_sec=0.5):
    """
    Detect Ca2+ transients as peaks in the trace.

    Returns list of dicts with rise_time, fall_time, A for each transient.
    """
    if min_prominence is None:
        min_prominence = np.std(trace) * 1.0

    min_distance = max(1, int(min_distance_sec * fs))

    # Find peaks
    peaks, props = find_peaks(trace, prominence=min_prominence, distance=min_distance)

    if len(peaks) < 2:
        return []

    # Find troughs between consecutive peaks
    transients = []

    for i in range(len(peaks)):
        peak_idx = peaks[i]

        # Find trough before peak
        if i == 0:
            search_start = max(0, peak_idx - int(2 * min_distance))
        else:
            search_start = peaks[i - 1]

        trough_before = search_start + np.argmin(trace[search_start:peak_idx])

        # Find trough after peak
        if i == len(peaks) - 1:
            search_end = min(len(trace), peak_idx + int(2 * min_distance))
        else:
            search_end = peaks[i + 1]

        trough_after = peak_idx + np.argmin(trace[peak_idx:search_end])

        # Compute rise and fall times
        rise_samples = peak_idx - trough_before
        fall_samples = trough_after - peak_idx

        if rise_samples < 1 or fall_samples < 1:
            continue

        rise_time = rise_samples / fs
        fall_time = fall_samples / fs
        total_time = rise_time + fall_time

        # Asymmetry: A = rise_time / total_time
        # A > 0.5 means rise is slower than fall (unusual for Ca2+)
        # A < 0.5 means rise is faster than fall (expected for Ca2+)
        #
        # To match EEG convention where A > 0.5 = "fast rise":
        # We report A = rise_fraction = rise / (rise + fall)
        # For Ca2+: expect A < 0.5 (fast rise, slow fall)
        #
        # BUT in our framework, A = (t_fall - t_rise) / (t_fall + t_rise)
        # This gives A > 0 when fall > rise (expected for Ca2+)

        A = (fall_time - rise_time) / (fall_time + rise_time)
        rise_fraction = rise_time / total_time

        transients.append({
            'peak_idx': peak_idx,
            'trough_before': trough_before,
            'trough_after': trough_after,
            'rise_time': rise_time,
            'fall_time': fall_time,
            'total_time': total_time,
            'A': A,  # (fall - rise) / (fall + rise), > 0 = fast rise slow fall
            'rise_fraction': rise_fraction,
            'amplitude': trace[peak_idx] - trace[trough_before],
        })

    return transients


def compute_cell_asymmetry(trace, fs, lowcut=0.05, highcut=None,
                            min_prominence=None, min_cycles=5):
    """
    Compute asymmetry statistics for a single cell's Ca2+ trace.

    Returns dict with mean_A, median_A, std_A, n_transients, prop_positive.
    """
    # Preprocess
    filtered = preprocess_trace(trace, fs, lowcut=lowcut, highcut=highcut)

    # Detect transients
    transients = detect_transients(filtered, fs, min_prominence=min_prominence)

    if len(transients) < min_cycles:
        return None

    A_values = np.array([t['A'] for t in transients])
    rise_fractions = np.array([t['rise_fraction'] for t in transients])

    return {
        'mean_A': np.mean(A_values),
        'median_A': np.median(A_values),
        'std_A': np.std(A_values),
        'n_transients': len(transients),
        'prop_positive': np.mean(A_values > 0),  # fraction with fall > rise
        'mean_rise_fraction': np.mean(rise_fractions),
        'mean_rise_time': np.mean([t['rise_time'] for t in transients]),
        'mean_fall_time': np.mean([t['fall_time'] for t in transients]),
        'transients': transients,
    }


def analyze_population(traces, fs, cell_coords=None, **kwargs):
    """
    Analyze asymmetry across a population of cells.

    Parameters
    ----------
    traces : array (n_cells, n_timepoints) or list of arrays
        Ca2+ traces for each cell
    fs : float
        Sampling frequency
    cell_coords : array (n_cells, 2) or None
        Spatial coordinates of each cell (x, y)

    Returns
    -------
    results : dict with population-level statistics
    cell_results : list of per-cell results
    """
    cell_results = []

    for i, trace in enumerate(traces):
        result = compute_cell_asymmetry(trace, fs, **kwargs)
        if result is not None:
            if cell_coords is not None and i < len(cell_coords):
                result['x'] = cell_coords[i, 0]
                result['y'] = cell_coords[i, 1]
            result['cell_id'] = i
            cell_results.append(result)

    if not cell_results:
        return None, []

    A_values = np.array([r['mean_A'] for r in cell_results])

    # Population statistics
    from scipy.stats import ttest_1samp, wilcoxon

    t_stat, p_ttest = ttest_1samp(A_values, 0)

    try:
        w_stat, p_wilcoxon = wilcoxon(A_values)
    except:
        p_wilcoxon = np.nan

    results = {
        'n_cells': len(cell_results),
        'n_total': len(traces),
        'mean_A': np.mean(A_values),
        'median_A': np.median(A_values),
        'std_A': np.std(A_values),
        'prop_positive': np.mean(A_values > 0),
        'ttest_vs_0': {'t': t_stat, 'p': p_ttest},
        'wilcoxon_vs_0': {'p': p_wilcoxon},
        'sigma_A': np.std(A_values),  # spatial variability
    }

    # Spatial analysis if coordinates available
    if cell_coords is not None and len(cell_results) > 10:
        coords = np.array([[r['x'], r['y']] for r in cell_results])
        results['spatial'] = _spatial_analysis(A_values, coords)

    return results, cell_results


def _spatial_analysis(A_values, coords):
    """Compute spatial statistics of A distribution."""
    from scipy.spatial.distance import pdist, squareform

    # Moran's I for spatial autocorrelation
    n = len(A_values)
    dist_matrix = squareform(pdist(coords))

    # Weight matrix: inverse distance, thresholded
    threshold = np.percentile(dist_matrix[dist_matrix > 0], 25)
    W = np.zeros((n, n))
    W[dist_matrix < threshold] = 1.0
    np.fill_diagonal(W, 0)

    if W.sum() == 0:
        return {'morans_I': np.nan}

    # Moran's I
    z = A_values - A_values.mean()
    numerator = n * np.sum(W * np.outer(z, z))
    denominator = W.sum() * np.sum(z**2)
    I = numerator / denominator if denominator != 0 else 0

    # Sign entropy (same as EEG analysis)
    p_pos = np.mean(A_values > 0)
    if 0 < p_pos < 1:
        H = -p_pos * np.log2(p_pos) - (1 - p_pos) * np.log2(1 - p_pos)
    else:
        H = 0.0

    return {
        'morans_I': I,
        'H_sign': H,
        'p_positive': p_pos,
        'sigma_spatial': np.std(A_values),
    }
