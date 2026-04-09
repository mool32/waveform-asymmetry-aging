"""
Phase 1: LEMON EEG preprocessing pipeline.

Steps:
1. Load raw EEG (BrainVision format, BIDS)
2. Standard artifact rejection (ICA)
3. Cardiac artifact removal (ICA components correlated with ECG)
4. Wavelet/bandpass decomposition into 6 scales
"""

import numpy as np
import mne
from pathlib import Path


# EEG frequency bands for asymmetry analysis
EEG_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 50),
    "high_gamma": (50, 80),
}


def load_lemon_eeg(subject_id, bids_root, task="EC"):
    """Load LEMON EEG data in BIDS format.

    Parameters
    ----------
    subject_id : str
        E.g. 'sub-010002'
    bids_root : str or Path
        Root of BIDS dataset.
    task : str
        'EC' (eyes closed) or 'EO' (eyes open).

    Returns
    -------
    raw : mne.io.Raw
        Raw EEG data.
    """
    bids_root = Path(bids_root)

    # LEMON BIDS structure: sub-XXX/eeg/sub-XXX_task-XX_eeg.vhdr
    vhdr_path = bids_root / subject_id / "eeg" / f"{subject_id}_task-{task}_eeg.vhdr"

    if not vhdr_path.exists():
        # Try alternative BIDS layouts
        from glob import glob
        pattern = str(bids_root / subject_id / "**" / f"*task-{task}*eeg.vhdr")
        matches = glob(pattern, recursive=True)
        if matches:
            vhdr_path = Path(matches[0])
        else:
            raise FileNotFoundError(
                f"EEG file not found for {subject_id}, task={task} in {bids_root}"
            )

    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True)
    return raw


def preprocess_eeg(raw, ecg_channel=None, l_freq=0.5, h_freq=100.0,
                   n_ica_components=20, random_state=42):
    """Full preprocessing pipeline for LEMON EEG.

    Parameters
    ----------
    raw : mne.io.Raw
    ecg_channel : str or None
        Name of ECG channel. If None, will try to find it automatically.
    l_freq, h_freq : float
        Bandpass filter bounds.
    n_ica_components : int
        Number of ICA components.
    random_state : int

    Returns
    -------
    raw_clean : mne.io.Raw
        Cleaned EEG.
    ecg_signal : array or None
        ECG signal extracted before removing ECG channel.
    ecg_sfreq : float
        ECG sampling rate.
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    """
    # Extract ECG if present
    ecg_signal = None
    ecg_sfreq = raw.info["sfreq"]

    if ecg_channel is None:
        # Try to find ECG channel
        ecg_candidates = [ch for ch in raw.ch_names
                          if "ECG" in ch.upper() or "EKG" in ch.upper()]
        if ecg_candidates:
            ecg_channel = ecg_candidates[0]

    if ecg_channel and ecg_channel in raw.ch_names:
        ecg_signal = raw.get_data(picks=[ecg_channel]).flatten()
        ecg_sfreq = raw.info["sfreq"]

    # Pick EEG channels only
    raw_eeg = raw.copy().pick_types(eeg=True)

    # Set average reference
    raw_eeg.set_eeg_reference("average", projection=True)
    raw_eeg.apply_proj()

    # Bandpass filter
    raw_eeg.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")

    # Notch filter for line noise (50 Hz for European data)
    raw_eeg.notch_filter(freqs=[50, 100], fir_design="firwin")

    # ICA for artifact rejection
    ica = mne.preprocessing.ICA(
        n_components=n_ica_components,
        method="fastica",
        random_state=random_state,
        max_iter=1000,
    )
    ica.fit(raw_eeg)

    # Find and remove eye blink components
    eog_indices, eog_scores = ica.find_bads_eog(raw_eeg)
    ica.exclude.extend(eog_indices)

    # Find and remove cardiac artifact components (CRITICAL per proposal)
    if ecg_signal is not None:
        # Create temporary raw with ECG for correlation
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name=ecg_channel)
        ica.exclude.extend(ecg_indices)
        print(f"  Removing {len(eog_indices)} EOG + {len(ecg_indices)} ECG components")
    else:
        print(f"  Removing {len(eog_indices)} EOG components (no ECG channel found)")

    # Apply ICA
    raw_clean = ica.apply(raw_eeg.copy())

    return raw_clean, ecg_signal, ecg_sfreq, ica


def compute_eeg_asymmetry(raw_clean, bands=None):
    """Compute A(τ) profile from cleaned EEG for each channel.

    Parameters
    ----------
    raw_clean : mne.io.Raw
        Preprocessed EEG.
    bands : dict or None
        Frequency bands. Default: EEG_BANDS.

    Returns
    -------
    results : dict
        channel_name -> asymmetry profile (from compute_asymmetry_profile).
    grand_average : dict
        Band name -> mean A(τ) across channels.
    """
    from ..utils.asymmetry import compute_asymmetry_profile

    if bands is None:
        bands = EEG_BANDS

    sfreq = raw_clean.info["sfreq"]
    data = raw_clean.get_data()  # (n_channels, n_samples)

    results = {}
    for i, ch_name in enumerate(raw_clean.ch_names):
        profile = compute_asymmetry_profile(data[i], sfreq, bands, method="bandpass")
        results[ch_name] = profile

    # Grand average across channels
    grand_average = {}
    for band_name in bands:
        A_values = [results[ch][band_name]["A"] for ch in results
                    if not np.isnan(results[ch][band_name]["A"])]
        grand_average[band_name] = {
            "A_mean": np.mean(A_values) if A_values else np.nan,
            "A_std": np.std(A_values) if A_values else np.nan,
            "A_median": np.median(A_values) if A_values else np.nan,
            "n_channels": len(A_values),
            "center_freq": (bands[band_name][0] + bands[band_name][1]) / 2,
        }

    return results, grand_average
