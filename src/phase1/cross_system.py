"""
Phase 1: Cross-system coherence analysis.

Computes ρ_cross — correlation between A_EEG(τ) and A_HRV(τ) profiles —
the proposed novel biomarker of aging.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.interpolate import interp1d


def compute_cross_coherence(eeg_profile, hrv_profile):
    """Compute cross-system asymmetry coherence ρ_cross.

    Correlates EEG A(τ) profile with HRV A(τ) profile after
    interpolating to a common frequency axis.

    Parameters
    ----------
    eeg_profile : dict
        Band name -> {'A': float, 'center_freq': float, ...}
    hrv_profile : dict
        Band name -> {'A': float, 'center_freq': float, ...}

    Returns
    -------
    rho_pearson : float
        Pearson correlation.
    rho_spearman : float
        Spearman correlation.
    p_pearson : float
    p_spearman : float
    eeg_A_interp : array
        EEG A values at common frequencies.
    hrv_A_interp : array
        HRV A values at common frequencies.
    common_freqs : array
        Common frequency axis (log-spaced).
    """
    # Extract (freq, A) pairs
    eeg_freqs = np.array([eeg_profile[b]["center_freq"] for b in eeg_profile])
    eeg_A = np.array([eeg_profile[b]["A"] for b in eeg_profile])

    hrv_freqs = np.array([hrv_profile[b]["center_freq"] for b in hrv_profile])
    hrv_A = np.array([hrv_profile[b]["A"] for b in hrv_profile])

    # Remove NaN
    valid_eeg = ~np.isnan(eeg_A)
    valid_hrv = ~np.isnan(hrv_A)

    eeg_freqs = eeg_freqs[valid_eeg]
    eeg_A = eeg_A[valid_eeg]
    hrv_freqs = hrv_freqs[valid_hrv]
    hrv_A = hrv_A[valid_hrv]

    if len(eeg_A) < 2 or len(hrv_A) < 2:
        return np.nan, np.nan, np.nan, np.nan, None, None, None

    # Interpolate both to common log-frequency axis
    all_freqs = np.sort(np.concatenate([eeg_freqs, hrv_freqs]))
    log_eeg_freqs = np.log10(eeg_freqs)
    log_hrv_freqs = np.log10(hrv_freqs)
    log_common = np.log10(all_freqs)

    # Use linear interpolation in log-frequency space
    eeg_interp = interp1d(log_eeg_freqs, eeg_A, kind="linear",
                          fill_value="extrapolate")
    hrv_interp = interp1d(log_hrv_freqs, hrv_A, kind="linear",
                          fill_value="extrapolate")

    eeg_A_common = eeg_interp(log_common)
    hrv_A_common = hrv_interp(log_common)

    # Correlation
    if len(log_common) < 3:
        return np.nan, np.nan, np.nan, np.nan, eeg_A_common, hrv_A_common, all_freqs

    rho_p, p_p = pearsonr(eeg_A_common, hrv_A_common)
    rho_s, p_s = spearmanr(eeg_A_common, hrv_A_common)

    return rho_p, rho_s, p_p, p_s, eeg_A_common, hrv_A_common, all_freqs


def compute_cross_transfer_entropy(eeg_bands, hrv_bands, eeg_sfreq, hrv_sfreq):
    """Compute transfer entropy between EEG and HRV band envelopes.

    Parameters
    ----------
    eeg_bands : dict
        EEG band name -> signal.
    hrv_bands : dict
        HRV band name -> signal.
    eeg_sfreq : float
    hrv_sfreq : float

    Returns
    -------
    te_eeg_to_hrv : dict
        (eeg_band, hrv_band) -> TE.
    te_hrv_to_eeg : dict
        (hrv_band, eeg_band) -> TE.
    net_flow : float
        Mean TE(EEG→HRV) - Mean TE(HRV→EEG).
        Positive = brain drives heart. Negative = heart drives brain.
    """
    from ..utils.transfer_entropy import transfer_entropy_ksg

    # Compute envelopes at common sampling rate
    from scipy.signal import hilbert, resample
    common_sfreq = min(eeg_sfreq, hrv_sfreq, 4.0)  # Use HRV rate

    def get_envelope(signal, orig_sfreq, target_sfreq):
        env = np.abs(hilbert(signal))
        if orig_sfreq != target_sfreq:
            n_samples = int(len(env) * target_sfreq / orig_sfreq)
            env = resample(env, n_samples)
        return env

    te_eeg_to_hrv = {}
    te_hrv_to_eeg = {}

    for eeg_band, eeg_sig in eeg_bands.items():
        eeg_env = get_envelope(eeg_sig, eeg_sfreq, common_sfreq)
        for hrv_band, hrv_sig in hrv_bands.items():
            hrv_env = get_envelope(hrv_sig, hrv_sfreq, common_sfreq)

            # Match lengths
            min_len = min(len(eeg_env), len(hrv_env))
            if min_len < 100:
                continue

            te_eeg_to_hrv[(eeg_band, hrv_band)] = transfer_entropy_ksg(
                eeg_env[:min_len], hrv_env[:min_len], k=5, dim=3
            )
            te_hrv_to_eeg[(hrv_band, eeg_band)] = transfer_entropy_ksg(
                hrv_env[:min_len], eeg_env[:min_len], k=5, dim=3
            )

    # Net flow
    all_eeg2hrv = [v for v in te_eeg_to_hrv.values() if not np.isnan(v)]
    all_hrv2eeg = [v for v in te_hrv_to_eeg.values() if not np.isnan(v)]

    net_flow = (
        (np.mean(all_eeg2hrv) if all_eeg2hrv else 0)
        - (np.mean(all_hrv2eeg) if all_hrv2eeg else 0)
    )

    return te_eeg_to_hrv, te_hrv_to_eeg, net_flow
