"""
Transfer entropy computation between wavelet scales.

TE(X → Y) measures how much past X reduces uncertainty about future Y,
beyond what past Y alone provides. Used to quantify inter-scale information flow.
"""

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree


def _embed(x, dim, lag=1):
    """Create time-delay embedding of a signal."""
    n = len(x) - (dim - 1) * lag
    if n <= 0:
        raise ValueError(f"Signal too short for embedding: len={len(x)}, dim={dim}, lag={lag}")
    indices = np.arange(n)[:, None] + np.arange(dim)[None, :] * lag
    return x[indices]


def transfer_entropy_ksg(source, target, k=5, dim=1, lag=1):
    """Compute transfer entropy using KSG (Kraskov-Stögbauer-Grassberger) estimator.

    TE(source → target) = I(target_future; source_past | target_past)

    Parameters
    ----------
    source : array, shape (n_samples,)
    target : array, shape (n_samples,)
    k : int
        Number of nearest neighbors for KSG estimator.
    dim : int
        Embedding dimension.
    lag : int
        Embedding lag.

    Returns
    -------
    te : float
        Transfer entropy in nats.
    """
    n = min(len(source), len(target))
    source = source[:n]
    target = target[:n]

    # Create embeddings
    # target_past: embedded target history
    # source_past: embedded source history
    # target_future: next value of target
    max_offset = dim * lag
    if n <= max_offset + 1:
        return np.nan

    target_past = _embed(target[:-1], dim, lag)[:n - max_offset - 1]
    source_past = _embed(source[:-1], dim, lag)[:n - max_offset - 1]
    target_future = target[max_offset + 1: max_offset + 1 + len(target_past)]

    min_len = min(len(target_past), len(source_past), len(target_future))
    target_past = target_past[:min_len]
    source_past = source_past[:min_len]
    target_future = target_future[:min_len].reshape(-1, 1)

    if min_len <= k + 1:
        return np.nan

    # Joint space: (target_future, target_past, source_past)
    joint = np.hstack([target_future, target_past, source_past])

    # Marginal spaces
    target_joint = np.hstack([target_future, target_past])  # (Y_f, Y_p)
    cond_joint = np.hstack([target_past, source_past])       # (Y_p, X_p)

    # KSG estimator
    # Find k-th neighbor distance in joint space
    tree_joint = KDTree(joint)
    dists, _ = tree_joint.query(joint, k=k + 1)
    eps = dists[:, -1]  # distance to k-th neighbor

    # Count neighbors within eps in marginal spaces
    tree_tf_tp = KDTree(target_joint)
    tree_tp_sp = KDTree(cond_joint)
    tree_tp = KDTree(target_past)

    n_tf_tp = np.array([
        tree_tf_tp.query_radius(target_joint[i:i+1], r=eps[i] + 1e-10,
                                count_only=True)[0] - 1
        for i in range(min_len)
    ])
    n_tp_sp = np.array([
        tree_tp_sp.query_radius(cond_joint[i:i+1], r=eps[i] + 1e-10,
                                count_only=True)[0] - 1
        for i in range(min_len)
    ])
    n_tp = np.array([
        tree_tp.query_radius(target_past[i:i+1], r=eps[i] + 1e-10,
                             count_only=True)[0] - 1
        for i in range(min_len)
    ])

    # TE = <psi(n_tp) - psi(n_tf_tp) - psi(n_tp_sp)> + psi(k)
    # where psi = digamma function
    valid = (n_tf_tp > 0) & (n_tp_sp > 0) & (n_tp > 0)
    if valid.sum() < k:
        return np.nan

    te = (
        digamma(k)
        + np.mean(digamma(n_tp[valid]))
        - np.mean(digamma(n_tf_tp[valid]))
        - np.mean(digamma(n_tp_sp[valid]))
    )

    return max(te, 0.0)  # TE is non-negative by definition


def interscale_transfer_entropy(bands, sfreq, scale_names=None, k=5, dim=3):
    """Compute transfer entropy between all pairs of frequency bands.

    Parameters
    ----------
    bands : dict
        Band name -> signal array (output of bandpass_decompose).
    sfreq : float
        Sampling frequency.
    scale_names : list or None
        Ordered list of band names (low freq to high freq).
        If None, uses sorted keys.
    k : int
        KSG neighbor count.
    dim : int
        Embedding dimension.

    Returns
    -------
    te_matrix : dict
        (source_band, target_band) -> TE value.
    net_flow : dict
        (band_i, band_j) -> TE(i→j) - TE(j→i), for i < j in frequency.
        Positive = net upward flow (fast → slow).
    """
    if scale_names is None:
        scale_names = sorted(bands.keys())

    # Downsample high-frequency bands to common rate for TE computation
    # Use amplitude envelope (Hilbert) for fair comparison across scales
    envelopes = {}
    for name in scale_names:
        analytic = np.abs(np.array(bands[name], dtype=np.float64))
        # Smooth envelope
        from scipy.ndimage import uniform_filter1d
        kernel_size = max(int(sfreq * 0.1), 3)  # 100ms smoothing
        envelopes[name] = uniform_filter1d(analytic, kernel_size)

    te_matrix = {}
    for src in scale_names:
        for tgt in scale_names:
            if src == tgt:
                te_matrix[(src, tgt)] = 0.0
            else:
                te_matrix[(src, tgt)] = transfer_entropy_ksg(
                    envelopes[src], envelopes[tgt], k=k, dim=dim
                )

    # Net flow: positive = source has higher frequency
    net_flow = {}
    for i, s1 in enumerate(scale_names):
        for j, s2 in enumerate(scale_names):
            if i < j:  # s1 = lower freq, s2 = higher freq
                net_flow[(s2, s1)] = te_matrix[(s2, s1)] - te_matrix[(s1, s2)]

    return te_matrix, net_flow
