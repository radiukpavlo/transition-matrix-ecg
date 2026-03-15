"""Pacing spike detection and interpolation-based removal."""

from __future__ import annotations


def detect_pacing_spikes(signal, threshold_scale: float = 8.0):  # type: ignore[no-untyped-def]
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("detect_pacing_spikes requires numpy") from exc

    x = np.asarray(signal, dtype=float)
    diff = np.abs(np.diff(x, axis=0))
    threshold = diff.mean() + threshold_scale * diff.std()
    spike_idx = np.argwhere(diff > threshold)
    return spike_idx.tolist()


def remove_pacing_spikes(signal, spike_idx):  # type: ignore[no-untyped-def]
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("remove_pacing_spikes requires numpy") from exc

    x = np.asarray(signal, dtype=float).copy()
    for sample_idx, lead_idx in spike_idx:
        left = max(sample_idx - 1, 0)
        right = min(sample_idx + 1, x.shape[0] - 1)
        x[sample_idx, lead_idx] = (x[left, lead_idx] + x[right, lead_idx]) / 2.0
    return x
