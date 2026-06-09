import numpy as np
from scipy.signal import lfilter, find_peaks


def matched_filter(sig, fs, template, f_est, P=0.975):
    """
    Takes a signal and finds locations of an artifact using a matched filter.

    Parameters
    ----------
    sig : np.ndarray
        Signal to clean (1-D column vector).
    fs : float
        Sampling rate (Hz).
    template : np.ndarray
        Waveform for matching.
    f_est : float or array-like
        Estimated occurrence rate of artifact (Hz). Can be a scalar or a
        two-element range (e.g. [50/60, 100/60] for 50-100 bpm).
    P : float, optional
        Percentile for thresholding matches (0–1). Default 0.975.

    Returns
    -------
    locs : np.ndarray
        Indices where matches begin (0-based).
    """
    # Convert percent to fraction if needed
    if P > 1:
        P = P / 100.0

    # Run matched filter (correlate template reversed with signal)
    m_filt = lfilter(template[::-1], [1.0], sig)

    # Minimum time interval between matches
    f_est = np.atleast_1d(f_est)
    if len(f_est) == 1:
        t_min = 1.0 / f_est[0] / 2.0
    else:
        t_min = 0.9 / np.max(f_est)

    # Threshold: P-th percentile of |mFilt|
    threshold = np.sort(np.abs(m_filt))[::-1][round(len(m_filt) * (1 - P))]

    # Find peaks above threshold, separated by at least t_min
    locs_end, _ = find_peaks(
        m_filt,
        distance=int(t_min * fs),
        height=threshold,
    )

    # Convert end-of-template indices to start-of-template indices (0-based)
    locs = locs_end - len(template) + 1
    return locs
