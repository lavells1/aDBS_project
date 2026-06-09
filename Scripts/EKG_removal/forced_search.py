import numpy as np
from scipy.signal import lfilter
from scipy.stats import mode as scipy_mode


def _mode(arr):
    """Return the scalar mode of an integer array."""
    result = scipy_mode(arr, keepdims=False)
    return int(result.mode)


def forced_search(sig, template, locs, W):
    """
    Find inter-match intervals that are longer than expected and perform a
    forced search for missed templates.

    Parameters
    ----------
    sig : np.ndarray
        Signal (1-D), may be zero-padded at both ends.
    template : np.ndarray
        Waveform for matching.
    locs : np.ndarray
        Indices where template matches begin (0-based).
    W : int
        Search half-width in samples. The total search window is 2*W samples.

    Returns
    -------
    locs : np.ndarray
        Updated indices where template matches begin (0-based), sorted.
    """
    # Find padding extents
    nonzero_mask = sig != 0
    pad_changes = np.where(np.diff(nonzero_mask.astype(int)) != 0)[0]
    front_pad = pad_changes[0] + 1   # first non-zero index
    end_pad = len(sig) - (pad_changes[-1] + 1)  # samples after last non-zero

    # Convert start-of-template locs to end-of-template locs
    locs_end = locs + len(template) - 1

    d = np.diff(locs_end)
    md = _mode(d)

    # Estimate missed matches at the beginning
    N_front = int(np.floor((locs_end[0] - front_pad) / md))
    locs_addend = locs_end.copy().tolist()
    if N_front > 0:
        locs_addend = [int(locs_end[0] - round((N_front + 1.25) * md))] + locs_addend

    # Estimate missed matches at the end
    N_end = int(np.floor((len(sig) - end_pad - locs_end[-1]) / md + 1))
    if N_end > 0:
        locs_addend = locs_addend + [int(locs_end[-1] + round((N_end + 1.25) * md))]

    locs_addend = np.array(locs_addend)
    d_addend = np.diff(locs_addend)
    i_poss = np.where(d_addend > 1.5 * md)[0]

    # Pre-compute full matched filter output
    m_filt_full = lfilter(template[::-1], [1.0], sig)

    new_locs = []
    for i in i_poss:
        new_locs_temp = []
        est_loc = int(locs_addend[i] + md)

        while est_loc < (locs_addend[i + 1] - len(template) * 3 / 4):
            m_filt = m_filt_full.copy()

            # Zero out everything outside the search window
            lo = max(0, est_loc - W)
            hi = min(len(m_filt), est_loc + W)
            mask = np.zeros(len(m_filt), dtype=bool)
            mask[lo:hi] = True
            m_filt[~mask] = 0

            idx = int(np.argmax(m_filt))
            start_idx = idx - len(template) + 1

            if start_idx > 0 and idx < len(sig):
                new_locs_temp.append(idx)
                est_loc = new_locs_temp[-1] + md
            else:
                est_loc += md

        new_locs.extend(new_locs_temp)

    locs_end = np.sort(np.concatenate([locs_end, np.array(new_locs, dtype=int)]))
    locs = locs_end - len(template) + 1
    return locs
