import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from matched_filter import matched_filter
from forced_search import forced_search
from plot_temp_match import plot_temp_match


def remove_art_temp_match(
    ptID,
    Hemisphere,
    sig_orig,
    fs,
    temp_seed,
    f_est,
    t_inc,
    t_ign,
    search=True,
    scaled=False,
    pl=False,
):
    """
    Remove a repetitive artifact from a signal using a modified Woody's
    adaptive filter with optional forced search.

    Parameters
    ----------
    ptID : str
        Patient ID.
    Hemisphere : str
        Hemisphere.
    sig_orig : np.ndarray
        Signal to clean (1-D array).
    fs : float
        Sampling rate (Hz).
    temp_seed : np.ndarray
        Seed waveform for template matching.
    f_est : float or array-like
        Estimated occurrence rate of artifact (Hz). Scalar or [min, max].
    t_inc : np.ndarray
        (N, 2) array of [start, stop] times (seconds) where the artifact is
        present. Pass [[0, 0]] to use the entire signal.
    t_ign : np.ndarray or None
        (N, 2) array of [start, stop] times (seconds) to ignore (e.g. motion
        artifact). Pass None or empty array if none.
    search : bool, optional
        Whether to include a forced search for missed templates.
    scaled : bool, optional
        Whether to scale each template before subtraction.
    pl : bool, optional
        Whether to plot the result.

    Returns
    -------
    sig_clean : np.ndarray
        Signal with artifact subtracted.
    art_rem : np.ndarray
        Artifact vector that was subtracted.
    art_log : list of np.ndarray
        List of (indices, template_values) arrays for each subtracted artifact.
    template : np.ndarray
        Final averaged Woody's filter template.
    """
    t_inc = np.atleast_2d(t_inc)
    if t_ign is None or (np.ndim(t_ign) == 0) or len(t_ign) == 0:
        t_ign = np.empty((0, 2))
    t_ign = np.atleast_2d(t_ign)

    # ------------------------------------------------------------------ #
    # Extract the excerpt that contains the artifact
    # ------------------------------------------------------------------ #
    if np.sum(t_inc[0]) == 0:
        sig_ex = sig_orig.copy()
    else:
        sig_ex = np.zeros_like(sig_orig)
        for j in range(t_inc.shape[0]):
            i0 = round(t_inc[j, 0] * fs)
            i1 = round(t_inc[j, 1] * fs)
            sig_ex[i0:i1] = sig_orig[i0:i1]

    # ------------------------------------------------------------------ #
    # Zero-pad signal
    # ------------------------------------------------------------------ #
    pad = len(temp_seed)
    sig_pad = np.concatenate([np.zeros(pad), sig_ex, np.zeros(pad)])

    # ------------------------------------------------------------------ #
    # Iteratively update template until match locations converge
    # ------------------------------------------------------------------ #
    old_locs = np.array([1])
    locs = np.array([0])
    template = temp_seed.copy()

    while len(np.setdiff1d(old_locs, locs)) > 0:
        old_locs = locs.copy()
        locs = matched_filter(sig_pad, fs, template, f_est, 0.975)

        i_temp = np.arange(len(template))  # 0-based offsets within template
        # Build index matrix: each row = indices in sig_pad for one match
        i_matches = locs[:, None] + i_temp[None, :]  # shape (n_locs, len_template)

        # Clip to valid range
        valid = (i_matches >= 0) & (i_matches < len(sig_pad))
        i_matches_clipped = np.clip(i_matches, 0, len(sig_pad) - 1)
        matches = sig_pad[i_matches_clipped]
        matches[~valid] = np.nan

        # Identify matches to ignore based on t_ign
        bool_ign = np.zeros(len(locs), dtype=bool)
        for i_ign in range(t_ign.shape[0]):
            t0, t1 = t_ign[i_ign]
            # Ignore match if all covered indices fall within [t0, t1]
            in_window = np.all(
                (i_matches / fs > t0) & (i_matches / fs < t1), axis=1
            )
            bool_ign |= in_window

        template = np.nanmean(matches[~bool_ign, :], axis=0)

    # ------------------------------------------------------------------ #
    # Forced search for missed matches
    # ------------------------------------------------------------------ #
    if search:
        locs = forced_search(sig_pad, template, locs, round(fs / np.min(f_est) * 0.1))

    # ------------------------------------------------------------------ #
    # Subtract artifact from signal
    # ------------------------------------------------------------------ #
    locs = locs - pad  # shift back to original signal indexing
    locs = locs[locs > -len(template)]

    i_temp = np.arange(len(template))
    art_rem = np.zeros(len(sig_orig))
    art_log = []

    for j, loc in enumerate(locs):
        i_full_temp = i_temp + loc  # 0-based indices in sig_orig
        i_inc = (i_full_temp >= 0) & (i_full_temp < len(sig_orig))

        c = 1.0
        if scaled:
            X = template[i_inc].reshape(-1, 1)
            y = sig_orig[i_full_temp[i_inc]]
            reg = LinearRegression(fit_intercept=False).fit(X, y)
            c = reg.coef_[0]

        art_rem[i_full_temp[i_inc]] += template[i_inc] * c
        art_log.append(
            np.column_stack([i_full_temp[i_inc], template[i_inc] * c])
        )

    sig_clean = sig_orig - art_rem

    # ------------------------------------------------------------------ #
    # Optional plot
    # ------------------------------------------------------------------ #
    if pl:
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.sca(ax)
        p_o, _ = plot_temp_match(ptID, Hemisphere, sig_orig, fs, template, locs, scaled)
        ax.set_xlim([0, 20])

        offset = np.max(sig_orig) - np.min(sig_orig)
        (p_c,) = ax.plot(
            np.arange(len(sig_orig)) / fs,
            sig_clean - offset,
            "b",
            label="noise removed",
        )
        p_o.set_label("original signal")
        ax.legend(handles=[p_o, p_c])
        ax.tick_params(labelsize=16)
        ax.set_xlabel("time (s)", fontsize=16)
        ax.set_ylabel("LFP (uV)", fontsize=16)

        # Inset showing template
        inset_ax = fig.add_axes([0.7, 0.15, 0.2, 0.2])
        inset_ax.plot(template, "k", linewidth=2)
        inset_ax.set_xlim([0, len(template)])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        plt.show()

    return sig_clean, art_rem, art_log, template
