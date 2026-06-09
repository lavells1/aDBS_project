import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def plot_temp_match(ptID, Hemisphere, sig, fs, template, locs, scaled=False):
    """
    Plot template matches overlaid on the raw signal.

    Parameters
    ----------
    ptID : str
        Patient ID.
    Hemisphere : str
        Hemisphere.
    sig : np.ndarray
        Signal (1-D column vector).
    fs : float
        Sampling rate (Hz).
    template : np.ndarray
        Matched template waveform.
    locs : np.ndarray
        Start indices of template matches (0-based).
    scaled : bool, optional
        Whether to scale each template match before plotting.

    Returns
    -------
    p_sig : matplotlib.lines.Line2D
        Handle for the raw signal plot.
    p_temps : list of matplotlib.lines.Line2D
        Handles for each template overlay plot.
    """
    i_temp = np.arange(len(template))
    t_sig = np.arange(len(sig)) / fs

    ax = plt.gca()
    (p_sig,) = ax.plot(t_sig, sig, "k", label="signal")

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [p["color"] for p in prop_cycle]

    p_temps = []
    for j, loc in enumerate(locs):
        color = colors[j % len(colors)]
        i_full_temp = i_temp + loc  # 0-based
        i_inc = (i_full_temp >= 0) & (i_full_temp < len(sig))

        c = 1.0
        if scaled:
            X = template[i_inc].reshape(-1, 1)
            y = sig[i_full_temp[i_inc]]
            reg = LinearRegression(fit_intercept=False).fit(X, y)
            c = reg.coef_[0]

        (line,) = ax.plot(
            i_full_temp[i_inc] / fs,
            template[i_inc] * c,
            linewidth=2,
            color=color,
        )
        ax.text(
            loc / fs,
            np.min(template) - 0.1 * np.ptp(template),
            str(j + 1),
            color=color,
        )
        p_temps.append(line)

    ax.set_title(f"{ptID} : {Hemisphere} Hemisphere")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("LFP (uV)")
    return p_sig, p_temps
