"""
ECG artifact removal on an aDBS-setup clip saved from the waveform viewer.

Reads a clip JSON exported by
    Scripts/TimeseriesVisualisation/adaptpd_sensing_waveform_viewer.ipynb
and runs the same RemoveArtTempMatch logic as for_aryaman.py.

Requirements
------------
    pip install numpy scipy matplotlib scikit-learn
"""
#%%
import json

import numpy as np

from remove_art_temp_match import remove_art_temp_match

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
JSON_PATH = (
    "/project/hammer_neuromod/Projects/PD_ADAPT_BetaStab/aDBS_project/"
    "ProcessedData/aDBS_Setup_Clips_Raw/"
    "BTX569588_ADBS_SETUP_NA_CalibrationTests_ONE_THREE_ONE_THREE_full_aDBS_Setup_20260609-142325.json"
)
HEMISPHERE = "Left"   # which hemisphere panel in the clip to clean: 'Left' or 'Right'

# QRS seed: timestamp (seconds, clip-relative) of a clean QRS complex used to
# build the template. Inspect the signal and set this to a beat you can see.
T_samp_QRS = 4.556    # seconds

# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #
with open(JSON_PATH) as f:
    clip = json.load(f)

panel = clip["hemispheres"][HEMISPHERE]
sig = np.asarray(panel["voltage"], dtype=float)   # microvolts
fs = float(panel["sample_rate_hz"])

print(
    f"{clip['patient']} | {panel['channel']} | {panel['stream']}/{panel['pass'] or '-'} | "
    f"{len(sig)} samples @ {fs:g} Hz ({len(sig) / fs:.2f}s)"
)

# QRS seed: extract a window around the known QRS timestamp
temp_seed = sig[
    int(np.floor((T_samp_QRS - 0.2) * fs)) : int(np.floor((T_samp_QRS + 0.42) * fs))
]

# ------------------------------------------------------------------ #
# Run artifact removal
# ------------------------------------------------------------------ #
sig_clean, art_rem, art_log, template = remove_art_temp_match(
    sig_orig=sig,
    fs=fs,
    temp_seed=temp_seed,
    f_est=[50 / 60, 100 / 60],   # 50-100 bpm in Hz
    t_inc=[[0, 0]],               # use whole signal
    t_ign=None,
    search=True,
    scaled=False,
    pl=True,
)
#%%
