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
import glob
import os
import numpy as np
from remove_art_temp_match import remove_art_temp_match
import matplotlib.pyplot as plt
# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

ID = "ATC028888"


json_files = glob.glob(f"/project/hammer_neuromod/Projects/PD_ADAPT_BetaStab/aDBS_project/ProcessedData/aDBS_Setup_Clips_Raw/{ID}_*.json")
if not json_files:
    raise FileNotFoundError(f"No JSON files found starting with ID {ID}")
JSON_PATH = json_files[0]  # or apply further logic if multiple matches
print(f"Using JSON file: {os.path.basename(JSON_PATH)}")


# Right Hemisphere
HEMISPHERE = "Left"   # which hemisphere panel in the clip to clean: 'Left' or 'Right'


#%%
# QRS seed: timestamp (seconds, clip-relative) of a clean QRS complex used to
# build the template. Inspect the signal and set this to a beat you can see.
   # seconds

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


#%% plot the signal
time_lim = 1
time = np.arange(len(sig)) / fs
plt.plot(time, sig)
plt.xlim(0, time_lim)
plt.xticks(np.arange(0, time_lim, 0.1), size=5)
plt.xlabel("Time (s)")
plt.ylabel("LFP (uV)")
plt.title(f"{ID} {HEMISPHERE} Hemisphere")
plt.show()


#%%
T_samp_QRS = 0.2

# QRS seed: extract a window around the known QRS timestamp
temp_seed = sig[
    int(np.floor((T_samp_QRS - 0.2) * fs)) : int(np.floor((T_samp_QRS + 0.4) * fs))
]

# ------------------------------------------------------------------ #
# Run artifact removal
# ------------------------------------------------------------------ #
sig_clean, art_rem, art_log, template = remove_art_temp_match(
    ptID=ID,
    Hemisphere=HEMISPHERE,
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

# plot sig_clean
plt.plot(sig_clean)
plt.xlabel("Time (s)")
plt.ylabel("LFP (uV)")
plt.title(f"{ID} {HEMISPHERE} Hemisphere")
plt.show()

clip["hemispheres"][HEMISPHERE]["voltage"] = sig_clean.tolist()
with open(f"/project/hammer_neuromod/Projects/PD_ADAPT_BetaStab/aDBS_project/ProcessedData/aDBS_Setup_Clips_noEKG/{ID}_ADBS_SETUP_NA_CalibrationTests_ONE_THREE_ONE_THREE_full_aDBS_Setup_20260609-142325_noEKG.json", "w") as f:
    json.dump(clip, f)

#%%
