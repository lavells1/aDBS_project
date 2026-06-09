"""
Example usage script — equivalent to ForAryaman.m

Loads example data and runs RemoveArtTempMatch to clean ECG artifact
from a neural signal.

Requirements
------------
    pip install numpy scipy matplotlib scikit-learn scipy
    (mat file loading requires scipy.io.loadmat)
"""
#%%
import numpy as np
import scipy.io as sio

from remove_art_temp_match import remove_art_temp_match

# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #
mat = sio.loadmat(
    "/project/hammer_neuromod/Projects/PD_ADAPT_BetaStab/aDBS_project/EKG_removal/TemplateSubtraction_exampleData_WithoutFS_AG.mat"
)
sig = mat["RSTN_TempSub"]["FilteredTruncData"][0, 0].flatten()

fs = 250

# QRS seed: extract a window around the known QRS timestamp
T_samp_QRS = 4.556  # seconds
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
    f_est=[50 / 60, 100 / 60],   # 50–100 bpm in Hz
    t_inc=[[0, 0]],               # use whole signal
    t_ign=None,
    search=True,
    scaled=False,
    pl=True,
)
#%%