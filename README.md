# aDBS_project

This repository is designed to carry out analyses on neural data from aDBS devices with sensing and stimulation capabilities.

Pipeline:

- .venv = virtual environment 
- Scripts:
    - Timeseries Visualisation : reads in the json files for a unique defined patient, displayes and saves the selected file from aDBS Setup ('Calibration Test' for specific channel used in aDBS)
    - EKG Removal: reads in saved raw clip, displays timeseries data and user identifies peak in R wave of QRS complex, 'Template Subtraction Pipeline' carried out and save clip without EKG artifact.
    
