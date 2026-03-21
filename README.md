# PharmaHacks 2026 — Challenge 3: AD vs CN EEG Classification 

This project builds a leakage-safe EEG classification pipeline for the **Alzheimer’s Disease (AD) vs Healthy Control (CN)** task from the PharmaHacks 2026 challenge.

## Approach
The pipeline uses:
- **19-channel resting-state EEG**
- **0.5–45 Hz bandpass filtering**
- **Downsampling from 500 Hz to 128 Hz**
- **30-second windows with 15-second overlap**
- Engineered features:
  - Relative Band Power (RBP)
  - Spectral Coherence Connectivity (SCC)
  - Hjorth parameters
  - Shannon entropy

## Validation
To avoid data leakage:
- **StratifiedGroupKFold** was used with **subject IDs as groups**
- All windows from the same subject stayed in the same fold
- Final evaluation was done at the **subject level**

## Models
- **Baseline:** XGBoost on engineered EEG features
- **Experimental:** EEGNet on raw EEG windows

## Best Result
Frozen baseline result:
- **Subject-level F1:** 0.800
- **Balanced Accuracy:** 0.632
- **Decision Threshold:** 0.43

## Main Finding
The feature-based **XGBoost baseline outperformed EEGNet** under the same leakage-safe subject-wise evaluation. This suggests that on a small clinical EEG dataset, handcrafted spectral and connectivity features were more data-efficient than end-to-end deep learning.

## Notes
- Hidden test data was not used for model selection
- Main limitation: lower CN recall relative to AD recall
- Presentation assets include spectral slowing and feature-importance plots
