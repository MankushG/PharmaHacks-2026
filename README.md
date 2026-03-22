# PharmaHacks 2026 — Challenge 3: AD vs CN EEG Classification 

This project implements a leakage-safe machine learning pipeline to distinguish between **Alzheimer’s Disease (AD)** and **Healthy Control (CN)** subjects using resting-state EEG data.

## 🧠 Project Overview
The pipeline processes 19-channel EEG recordings to extract clinically relevant spectral and connectivity biomarkers. Unlike standard deep learning approaches that may overfit on small clinical samples, this project utilizes a **biomarker-rich feature engineering** strategy combined with regularized linear models to achieve high interpretability and robust generalization.

## 🛠 Technical Approach

### Data Preprocessing
- **Signal Specs:** 19-channel resting-state EEG (International 10-20 system).
- **Filtering:** 0.5–45 Hz bandpass (4th order Butterworth).
- **Resampling:** Downsampled from 500 Hz to 128 Hz for computational efficiency.
- **Artifact Rejection:** Automatic rejection of windows with amplitudes exceeding ±500 µV or flatline signals.
- **Segmentation:** 30-second sliding windows with a 15-second (50%) overlap.

### Feature Engineering (240 Features)
The pipeline extracts features across five frequency bands (Delta, Theta, Alpha, Beta, Gamma) and five brain regions (Frontal, Temporal, Central, Parietal, Occipital):
- **Relative Band Power (RBP):** Regional and global distribution of spectral energy.
- **Connectivity:** Spectral Coherence (SCC) between all regional pairs (e.g., Frontal-Occipital coherence).
- **Signal Complexity:** Hjorth Parameters (Activity, Mobility, and Complexity) and Spectral Entropy.
- **Clinical Ratios:** Specialized biomarkers including the **Slowing Index** and Theta/Alpha ratios.

## 🔬 Validation Strategy
To prevent **Data Leakage**, this pipeline enforces a strict **Subject-Wise Validation** protocol:
- **StratifiedGroupKFold:** Subject IDs are used as groups to ensure that windows from the same participant never appear in both training and validation folds simultaneously.
- **Subject-Level Inference:** Probabilities from individual 30s windows are aggregated at the subject level to provide a single diagnostic prediction per patient.
- **Threshold Tuning:** The decision threshold is optimized on the validation set to maximize the **Balanced Accuracy**, addressing the class imbalance between AD and CN samples.

## 📈 Performance & Findings

### Best Model
The top-performing architecture was a **Regularized Logistic Regression (ElasticNet)** using the top 60 features selected via ANOVA F-value ranking.

| Metric | Cross-Validation Score (Dev) |
| :--- | :--- |
| **Balanced Accuracy** | **0.8215** |
| **F1 Score** | **0.8182** |
| **ROC-AUC** | **0.8154** |
| **Optimized Threshold** | **0.51** |

### Key Findings
1. **Feature-based vs. Deep Learning:** The regularized XGBoost and Logistic Regression baselines consistently outperformed end-to-end models like EEGNet. On this clinical dataset, handcrafted spectral features proved more data-efficient than raw signal learning.
2. **Spectral Slowing:** Feature importance analysis confirmed "Spectral Slowing" as a primary biomarker; AD subjects exhibited significantly higher Global Slowing Indices and reduced Occipital Alpha power compared to controls.
3. **Connectivity Deficits:** Reduced coherence in the Alpha band between temporal and parietal regions was a high-ranking predictor for AD classification.

## 📂 Project Structure
- `pharmahacks.ipynb`: The complete end-to-end pipeline (Preprocessing -> Feature Extraction -> Training -> Evaluation).
- `dev_cv_results_full.csv`: Detailed metric breakdown for all tested models.
- `best_dev_model_full.pkl`: The frozen, trained pipeline for inference.
- `presentation_figures/`: Automatically generated visualizations including ROC curves, confusion matrices, and biomarker boxplots.

## 🚀 Usage
The notebook is designed for a Google Colab environment with data hosted on Google Drive. 
1. Place `.npy` EEG files in `/Train/AD/` and `/Train/CN/`.
2. Update the `BASE_DIR` in the config cell.
3. Run the pipeline to regenerate features or perform inference on the `/testing/` folder.
