# EEG-Based Mental Workload Detection

## Overview

A Final Year Project (ECE) focused on detecting mental workload from EEG signals using Digital Signal Processing (DSP), Machine Learning (ML), and Deep Learning (DL).

The project uses the STEW (Simultaneous Task EEG Workload) dataset to study EEG patterns associated with rest and multitasking conditions.

> Current implementation is offline. Real-time inference is planned as a future extension.

## Project Pipeline

STEW Raw EEG
     ↓
Filtering & Preprocessing
     ↓
4-second Windowing
     ↓
Subject-wise Train/Validation/Test Split
     ↓
Data Quality & Signal Audit
     ↓
Feature Extraction
     ↓
ML / DL Models
     ↓
Final Evaluation
     ↓
Real-time Extension

## Dataset

Dataset: STEW (Simultaneous Task EEG Workload)

- 48 subjects
- 14 EEG channels
- Sampling frequency: 128 Hz
- 2 conditions:
  - Low workload: Rest
  - High workload: SIMKAP multitasking condition
- 74 windows per recording
- Window size: 4 seconds (512 samples)
- 50% overlap

### Important

The labels represent rest vs. multitasking, rather than a continuous measurement of workload. The model therefore learns EEG patterns associated with these two conditions as a workload proxy.

## Preprocessing

- 50 Hz notch filtering
- 1–40 Hz Butterworth band-pass filter
- 4th-order filter
- Zero-phase filtfilt
- 4-second windows
- 50% overlap
- 14 EEG channels

### Channel Order

AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

The final EEG dataset is filtered but not additionally normalized.

## Final Dataset

X_filtered.npy
Shape: (7104, 512, 14)
Type: float32

Total windows: 7104
Low workload: 3552
High workload: 3552
Subjects: 48

## Subject-wise Data Split

The split is performed at the subject level to prevent data leakage.

| Split | Subjects | Windows |
|---|---:|---:|
| Train | 30 | 4440 |
| Validation | 8 | 1184 |
| Test | 10 | 1480 |
| Total | 48 | 7104 |

No subject appears in more than one split.

### Cross-Validation

5-fold subject-wise cross-validation is performed only on the 38 development subjects.

The final 10 test subjects remain untouched until the complete pipeline is frozen.

## Data Audit

The dataset was checked for:

- Shape and numerical validity
- Missing/invalid values
- Subject-wise consistency
- Amplitude variation
- Extreme-value concentration
- PSD and frequency-band behaviour

The audit showed considerable between-subject amplitude variation.

No subjects or windows were removed based on this audit.

## Normalization Study

Three variants were compared:

A. No additional scaling
B. Z-score scaling
C. Robust scaling

Subject-wise cross-validation showed that channel-wise scaling produced almost identical classical ML results.

Therefore:

No additional global channel-wise normalization is applied to the shared EEG dataset.

Model-specific normalization can still be tested for deep learning, but it must be fitted using training data only.

## Baseline Result

A Logistic Regression baseline achieved approximately:

Accuracy: ~76%
F1-score: ~76%
AUC: ~0.83–0.84

This provides a baseline for comparison with future models.

## Next Steps

### 1. Feature Extraction

Extract:

- Time-domain features
- Frequency-domain / band-power features
- Relative band power
- Hjorth parameters
- Other suitable EEG features

Compare feature families:

Time-domain
Frequency-domain
Time + Frequency
Time + Frequency + Hjorth
Selected features

### 2. Classical ML

Evaluate models such as:

- Logistic Regression
- SVM
- Random Forest
- XGBoost / LightGBM

Use the same subject-wise folds for fair comparison.

### 3. Deep Learning

Evaluate models such as:

- 1D CNN
- EEGNet
- CNN-LSTM / suitable temporal models

Training must use only the development subjects.

### 4. Final Test

After feature and model selection is completely frozen:

- Evaluate once on the 10 held-out test subjects.
- Do not use test data for feature selection, tuning, normalization, or early stopping.

## Repository Structure

FYP/
│
├── preprocessing/
│   ├── preprocessing.py
│   ├── build_dataset.py
│   ├── make_splits.py
│   ├── audit_dataset.py
│   ├── audit_robust.py
│   └── normalisation.py
│
├── processed_data/
│   └── contract_v1/
│       ├── X_filtered.npy
│       ├── y.npy
│       ├── metadata.csv
│       ├── splits.json
│       └── config.json
│
├── results/
│   ├── step3_audit/
│   ├── step3b_audit/
│   └── step4_normalization/
│
└── README.md

X_filtered.npy is stored using Git LFS.

The raw STEW dataset is not included in the repository.

## Reproducing the Dataset

From the project root:

python preprocessing/build_dataset.py
python preprocessing/make_splits.py
python preprocessing/audit_dataset.py
python preprocessing/audit_robust.py
python preprocessing/normalisation.py

## Important Rules

1. Use subject-wise splitting for all evaluation.
2. Never use test subjects during development.
3. Fit scalers using training subjects only.
4. Do not remove subjects/windows based on test performance.
5. Use the same folds when comparing models.
6. Document all feature engineering and model choices.
7. Keep the frozen preprocessing dataset unchanged unless a modification is documented and validated.

## Limitations

- STEW provides condition-based labels rather than continuous workload scores.
- EEG signals contain subject-specific variations.
- Eye/muscle activity and other physiological effects may influence recordings.
- Current implementation is offline because filtering uses zero-phase filtfilt.
- Real-time EEG acquisition and inference are future work.

## Current Status

| Component | Status |
|---|---|
| Dataset & preprocessing | Complete |
| Subject-wise split | Complete |
| 5-fold CV | Complete |
| Data audit | Complete |
| Normalization study | Complete |
| Feature extraction | Next |
| Classical ML | Pending |
| Deep Learning / EEGNet | Pending |
| Final test | Pending |
| Real-time system | Future work |

## References

- STEW: Simultaneous Task EEG Workload Dataset
- Emotiv EEG recordings
- Standard EEG signal-processing and machine-learning methods
