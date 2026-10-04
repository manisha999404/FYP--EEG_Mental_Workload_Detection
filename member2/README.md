# Member 2 — Feature Extraction and Classical Machine Learning

## Contribution

This module focuses on extracting frequency-domain features from the preprocessed EEG signals and using classical machine learning for mental workload classification.

The workflow implemented in this module is:

STEW EEG Dataset  
↓  
Preprocessed EEG Windows  
↓  
Welch Power Spectral Density (PSD)  
↓  
Frequency Band Power Extraction  
↓  
70-Dimensional Feature Vector  
↓  
Subject-Wise Cross-Validation  
↓  
Standardization  
↓  
Logistic Regression  
↓  
Performance Evaluation

---

## 1. Input Data

The feature extraction pipeline uses the preprocessed EEG data provided by Member 1.

Each EEG window contains:

- 14 EEG channels
- Sampling frequency: 128 Hz
- Window duration: 4 seconds
- Samples per window: 512
- 50% overlap between consecutive windows

Therefore, each window has the shape:

`(512, 14)`

The complete dataset contains:

`7104 EEG windows`

---

## 2. Power Spectral Density

Welch's method is used to estimate the Power Spectral Density (PSD) of each EEG channel.

Configuration:

- Sampling frequency: 128 Hz
- Window: Hann
- `nperseg = 256`
- `noverlap = 128`
- Detrending: constant
- Frequency resolution: 0.5 Hz

PSD is used because mental workload-related changes can be studied through the distribution of EEG power across different frequency bands.

---

## 3. EEG Frequency Bands

Five frequency bands are extracted:

| Band | Frequency Range |
|---|---|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Low Gamma | 30–40 Hz |

For every EEG channel, the power in each frequency band is calculated using numerical integration of the PSD.

---

## 4. Feature Extraction

Absolute band power is currently used for the classical ML baseline.

There are:

- 14 EEG channels
- 5 frequency bands

Therefore:

`14 × 5 = 70 features per EEG window`

The resulting feature matrix has the shape:

`7104 × 70`

A relative band-power feature extraction pipeline was also implemented for exploratory comparison.

---

## 5. Classification

A Logistic Regression classifier is used as the initial classical machine-learning baseline.

The model pipeline consists of:

1. StandardScaler
2. Logistic Regression

Standardization is performed inside the cross-validation pipeline so that information from validation subjects does not leak into the training process.

---

## 6. Subject-Wise Cross-Validation

Randomly splitting individual EEG windows is avoided because multiple windows originate from the same subjects and overlapping windows can be highly correlated.

Instead, subject-wise cross-validation is used.

The development subjects are divided into 5 folds.

For every fold:

- Training subjects are used to train the model.
- Validation subjects are kept completely separate.
- The scaler is fitted only on the training data.
- The trained model is evaluated on the validation subjects.

This provides a more realistic estimate of how the model may generalize to unseen subjects.

---

## 7. Preliminary Results

The current Logistic Regression baseline produced the following 5-fold subject-wise cross-validation results:

| Metric | Mean ± Standard Deviation |
|---|---:|
| Accuracy | 66.46 ± 9.87% |
| Balanced Accuracy | 66.46 ± 9.87% |
| F1 Score | 67.26 ± 9.52% |
| ROC-AUC | 69.23 ± 8.52% |

These results are considered a preliminary classical-ML baseline and should not be interpreted as the final performance of the complete project.

---

## 8. Notebook

The complete implementation is available in:

`01_feature_extraction_and_ml.ipynb`

The notebook contains:

- EEG window inspection
- Welch PSD computation
- Frequency-band definition
- Absolute band-power extraction
- Relative band-power extraction
- Feature-matrix construction
- Subject-wise data splitting
- Logistic Regression
- 5-fold subject-wise cross-validation
- Performance evaluation

---

## 9. Future Work

Planned extensions include:

- Comparison with SVM
- Comparison with Random Forest
- Comparison of different feature representations
- Deep-learning-based EEG classification
- EEGNet integration
- Real-time EEG processing
- Visualization/dashboard integration

---

## Note

The current implementation is based on the publicly available STEW dataset and preprocessed EEG data.

The project should only be described as real-time after the trained pipeline is successfully integrated with live EEG acquisition hardware.
