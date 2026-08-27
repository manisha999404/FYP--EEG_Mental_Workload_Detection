# EEG Mental Workload Detection

## Real-Time EEG-Based Mental Workload Detection Using Signal Processing, Machine Learning and Deep Learning

## Overview

This project aims to detect mental workload using EEG (Electroencephalography) signals.

The system processes EEG signals using Digital Signal Processing (DSP), Machine Learning (ML), and Deep Learning (DL) techniques to classify mental workload into:

- **Low Workload**
- **High Workload**

The project uses the **STEW (Simultaneous Task EEG Workload) Dataset**.

## Project Pipeline

Raw EEG Data  
↓  
Signal Preprocessing  
↓  
EEG Segmentation  
↓  
Subject-wise Train/Test Split  
↓  
Train-based Normalization  
↓  
Feature Extraction  
↓  
Machine Learning  
↓  
Deep Learning (EEGNet)  
↓  
Mental Workload Prediction

## Dataset

The STEW (Simultaneous Task EEG Workload) Dataset contains EEG recordings from **48 subjects** under low- and high-workload conditions.

- **14 EEG channels**
- **Sampling frequency:** 128 Hz
- **19,200 samples per recording**
- **96 recordings in total**

The raw dataset is not included in this repository due to its size.

## Preprocessing

The current preprocessing pipeline consists of:

1. **50 Hz Notch Filtering** — removal of power-line interference.
2. **1–40 Hz Bandpass Filtering** — retaining the relevant EEG frequency range.
3. **EEG Segmentation** — 4-second windows with 50% overlap.
4. **Subject-wise Train/Test Split** — 38 training subjects and 10 testing subjects.
5. **Train-based Normalization** — normalization parameters are calculated only from training data to prevent data leakage.

### Processed Dataset

- **7,104 EEG windows**
- Each window: **512 samples × 14 channels**
- Low workload: **3,552 windows**
- High workload: **3,552 windows**

Final dataset:

X_train → (5624, 512, 14)  
X_test → (1480, 512, 14)

## Repository Structure

FYP/
├── STEW Dataset/              
├── notebooks/
│   └── EEG_preprocessing.ipynb
├── preprocessing/
│   └── preprocessing.py
├── feature_extraction/
├── machine_learning/
├── deep_learning/
├── processed_data/            
├── results/
├── utils/
├── .gitignore
└── README.md

## Current Status

**Completed:** EEG dataset analysis, DSP preprocessing, segmentation, subject-wise splitting, train-based normalization, and reusable preprocessing pipeline.

**Next:** Feature extraction, Machine Learning, EEGNet, visualization, and final integration.

## References

1. **STEW: Simultaneous Task EEG Workload Data Set** — IEEE DataPort, 2018.
2. **EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces** — Journal of Neural Engineering, 2018.
3. **Classification of Mental Workload Using Brain Connectivity and Machine Learning on EEG Data**.
