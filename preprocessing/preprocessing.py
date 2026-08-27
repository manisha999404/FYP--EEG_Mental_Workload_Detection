import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


# ============================================================
# EEG PREPROCESSING CONFIGURATION
# ============================================================

SAMPLING_RATE = 128

NOTCH_FREQUENCY = 50
NOTCH_Q = 30

LOW_CUTOFF = 1
HIGH_CUTOFF = 40
FILTER_ORDER = 4

WINDOW_SECONDS = 4
OVERLAP = 0.5


# ============================================================
# FILTER DESIGN
# ============================================================

def create_notch_filter():
    """Create a 50 Hz notch filter."""
    
    b_notch, a_notch = iirnotch(
        NOTCH_FREQUENCY,
        NOTCH_Q,
        SAMPLING_RATE
    )

    return b_notch, a_notch


def create_bandpass_filter():
    """Create a 1–40 Hz Butterworth bandpass filter."""

    nyquist = SAMPLING_RATE / 2

    low = LOW_CUTOFF / nyquist
    high = HIGH_CUTOFF / nyquist

    b, a = butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass"
    )

    return b, a


# ============================================================
# EEG SEGMENTATION
# ============================================================

def segment_eeg(data):
    """
    Segment EEG data into overlapping fixed-length windows.

    Input:
        data → (samples, channels)

    Output:
        segments → (windows, window_samples, channels)
    """

    window_samples = int(
        WINDOW_SECONDS * SAMPLING_RATE
    )

    step_samples = int(
        window_samples * (1 - OVERLAP)
    )

    segments = []

    for start in range(
        0,
        len(data) - window_samples + 1,
        step_samples
    ):

        end = start + window_samples

        segments.append(
            data[start:end]
        )

    return np.array(segments)


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_eeg(file_path):
    """
    Complete preprocessing pipeline for one STEW EEG recording.

    Steps:
        1. Load raw EEG
        2. 50 Hz notch filtering
        3. 1–40 Hz bandpass filtering
        4. 4-second segmentation with 50% overlap

    Normalization is intentionally NOT performed here.
    Normalization parameters must be calculated from the
    training data only to avoid data leakage.
    """

    # Load raw EEG
    data = np.loadtxt(file_path)

    # Create filters
    b_notch, a_notch = create_notch_filter()
    b_bandpass, a_bandpass = create_bandpass_filter()

    # 50 Hz notch filtering
    notched_data = filtfilt(
        b_notch,
        a_notch,
        data,
        axis=0
    )

    # 1–40 Hz bandpass filtering
    filtered_data = filtfilt(
        b_bandpass,
        a_bandpass,
        notched_data,
        axis=0
    )

    # Segment EEG
    segments = segment_eeg(filtered_data)

    return segments