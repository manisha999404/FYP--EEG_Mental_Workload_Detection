"""
build_dataset.py
----------------
Builds the model-ready STEW dataset using the EXISTING preprocessing.py.

Output (default: <repo>/processed_data/contract_v1/):
    X_filtered.npy : (n_windows, 512, 14) float32, filtered, NOT normalized
    y.npy          : (n_windows,) int64, 0 = low/rest, 1 = high/task
    metadata.csv   : subject, condition, label, window, recording
    config.json    : preprocessing parameters, versions, data hash

Run from the repo root:
    python preprocessing/build_dataset.py
    python preprocessing/build_dataset.py --legacy-dir processed_data --overwrite
"""

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

# preprocessing.py sits in the same folder as this script
import preprocessing as prep

CHANNEL_NAMES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                 "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
FILE_PATTERN = re.compile(r"^sub(\d+)_(hi|lo)\.txt$")
LABEL_OF = {"lo": 0, "hi": 1}
CONDITION_OF = {"lo": "low", "hi": "high"}
SEED = 42  # no randomness is used in this script; recorded for completeness


def list_recordings(dataset_dir: Path):
    """Return a sorted list of (subject_id, condition_code, path)."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    recordings = []
    for path in sorted(dataset_dir.glob("sub*.txt")):  # ratings.txt is excluded
        match = FILE_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected file name: {path.name}")
        recordings.append((int(match.group(1)), match.group(2), path))

    if not recordings:
        raise FileNotFoundError(f"No sub*.txt recordings in {dataset_dir}")
    return recordings


def build_dataset(dataset_dir: Path):
    """Run preprocess_eeg on every recording and stack the windows."""
    expected_shape = (prep.WINDOW_SECONDS * prep.SAMPLING_RATE, len(CHANNEL_NAMES))
    all_windows, rows = [], []

    for subject, cond_code, path in list_recordings(dataset_dir):
        windows = prep.preprocess_eeg(path)

        # Edge-case checks: wrong shape or non-finite values must stop the build
        if windows.ndim != 3 or windows.shape[1:] != expected_shape:
            raise ValueError(f"{path.name}: got {windows.shape}, "
                             f"expected (n, {expected_shape[0]}, {expected_shape[1]})")
        if not np.isfinite(windows).all():
            raise ValueError(f"{path.name}: contains NaN or Inf")

        all_windows.append(windows.astype(np.float32))
        for window_index in range(len(windows)):
            rows.append({
                "subject": subject,
                "condition": CONDITION_OF[cond_code],
                "label": LABEL_OF[cond_code],
                "window": window_index,
                "recording": path.name,
            })
        print(f"  {path.name:14s} -> {windows.shape}")

    X = np.concatenate(all_windows, axis=0)
    metadata = pd.DataFrame(rows)
    y = metadata["label"].to_numpy(dtype=np.int64)
    return X, y, metadata


def sanity_checks(X, y, metadata):
    """Every subject must have the same number of low and high windows."""
    assert len(X) == len(y) == len(metadata), "Length mismatch"
    counts = metadata.groupby(["subject", "condition"]).size().unstack()
    if counts.isna().any().any():
        raise ValueError("Some subject is missing a low or a high recording")
    if not (counts["low"] == counts["high"]).all():
        raise ValueError("Low/high window counts differ for some subject")
    print(f"\nSubjects: {metadata['subject'].nunique()}")
    print(f"Windows per recording: {sorted(counts.stack().unique())}")
    print(f"Class balance (label 0 / 1): {np.bincount(y).tolist()}")


def save_outputs(X, y, metadata, out_dir: Path, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "X_filtered.npy").exists() and not overwrite:
        raise FileExistsError(f"{out_dir} already has a dataset. "
                              "Use --overwrite to replace it.")

    np.save(out_dir / "X_filtered.npy", X)
    np.save(out_dir / "y.npy", y)
    metadata.to_csv(out_dir / "metadata.csv", index=False)

    config = {
        "sampling_rate_hz": prep.SAMPLING_RATE,
        "notch_hz": prep.NOTCH_FREQUENCY,
        "notch_q": prep.NOTCH_Q,
        "bandpass_hz": [prep.LOW_CUTOFF, prep.HIGH_CUTOFF],
        "bandpass_order": prep.FILTER_ORDER,
        "filtering": "zero-phase filtfilt (offline, non-causal)",
        "window_seconds": prep.WINDOW_SECONDS,
        "overlap": prep.OVERLAP,
        "normalization": "none (must be fitted on training subjects only)",
        "layout": "(windows, time, channels) - channels last",
        "channel_names": CHANNEL_NAMES,
        "shape": list(X.shape),
        "X_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
        "seed": SEED,
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__,
                     "scipy": scipy.__version__,
                     "pandas": pd.__version__},
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved to {out_dir}")


def verify_against_legacy(X, metadata, legacy_dir: Path):
    """
    Prove the new data equals your notebook-built data.
    Legacy X_train/X_test were (X_filtered - train_mean) / train_std, so we
    apply the saved legacy statistics to the new windows and compare.
    """
    train_mean = np.load(legacy_dir / "train_mean.npy")
    train_std = np.load(legacy_dir / "train_std.npy")
    cols = ["subject", "condition", "label", "window"]

    for name in ("train", "test"):
        legacy_meta = pd.read_csv(legacy_dir / f"{name}_metadata.csv")
        legacy_X = np.load(legacy_dir / f"X_{name}.npy")

        mask = metadata["subject"].isin(legacy_meta["subject"].unique()).to_numpy()
        new_meta = metadata.loc[mask, cols].reset_index(drop=True)
        same_rows = new_meta.equals(legacy_meta[cols].reset_index(drop=True))

        rebuilt = (X[mask] - train_mean) / train_std
        same_values = (rebuilt.shape == legacy_X.shape
                       and np.allclose(rebuilt, legacy_X, atol=1e-3))
        max_diff = (float(np.abs(rebuilt - legacy_X).max())
                    if rebuilt.shape == legacy_X.shape else float("nan"))

        print(f"[{name}] metadata rows identical: {same_rows} | "
              f"values identical: {same_values} | max abs diff: {max_diff:.2e}")
        if not (same_rows and same_values):
            raise AssertionError(f"Legacy check FAILED for {name} split")
    print("Legacy check PASSED: new dataset reproduces the notebook dataset.")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the STEW model-ready dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=repo_root / "STEW Dataset")
    parser.add_argument("--out-dir", type=Path,
                        default=repo_root / "processed_data" / "contract_v1")
    parser.add_argument("--legacy-dir", type=Path, default=None,
                        help="processed_data folder from the notebook, to verify against")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"Reading recordings from: {args.dataset_dir}")
    X, y, metadata = build_dataset(args.dataset_dir)
    sanity_checks(X, y, metadata)
    print(f"X shape: {X.shape} | dtype: {X.dtype}")

    save_outputs(X, y, metadata, args.out_dir, args.overwrite)
    if args.legacy_dir is not None:
        verify_against_legacy(X, metadata, args.legacy_dir)


if __name__ == "__main__":
    main()