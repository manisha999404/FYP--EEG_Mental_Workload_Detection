"""
make_splits.py
--------------
Creates the SUBJECT-LEVEL train / validation / test split and subject-wise
cross-validation folds for the STEW dataset.

The subject assignments in splits.json are the single source of truth.
Windows are always recovered from metadata.csv (never stored here), so the
split is independent of the X arrays.

Design
    test        : 10 subjects, FROZEN from the original notebook split
                  (train_test_split(subjects, test_size=0.2, random_state=42))
    development : the other 38 subjects
        train       : 30 subjects  (seeded random choice from development)
        validation  :  8 subjects  (fixed split, e.g. DL early stopping)
        CV          : 5 subject-wise folds over all 38 development subjects
                      (for classical ML / model selection). Test is never used.

Run from the repo root:
    python preprocessing/make_splits.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Frozen from the original notebook split (random_state=42, test_size=0.2).
# Hard-coded so this file does not depend on a scikit-learn version.
FROZEN_TEST_SUBJECTS = [5, 13, 20, 25, 26, 27, 28, 38, 41, 44]

DEFAULT_SEED = 42
DEFAULT_N_VAL = 8
DEFAULT_N_FOLDS = 5


# ------------------------------------------------------------------
# Helpers reusable by later steps (ML / DL scripts import these)
# ------------------------------------------------------------------
def load_splits(splits_path):
    with open(splits_path) as f:
        return json.load(f)


def window_indices(metadata: pd.DataFrame, subjects):
    """Row indices of metadata.csv (= rows of X) that belong to `subjects`."""
    return np.flatnonzero(metadata["subject"].isin(subjects).to_numpy())


# ------------------------------------------------------------------
# Split construction
# ------------------------------------------------------------------
def make_splits(all_subjects, seed, n_val, n_folds):
    all_subjects = sorted(int(s) for s in all_subjects)
    test = sorted(FROZEN_TEST_SUBJECTS)

    missing = set(test) - set(all_subjects)
    if missing:
        raise ValueError(f"Frozen test subjects not in metadata: {sorted(missing)}")

    development = [s for s in all_subjects if s not in set(test)]
    if not 0 < n_val < len(development):
        raise ValueError("n_val must be between 1 and the number of development subjects")
    if not 2 <= n_folds <= len(development):
        raise ValueError("n_folds must be between 2 and the number of development subjects")

    # Fixed validation split: seeded permutation of the development subjects
    rng_split = np.random.RandomState(seed)
    permuted = rng_split.permutation(development)
    validation = sorted(int(s) for s in permuted[:n_val])
    train = sorted(int(s) for s in permuted[n_val:])

    # CV folds over all development subjects. Different seed (seed + 1) so the
    # fold order is not just a copy of the validation draw.
    rng_cv = np.random.RandomState(seed + 1)
    cv_order = rng_cv.permutation(development)
    fold_groups = np.array_split(cv_order, n_folds)  # sizes differ by at most 1
    folds = []
    for k, val_group in enumerate(fold_groups):
        val_k = sorted(int(s) for s in val_group)
        train_k = sorted(s for s in development if s not in set(val_k))
        folds.append({"fold": k, "train_subjects": train_k, "val_subjects": val_k})

    return {
        "meta": {
            "unit_of_split": "subject (all windows of a subject stay together)",
            "seed_train_val_split": seed,
            "seed_cv_folds": seed + 1,
            "n_subjects": len(all_subjects),
            "test_subjects_origin": "frozen from original notebook split "
                                    "(random_state=42, test_size=0.2)",
            "normalization": "none - must be fitted on training subjects only",
        },
        "train_subjects": train,
        "validation_subjects": validation,
        "test_subjects": test,
        "development_subjects": development,
        "cv": {"n_folds": n_folds, "folds": folds},
    }


# ------------------------------------------------------------------
# Validation (raises AssertionError on any problem)
# ------------------------------------------------------------------
def _summary(metadata, subjects):
    part = metadata[metadata["subject"].isin(subjects)]
    return (len(subjects), len(part),
            int((part["label"] == 0).sum()), int((part["label"] == 1).sum()))


def validate_splits(splits, metadata):
    all_subjects = set(int(s) for s in metadata["subject"].unique())
    train = set(splits["train_subjects"])
    val = set(splits["validation_subjects"])
    test = set(splits["test_subjects"])
    dev = set(splits["development_subjects"])

    # ---- 1. train / val / test ----
    print("=" * 66)
    print("TRAIN / VALIDATION / TEST (subject level)")
    print("=" * 66)
    print(f"{'split':<12}{'subjects':>9}{'windows':>9}{'low':>8}{'high':>8}")
    for name, subs in [("train", train), ("validation", val), ("test", test)]:
        n_sub, n_win, n_low, n_high = _summary(metadata, subs)
        print(f"{name:<12}{n_sub:>9}{n_win:>9}{n_low:>8}{n_high:>8}")
        assert n_low == n_high, f"{name}: low/high windows are not balanced"
    for name, subs in [("train", train), ("validation", val), ("test", test)]:
        print(f"  {name} subjects: {sorted(subs)}")

    assert not (train & val), "Subject overlap: train/validation"
    assert not (train & test), "Subject overlap: train/test"
    assert not (val & test), "Subject overlap: validation/test"
    assert train | val | test == all_subjects, "Not all subjects are accounted for"
    assert len(train) + len(val) + len(test) == len(all_subjects), \
        "A subject appears more than once"
    assert dev == train | val, "development_subjects != train + validation"
    print(f"\n[OK] no subject overlap | all {len(all_subjects)} subjects "
          f"accounted for exactly once")

    # ---- 2. cross-validation ----
    print("\n" + "=" * 66)
    print(f"{splits['cv']['n_folds']}-FOLD SUBJECT-WISE CV (over development subjects)")
    print("=" * 66)
    print(f"{'fold':<6}{'tr_subj':>8}{'val_subj':>9}{'tr_win':>8}{'val_win':>9}"
          f"{'val_low':>9}{'val_high':>9}")
    seen_in_val = []
    for fold in splits["cv"]["folds"]:
        f_train, f_val = set(fold["train_subjects"]), set(fold["val_subjects"])
        assert not (f_train & f_val), f"Fold {fold['fold']}: train/val subject overlap"
        assert f_train | f_val == dev, f"Fold {fold['fold']}: does not cover all dev subjects"
        assert not ((f_train | f_val) & test), f"Fold {fold['fold']}: test subject leaked in CV"
        _, tr_win, _, _ = _summary(metadata, f_train)
        _, va_win, va_low, va_high = _summary(metadata, f_val)
        assert va_low == va_high, f"Fold {fold['fold']}: validation classes unbalanced"
        print(f"{fold['fold']:<6}{len(f_train):>8}{len(f_val):>9}{tr_win:>8}"
              f"{va_win:>9}{va_low:>9}{va_high:>9}")
        seen_in_val += list(f_val)

    assert sorted(seen_in_val) == sorted(dev), \
        "Each development subject must be in exactly one CV validation fold"
    print("\n[OK] no subject overlap inside any fold | every development subject "
          "is validated exactly once | test subjects never used in CV")


# ------------------------------------------------------------------
def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_dir = repo_root / "processed_data" / "contract_v1"
    parser = argparse.ArgumentParser(description="Create subject-wise splits and CV folds.")
    parser.add_argument("--metadata", type=Path, default=default_dir / "metadata.csv")
    parser.add_argument("--out", type=Path, default=default_dir / "splits.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"{args.out} exists. Use --overwrite to replace it.")

    metadata = pd.read_csv(args.metadata)
    needed = {"subject", "label"}
    if not needed.issubset(metadata.columns):
        raise ValueError(f"metadata.csv must contain columns {needed}")

    # Both conditions must exist for every subject so pairs stay together
    per_subject = metadata.groupby("subject")["label"].nunique()
    assert (per_subject == 2).all(), "Some subject lacks a low or a high recording"

    splits = make_splits(metadata["subject"].unique(), args.seed, args.n_val, args.n_folds)
    validate_splits(splits, metadata)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()