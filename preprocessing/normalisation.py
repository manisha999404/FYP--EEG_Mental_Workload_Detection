#!/usr/bin/env python3
"""
Step 4 (revised): does raw-EEG normalization help? Subject-wise CV on development subjects only.

Raw-EEG scaling variants (constants fitted on each fold's TRAINING subjects only):
  A_none      no scaling
  B_zscore    channel-wise (X-mean)/std over train windows x time
  C_robust    channel-wise (X-median)/(1.4826*MAD) over train windows x time

Feature sets (extraction identical across variants):
  logabs        PRIMARY   log10(abs band power + eps), 5 bands x 14 ch
  logabs_rms    PRIMARY+  logabs + linear RMS per channel (amplitude features)
  relative      SECONDARY descriptive only (cancels per-channel scaling; NOT used for the decision)

Models (same Logistic Regression in both):
  LR_std        StandardScaler + LR          (primary model)
  LR_nostd      LR without feature standardization (sensitivity: scaling can matter here)

Reads (never writes): X_filtered.npy, metadata.csv, splits.json. Test subjects are never used.
Writes only to results/step4_normalization/.  Run from repo root:
    python preprocessing/step4_normalization.py
"""
import argparse, glob, json, os, platform, sys
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
SUBJECT_CANDIDATES = ["subject", "subject_id", "subj", "subj_id", "participant", "participant_id", "sub", "pid"]
LABEL_CANDIDATES = ["label", "y", "class", "target", "condition", "category", "group"]
VARIANTS = ["A_none", "B_zscore", "C_robust"]
FEATURE_SETS = ["logabs", "logabs_rms", "relative"]
PRIMARY_SETS = ["logabs", "logabs_rms"]
MODELS = ["LR_std", "LR_nostd"]
METRICS = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]


# ------------------------------------------------------------------ IO helpers
def find_one(root, name, explicit):
    """Locate a file. Never auto-chooses: several candidates -> error, pass the path explicitly."""
    if explicit:
        if not os.path.isfile(explicit):
            sys.exit(f"[ERROR] {name}: file not found: {explicit}")
        return explicit
    hits = sorted(p for p in glob.glob(os.path.join(root, "**", name), recursive=True)
                  if "step4_normalization" not in p)
    if not hits:
        sys.exit(f"[ERROR] could not find {name} under {root}; pass it explicitly.")
    if len(hits) > 1:
        sys.exit(f"[ERROR] several {name} found, refusing to choose: {hits}. Pass the intended one explicitly.")
    return hits[0]


META_CONTRACT = os.path.join("processed_data", "contract_v1", "metadata.csv")


def pick_col(df, explicit, candidates, what):
    if explicit:
        if explicit not in df.columns:
            sys.exit(f"[ERROR] {what} column '{explicit}' not in {list(df.columns)}")
        return explicit
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    sys.exit(f"[ERROR] cannot auto-detect {what} column. Columns: {list(df.columns)}. Use --{what}-col.")


def _first_key(d, options):
    for k in d:
        kl = str(k).lower()
        if any(kl == o or kl.startswith(o) for o in options):
            return k
    return None


def find_folds(obj):
    def as_fold(d):
        if not isinstance(d, dict):
            return None
        tk, vk = _first_key(d, ["train"]), _first_key(d, ["val", "valid"])
        if tk is not None and vk is not None and isinstance(d[tk], list) and isinstance(d[vk], list):
            return d[tk], d[vk]
        return None

    if isinstance(obj, dict):
        for k, v in obj.items():
            if "fold" in str(k).lower() or str(k).lower() in ("cv", "cv_splits"):
                items = list(v.values()) if isinstance(v, dict) else v if isinstance(v, list) else None
                if items:
                    folds = [as_fold(i) for i in items]
                    if all(f is not None for f in folds):
                        return folds
        for v in obj.values():
            r = find_folds(v)
            if r:
                return r
    elif isinstance(obj, list):
        folds = [as_fold(i) for i in obj]
        if obj and all(f is not None for f in folds):
            return folds
        for v in obj:
            r = find_folds(v)
            if r:
                return r
    return None


def find_test_subjects(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower().startswith("test") and isinstance(v, list):
                return v
        for v in obj.values():
            r = find_test_subjects(v)
            if r is not None:
                return r
    return None


# ------------------------------------------------------------------ scaling (train-only fit)
def fit_scaler(X_train, variant):
    C = X_train.shape[2]
    flat = X_train.reshape(-1, C)
    if variant == "A_none":
        return np.zeros((1, 1, C)), np.ones((1, 1, C))
    if variant == "B_zscore":
        center, scale = flat.mean(0), flat.std(0)
    elif variant == "C_robust":
        center = np.median(flat, 0)
        scale = 1.4826 * np.median(np.abs(flat - center), 0)
    else:
        raise ValueError(variant)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return center.reshape(1, 1, C), scale.reshape(1, 1, C)


# ------------------------------------------------------------------ features (identical for all variants)
def band_powers(X, fs, nperseg):
    """Absolute band power (N, C, n_bands) and total 1-45 Hz power (N, C)."""
    nyq = fs / 2.0
    f, psd = welch(X, fs=fs, nperseg=min(nperseg, X.shape[1]), axis=1, detrend="constant")  # (N,F,C)
    bp = []
    for lo, hi in BANDS.values():
        m = (f >= lo) & (f < min(hi, nyq))
        bp.append(np.trapezoid(psd[:, m, :], f[m], axis=1))
    tm = (f >= 1.0) & (f < min(45.0, nyq))
    total = np.trapezoid(psd[:, tm, :], f[tm], axis=1)
    return np.stack(bp, axis=2), total


def extract_features(X, fs, nperseg, eps):
    bp, total = band_powers(X, fs, nperseg)
    n = X.shape[0]
    logabs = np.log10(bp + eps).reshape(n, -1)
    rms = np.sqrt(np.mean(X ** 2, axis=1))  # (N, C), linear amplitude
    relative = (bp / (total[:, :, None] + 1e-20)).reshape(n, -1)
    return {"logabs": logabs, "logabs_rms": np.hstack([logabs, rms]), "relative": relative}


# ------------------------------------------------------------------ models / metrics
def make_model(kind, seed):
    lr = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    return make_pipeline(StandardScaler(), lr) if kind == "LR_std" else make_pipeline(lr)


def compute_metrics(model, Xva, yva, classes):
    pred = model.predict(Xva)
    binary = len(classes) == 2
    out = {
        "accuracy": accuracy_score(yva, pred),
        "balanced_accuracy": balanced_accuracy_score(yva, pred),
        "f1": f1_score(yva, pred, average="binary", pos_label=classes[1], zero_division=0) if binary
        else f1_score(yva, pred, average="macro", zero_division=0),
        "roc_auc": np.nan,
    }
    try:
        proba = model.predict_proba(Xva)
        if binary and len(np.unique(yva)) == 2:
            out["roc_auc"] = roc_auc_score(yva, proba[:, 1])
        elif not binary and len(np.unique(yva)) == len(classes):
            out["roc_auc"] = roc_auc_score(yva, proba, multi_class="ovr", average="macro", labels=classes)
    except ValueError:
        pass
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--x"); ap.add_argument("--splits")
    ap.add_argument("--metadata", help=f"default: <root>/{META_CONTRACT} (explicit, never auto-detected)")
    ap.add_argument("--window-col"); ap.add_argument("--recording-col"); ap.add_argument("--condition-col")
    ap.add_argument("--expected-split-subjects", default="30,8,10")
    ap.add_argument("--expected-split-windows", default="4440,1184,1480")
    ap.add_argument("--subject-col"); ap.add_argument("--label-col")
    ap.add_argument("--fs", type=float, default=128.0)
    ap.add_argument("--nperseg", type=int, default=256)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--expected-shape", default="7104,512,14")
    ap.add_argument("--out-dir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = args.out_dir or os.path.join(root, "results", "step4_normalization")
    os.makedirs(out_dir, exist_ok=True)
    x_path = find_one(root, "X_filtered.npy", args.x)
    meta_path = args.metadata or os.path.join(root, META_CONTRACT)
    if not os.path.isfile(meta_path):
        sys.exit(f"[ERROR] contract metadata not found: {meta_path}")
    if not args.metadata and os.path.normpath(meta_path).split(os.sep)[-3:] != ["processed_data", "contract_v1", "metadata.csv"]:
        sys.exit("[ERROR] metadata is not processed_data/contract_v1/metadata.csv")
    splits_path = find_one(root, "splits.json", args.splits)
    print(f"X: {x_path}\nmetadata: {meta_path}\nsplits: {splits_path}")

    X = np.load(x_path, mmap_mode="r")
    exp = tuple(int(v) for v in args.expected_shape.split(","))
    if X.shape != exp:
        sys.exit(f"[ERROR] X shape {X.shape} != expected {exp}")
    meta = pd.read_csv(meta_path)
    if len(meta) != X.shape[0]:
        sys.exit(f"[ERROR] metadata rows ({len(meta)}) != X windows ({X.shape[0]}).")
    scol = pick_col(meta, args.subject_col, SUBJECT_CANDIDATES, "subject")
    lcol = pick_col(meta, args.label_col, LABEL_CANDIDATES, "label")
    subj = meta[scol].astype(str).to_numpy()
    classes_raw = np.unique(meta[lcol].to_numpy())
    y = np.searchsorted(classes_raw, meta[lcol].to_numpy())
    classes = np.arange(len(classes_raw))
    print(f"subject col='{scol}' ({len(np.unique(subj))} subjects), label col='{lcol}' classes={list(classes_raw)}")

    with open(splits_path) as fh:
        splits = json.load(fh)
    folds = find_folds(splits)
    if not folds:
        sys.exit(f"[ERROR] no CV folds in splits.json. Top-level keys: "
                 f"{list(splits)[:20] if isinstance(splits, dict) else type(splits)}")
    test_subj = {str(s) for s in (find_test_subjects(splits) or [])}
    known, dev_all = set(subj), set()
    for tr, va in folds:
        for s in list(tr) + list(va):
            if str(s) not in known:
                sys.exit(f"[ERROR] fold subject '{s}' not in metadata '{scol}' (IDs differ or splits hold indices).")
            dev_all.add(str(s))
    if test_subj and (dev_all & test_subj):
        sys.exit(f"[ERROR] test subjects inside CV folds: {sorted(dev_all & test_subj)}")
    print(f"{len(folds)} folds; dev subjects: {len(dev_all)}; test subjects excluded: {len(test_subj)}")

    # ------------------------------------------------------------------ validation (metadata / X / splits contract)
    V = {"metadata_path": meta_path, "checks": {}}
    def ok(name, cond, detail):
        V["checks"][name] = {"passed": bool(cond), "detail": detail}
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")
        if not cond:
            json.dump(V, open(os.path.join(out_dir, "validation_summary.json"), "w"), indent=2, default=str)
            sys.exit(f"[ERROR] validation failed: {name}")

    wcol = pick_col(meta, args.window_col, ["window_idx", "window_index", "window_id", "window", "win_idx", "epoch_idx"], "window")
    rcol = pick_col(meta, args.recording_col, ["recording_id", "recording", "rec_id", "record_id", "session_id", "session", "file", "filename"], "recording")
    ccol = pick_col(meta, args.condition_col, ["condition", "cond", "task", "state"], "condition")
    ok("row_count_matches_X", len(meta) == X.shape[0], f"metadata rows={len(meta)}, X windows={X.shape[0]}")
    ok("no_missing_in_key_columns", not meta[[scol, lcol, ccol, wcol, rcol]].isna().any().any(),
       f"checked {[scol, lcol, ccol, wcol, rcol]}")
    ok("window_index_unique_per_recording", not meta.duplicated([rcol, wcol]).any(),
       f"(recording, window) pairs unique; global 0..N-1 order={bool((meta[wcol].to_numpy() == np.arange(len(meta))).all())}")
    ok("recording_belongs_to_one_subject", (meta.groupby(rcol)[scol].nunique() == 1).all(),
       f"{meta[rcol].nunique()} recordings, {meta[scol].nunique()} subjects")
    ok("subject_label_condition_consistent",
       (meta.groupby(ccol)[lcol].nunique() == 1).all() and (meta.groupby(rcol)[ccol].nunique() == 1).all(),
       f"each condition maps to one label; each recording has one condition; conditions={sorted(meta[ccol].astype(str).unique())}")

    all_subj = set(subj)
    ok("split_subjects_exist_in_metadata", (dev_all | test_subj) <= all_subj, f"{len(dev_all | test_subj)} split subjects, all present")
    ok("test_subjects_found_in_splits", len(test_subj) > 0, f"{len(test_subj)} test subjects")
    ok("test_disjoint_from_development", not (test_subj & dev_all), "test ∩ (train ∪ val across folds) = ∅")
    top = splits if isinstance(splits, dict) else {}
    def top_list(prefixes):
        for k, v in top.items():
            if any(str(k).lower().startswith(q) for q in prefixes) and isinstance(v, list) and v and not isinstance(v[0], (dict, list)):
                return {str(i) for i in v}
        for v in top.values():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if any(str(k2).lower().startswith(q) for q in prefixes) and isinstance(v2, list) and v2 and not isinstance(v2[0], (dict, list)):
                        return {str(i) for i in v2}
        return None
    g_tr, g_va = top_list(["train"]), top_list(["val", "valid"])
    if g_tr is not None and g_va is not None:
        ok("train_val_test_disjoint_global", not (g_tr & g_va) and not (g_tr & test_subj) and not (g_va & test_subj),
           f"train={len(g_tr)}, val={len(g_va)}, test={len(test_subj)}; pairwise overlaps = 0")
        n_exp = tuple(int(v) for v in args.expected_split_subjects.split(","))
        ok("split_subject_counts", (len(g_tr), len(g_va), len(test_subj)) == n_exp,
           f"{(len(g_tr), len(g_va), len(test_subj))} vs expected {n_exp}")
        w_exp = tuple(int(v) for v in args.expected_split_windows.split(","))
        w_got = tuple(int(np.isin(subj, list(g)).sum()) for g in (g_tr, g_va, test_subj))
        ok("split_window_counts", w_got == w_exp, f"{w_got} vs expected {w_exp}")
        ok("dev_equals_train_plus_val", dev_all == (g_tr | g_va), f"{len(dev_all)} development subjects")
    for fi_, (a, b) in enumerate(folds):
        a, b = {str(i) for i in a}, {str(i) for i in b}
        ok(f"fold{fi_}_train_val_disjoint_and_no_test", not (a & b) and not ((a | b) & test_subj),
           f"train={len(a)}, val={len(b)}")
    vals = [{str(i) for i in b} for _, b in folds]
    ok("cv_validation_sets_mutually_disjoint", sum(len(v) for v in vals) == len(set().union(*vals)),
       f"{len(vals)} folds, {len(set().union(*vals))} distinct validation subjects")

    # Guard: X is only ever read at development windows; test windows are never loaded or fitted on.
    dev_mask = np.isin(subj, list(dev_all))
    test_mask = np.isin(subj, list(test_subj))
    def load_dev(idx):
        assert not test_mask[idx].any(), "attempt to read test windows"
        return np.asarray(X[idx], dtype=np.float64)
    dev_idx_all = np.where(dev_mask)[0]
    finite = all(np.isfinite(load_dev(dev_idx_all[i:i + 500])).all() for i in range(0, len(dev_idx_all), 500))
    ok("X_development_windows_finite", finite, f"{len(dev_idx_all)} development windows checked; test windows not read")
    V["test_windows_read"] = 0

    rows, diag = [], []
    for fi, (tr_s, va_s) in enumerate(folds):
        tr_s, va_s = {str(s) for s in tr_s}, {str(s) for s in va_s}
        if tr_s & va_s:
            sys.exit(f"[ERROR] fold {fi}: train/val overlap")
        if test_subj and ((tr_s | va_s) & test_subj):
            sys.exit(f"[ERROR] fold {fi}: test leakage")
        tr_idx = np.where(np.isin(subj, list(tr_s)))[0]
        va_idx = np.where(np.isin(subj, list(va_s)))[0]
        assert set(subj[tr_idx]) == tr_s and set(subj[va_idx]) == va_s, f"fold {fi}: index/subject mismatch"
        Xtr = load_dev(tr_idx)  # scaler fitted on these windows only
        Xva = load_dev(va_idx)
        ytr, yva = y[tr_idx], y[va_idx]

        feats = {}
        for v in VARIANTS:
            c, s = fit_scaler(Xtr, v)  # fitted on training subjects only
            feats[v] = (extract_features((Xtr - c) / s, args.fs, args.nperseg, args.eps),
                        extract_features((Xva - c) / s, args.fs, args.nperseg, args.eps))

        # diagnostic: max |difference| of what LR_std receives (train-standardized val features) vs A_none
        for fs_name in FEATURE_SETS:
            def seen(v):
                Ftr, Fva = feats[v][0][fs_name], feats[v][1][fs_name]
                return (Fva - Ftr.mean(0)) / (Ftr.std(0) + 1e-12)
            base = seen("A_none")
            for v in VARIANTS[1:]:
                diag.append({"fold": fi, "feature_set": fs_name, "variant": v,
                             "max_abs_diff_model_input": float(np.max(np.abs(seen(v) - base)))})

        for fs_name in FEATURE_SETS:
            for model_kind in MODELS:
                for v in VARIANTS:
                    Ftr, Fva = feats[v][0][fs_name], feats[v][1][fs_name]
                    model = make_model(model_kind, args.seed).fit(Ftr, ytr)
                    m = compute_metrics(model, Fva, yva, classes)
                    rows.append({"fold": fi, "feature_set": fs_name,
                                 "role": "primary" if fs_name in PRIMARY_SETS else "secondary_descriptive",
                                 "model": model_kind, "variant": v,
                                 "n_train_subjects": len(tr_s), "n_val_subjects": len(va_s),
                                 "n_train_windows": len(tr_idx), "n_val_windows": len(va_idx), **m})
                    print(f"fold {fi} {fs_name:10s} {model_kind:8s} {v:9s} "
                          + " ".join(f"{k}={m[k]:.4f}" for k in METRICS))

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(os.path.join(out_dir, "fold_results.csv"), index=False)
    diag_df = pd.DataFrame(diag)
    diag_df.to_csv(os.path.join(out_dir, "equivalence_diagnostic.csv"), index=False)

    srows = []
    for (fs_name, model_kind, v), sub in fold_df.groupby(["feature_set", "model", "variant"], sort=False):
        r = {"feature_set": fs_name, "role": sub.role.iloc[0], "model": model_kind, "variant": v, "n_folds": len(sub)}
        for k in METRICS:
            r[f"{k}_mean"], r[f"{k}_sd"] = sub[k].mean(), sub[k].std(ddof=1)
            r[f"{k}_n_valid"] = int(sub[k].notna().sum())
        srows.append(r)
    summ = pd.DataFrame(srows)
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "paths": {"X": x_path, "metadata": meta_path, "splits": splits_path, "out_dir": out_dir},
        "X_shape": list(X.shape), "subject_col": scol, "label_col": lcol, "classes": [str(c) for c in classes_raw],
        "n_folds": len(folds), "n_dev_subjects": len(dev_all), "n_test_subjects_excluded": len(test_subj),
        "variants": {"A_none": "no scaling",
                     "B_zscore": "channel-wise (X-mean)/std over train windows x time",
                     "C_robust": "channel-wise (X-median)/(1.4826*MAD) over train windows x time"},
        "feature_sets": {"logabs": "log10(abs band power + eps), PRIMARY",
                         "logabs_rms": "logabs + linear RMS per channel, PRIMARY",
                         "relative": "relative band power, SECONDARY descriptive only"},
        "welch": {"fs": args.fs, "nperseg": args.nperseg, "detrend": "constant", "bands_hz": BANDS,
                  "relative_total_range_hz": [1, 45]},
        "eps": args.eps,
        "models": {"LR_std": "StandardScaler + LogisticRegression(max_iter=5000, class_weight=balanced, C=1)",
                   "LR_nostd": "LogisticRegression(max_iter=5000, class_weight=balanced, C=1), no feature standardization"},
        "seed": args.seed, "test_data_used": False, "windows_removed": 0,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }
    config["metadata_contract"] = META_CONTRACT
    config["validation_all_passed"] = all(c["passed"] for c in V["checks"].values())
    with open(os.path.join(out_dir, "validation_summary.json"), "w") as fh:
        json.dump(V, fh, indent=2, default=str)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2, default=str)

    L = ["STEP 4 (REVISED) - NORMALIZATION COMPARISON, subject-wise CV on development subjects only",
         "=" * 78,
         f"Folds: {len(folds)} | dev subjects: {len(dev_all)} | test subjects untouched: {len(test_subj)} | windows removed: 0",
         "Scaling constants fitted on each fold's training subjects only; validation only transformed.",
         "Mean +/- SD across folds (ddof=1; folds where ROC-AUC is undefined are skipped).", ""]
    for role, title in [("primary", "PRIMARY (amplitude-retaining features) - basis for the normalization decision"),
                        ("secondary_descriptive", "SECONDARY / DESCRIPTIVE ONLY (relative band power) - not used for the decision")]:
        L.append(title)
        L.append("-" * len(title))
        for (fs_name, model_kind), g in summ[summ.role == role].groupby(["feature_set", "model"], sort=False):
            L.append(f"[{fs_name} | {model_kind}]")
            for _, r in g.iterrows():
                L.append(f"  {r.variant:9s} " + "  ".join(f"{k}={r[k + '_mean']:.4f}+/-{r[k + '_sd']:.4f}" for k in METRICS))
        L.append("")
    L.append(f"Metadata: {meta_path}")
    L.append("Validation: " + "; ".join(f"{k}={'PASS' if c['passed'] else 'FAIL'}" for k, c in V["checks"].items()))
    L.append("")
    L.append("Equivalence diagnostic: max |difference| in the features LR_std actually receives "
             "(train-standardized validation features) vs A_none, max over folds:")
    for (fs_name, v), g in diag_df.groupby(["feature_set", "variant"], sort=False):
        L.append(f"  {fs_name:10s} {v:9s} {g.max_abs_diff_model_input.max():.3e}")
    L += ["",
          "How to read this:",
          "  * Scaling a channel by s adds a constant (-2*log10 s) to all log10 band powers of that channel; "
          "LR's intercept / feature standardization absorbs it. Differences near 0 in the diagnostic mean "
          "LR_std cannot distinguish A/B/C for that feature set (only eps and median-centering can).",
          "  * Linear RMS and LR_nostd are where scaling can genuinely change results; compare those rows.",
          "  * Differences between variants smaller than the across-fold SD should not be called an improvement.",
          "  * Relative band power is retained for description only; no test data was used."]
    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()









# #!/usr/bin/env python3
# """
# Step 4 (revised): does raw-EEG normalization help? Subject-wise CV on development subjects only.

# Raw-EEG scaling variants (constants fitted on each fold's TRAINING subjects only):
#   A_none      no scaling
#   B_zscore    channel-wise (X-mean)/std over train windows x time
#   C_robust    channel-wise (X-median)/(1.4826*MAD) over train windows x time

# Feature sets (extraction identical across variants):
#   logabs        PRIMARY   log10(abs band power + eps), 5 bands x 14 ch
#   logabs_rms    PRIMARY+  logabs + linear RMS per channel (amplitude features)
#   relative      SECONDARY descriptive only (cancels per-channel scaling; NOT used for the decision)

# Models (same Logistic Regression in both):
#   LR_std        StandardScaler + LR          (primary model)
#   LR_nostd      LR without feature standardization (sensitivity: scaling can matter here)

# Reads (never writes): X_filtered.npy, metadata.csv, splits.json. Test subjects are never used.
# Writes only to results/step4_normalization/.  Run from repo root:
#     python preprocessing/step4_normalization.py
# """
# import argparse, glob, json, os, platform, sys
# from datetime import datetime

# import numpy as np
# import pandas as pd
# import sklearn
# from scipy.signal import welch
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
# SUBJECT_CANDIDATES = ["subject", "subject_id", "subj", "subj_id", "participant", "participant_id", "sub", "pid"]
# LABEL_CANDIDATES = ["label", "y", "class", "target", "condition", "category", "group"]
# VARIANTS = ["A_none", "B_zscore", "C_robust"]
# FEATURE_SETS = ["logabs", "logabs_rms", "relative"]
# PRIMARY_SETS = ["logabs", "logabs_rms"]
# MODELS = ["LR_std", "LR_nostd"]
# METRICS = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]


# # ------------------------------------------------------------------ IO helpers
# def find_one(root, name, explicit):
#     if explicit:
#         if not os.path.isfile(explicit):
#             sys.exit(f"[ERROR] {name}: file not found: {explicit}")
#         return explicit
#     hits = [p for p in glob.glob(os.path.join(root, "**", name), recursive=True)
#             if "step4_normalization" not in p]
#     if not hits:
#         sys.exit(f"[ERROR] could not find {name} under {root}; pass it explicitly.")
#     hits.sort(key=lambda p: (len(p.split(os.sep)), p))
#     if len(hits) > 1:
#         print(f"[WARN] several {name} found, using {hits[0]} (others: {hits[1:]}).")
#     return hits[0]


# def pick_col(df, explicit, candidates, what):
#     if explicit:
#         if explicit not in df.columns:
#             sys.exit(f"[ERROR] {what} column '{explicit}' not in {list(df.columns)}")
#         return explicit
#     lower = {c.lower(): c for c in df.columns}
#     for c in candidates:
#         if c in lower:
#             return lower[c]
#     sys.exit(f"[ERROR] cannot auto-detect {what} column. Columns: {list(df.columns)}. Use --{what}-col.")


# def _first_key(d, options):
#     for k in d:
#         kl = str(k).lower()
#         if any(kl == o or kl.startswith(o) for o in options):
#             return k
#     return None


# def find_folds(obj):
#     def as_fold(d):
#         if not isinstance(d, dict):
#             return None
#         tk, vk = _first_key(d, ["train"]), _first_key(d, ["val", "valid"])
#         if tk is not None and vk is not None and isinstance(d[tk], list) and isinstance(d[vk], list):
#             return d[tk], d[vk]
#         return None

#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             if "fold" in str(k).lower() or str(k).lower() in ("cv", "cv_splits"):
#                 items = list(v.values()) if isinstance(v, dict) else v if isinstance(v, list) else None
#                 if items:
#                     folds = [as_fold(i) for i in items]
#                     if all(f is not None for f in folds):
#                         return folds
#         for v in obj.values():
#             r = find_folds(v)
#             if r:
#                 return r
#     elif isinstance(obj, list):
#         folds = [as_fold(i) for i in obj]
#         if obj and all(f is not None for f in folds):
#             return folds
#         for v in obj:
#             r = find_folds(v)
#             if r:
#                 return r
#     return None


# def find_test_subjects(obj):
#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             if str(k).lower().startswith("test") and isinstance(v, list):
#                 return v
#         for v in obj.values():
#             r = find_test_subjects(v)
#             if r is not None:
#                 return r
#     return None


# # ------------------------------------------------------------------ scaling (train-only fit)
# def fit_scaler(X_train, variant):
#     C = X_train.shape[2]
#     flat = X_train.reshape(-1, C)
#     if variant == "A_none":
#         return np.zeros((1, 1, C)), np.ones((1, 1, C))
#     if variant == "B_zscore":
#         center, scale = flat.mean(0), flat.std(0)
#     elif variant == "C_robust":
#         center = np.median(flat, 0)
#         scale = 1.4826 * np.median(np.abs(flat - center), 0)
#     else:
#         raise ValueError(variant)
#     scale = np.where(scale < 1e-12, 1.0, scale)
#     return center.reshape(1, 1, C), scale.reshape(1, 1, C)


# # ------------------------------------------------------------------ features (identical for all variants)
# def band_powers(X, fs, nperseg):
#     """Absolute band power (N, C, n_bands) and total 1-45 Hz power (N, C)."""
#     nyq = fs / 2.0
#     f, psd = welch(X, fs=fs, nperseg=min(nperseg, X.shape[1]), axis=1, detrend="constant")  # (N,F,C)
#     bp = []
#     for lo, hi in BANDS.values():
#         m = (f >= lo) & (f < min(hi, nyq))
#         bp.append(np.trapezoid(psd[:, m, :], f[m], axis=1))
#     tm = (f >= 1.0) & (f < min(45.0, nyq))
#     total = np.trapezoid(psd[:, tm, :], f[tm], axis=1)
#     return np.stack(bp, axis=2), total


# def extract_features(X, fs, nperseg, eps):
#     bp, total = band_powers(X, fs, nperseg)
#     n = X.shape[0]
#     logabs = np.log10(bp + eps).reshape(n, -1)
#     rms = np.sqrt(np.mean(X ** 2, axis=1))  # (N, C), linear amplitude
#     relative = (bp / (total[:, :, None] + 1e-20)).reshape(n, -1)
#     return {"logabs": logabs, "logabs_rms": np.hstack([logabs, rms]), "relative": relative}


# # ------------------------------------------------------------------ models / metrics
# def make_model(kind, seed):
#     lr = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
#     return make_pipeline(StandardScaler(), lr) if kind == "LR_std" else make_pipeline(lr)


# def compute_metrics(model, Xva, yva, classes):
#     pred = model.predict(Xva)
#     binary = len(classes) == 2
#     out = {
#         "accuracy": accuracy_score(yva, pred),
#         "balanced_accuracy": balanced_accuracy_score(yva, pred),
#         "f1": f1_score(yva, pred, average="binary", pos_label=classes[1], zero_division=0) if binary
#         else f1_score(yva, pred, average="macro", zero_division=0),
#         "roc_auc": np.nan,
#     }
#     try:
#         proba = model.predict_proba(Xva)
#         if binary and len(np.unique(yva)) == 2:
#             out["roc_auc"] = roc_auc_score(yva, proba[:, 1])
#         elif not binary and len(np.unique(yva)) == len(classes):
#             out["roc_auc"] = roc_auc_score(yva, proba, multi_class="ovr", average="macro", labels=classes)
#     except ValueError:
#         pass
#     return out


# # ------------------------------------------------------------------ main
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     ap.add_argument("--x"); ap.add_argument("--metadata"); ap.add_argument("--splits")
#     ap.add_argument("--subject-col"); ap.add_argument("--label-col")
#     ap.add_argument("--fs", type=float, default=128.0)
#     ap.add_argument("--nperseg", type=int, default=256)
#     ap.add_argument("--eps", type=float, default=1e-12)
#     ap.add_argument("--expected-shape", default="7104,512,14")
#     ap.add_argument("--out-dir")
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()

#     root = os.path.abspath(args.root)
#     out_dir = args.out_dir or os.path.join(root, "results", "step4_normalization")
#     os.makedirs(out_dir, exist_ok=True)
#     x_path = find_one(root, "X_filtered.npy", args.x)
#     meta_path = find_one(root, "metadata.csv", args.metadata)
#     splits_path = find_one(root, "splits.json", args.splits)
#     print(f"X: {x_path}\nmetadata: {meta_path}\nsplits: {splits_path}")

#     X = np.load(x_path, mmap_mode="r")
#     exp = tuple(int(v) for v in args.expected_shape.split(","))
#     if X.shape != exp:
#         sys.exit(f"[ERROR] X shape {X.shape} != expected {exp}")
#     meta = pd.read_csv(meta_path)
#     if len(meta) != X.shape[0]:
#         sys.exit(f"[ERROR] metadata rows ({len(meta)}) != X windows ({X.shape[0]}).")
#     scol = pick_col(meta, args.subject_col, SUBJECT_CANDIDATES, "subject")
#     lcol = pick_col(meta, args.label_col, LABEL_CANDIDATES, "label")
#     subj = meta[scol].astype(str).to_numpy()
#     classes_raw = np.unique(meta[lcol].to_numpy())
#     y = np.searchsorted(classes_raw, meta[lcol].to_numpy())
#     classes = np.arange(len(classes_raw))
#     print(f"subject col='{scol}' ({len(np.unique(subj))} subjects), label col='{lcol}' classes={list(classes_raw)}")

#     with open(splits_path) as fh:
#         splits = json.load(fh)
#     folds = find_folds(splits)
#     if not folds:
#         sys.exit(f"[ERROR] no CV folds in splits.json. Top-level keys: "
#                  f"{list(splits)[:20] if isinstance(splits, dict) else type(splits)}")
#     test_subj = {str(s) for s in (find_test_subjects(splits) or [])}
#     known, dev_all = set(subj), set()
#     for tr, va in folds:
#         for s in list(tr) + list(va):
#             if str(s) not in known:
#                 sys.exit(f"[ERROR] fold subject '{s}' not in metadata '{scol}' (IDs differ or splits hold indices).")
#             dev_all.add(str(s))
#     if test_subj and (dev_all & test_subj):
#         sys.exit(f"[ERROR] test subjects inside CV folds: {sorted(dev_all & test_subj)}")
#     print(f"{len(folds)} folds; dev subjects: {len(dev_all)}; test subjects excluded: {len(test_subj)}")

#     rows, diag = [], []
#     for fi, (tr_s, va_s) in enumerate(folds):
#         tr_s, va_s = {str(s) for s in tr_s}, {str(s) for s in va_s}
#         if tr_s & va_s:
#             sys.exit(f"[ERROR] fold {fi}: train/val overlap")
#         if test_subj and ((tr_s | va_s) & test_subj):
#             sys.exit(f"[ERROR] fold {fi}: test leakage")
#         tr_idx = np.where(np.isin(subj, list(tr_s)))[0]
#         va_idx = np.where(np.isin(subj, list(va_s)))[0]
#         Xtr = np.asarray(X[tr_idx], dtype=np.float64)
#         Xva = np.asarray(X[va_idx], dtype=np.float64)
#         ytr, yva = y[tr_idx], y[va_idx]

#         feats = {}
#         for v in VARIANTS:
#             c, s = fit_scaler(Xtr, v)  # fitted on training subjects only
#             feats[v] = (extract_features((Xtr - c) / s, args.fs, args.nperseg, args.eps),
#                         extract_features((Xva - c) / s, args.fs, args.nperseg, args.eps))

#         # diagnostic: max |difference| of what LR_std receives (train-standardized val features) vs A_none
#         for fs_name in FEATURE_SETS:
#             def seen(v):
#                 Ftr, Fva = feats[v][0][fs_name], feats[v][1][fs_name]
#                 return (Fva - Ftr.mean(0)) / (Ftr.std(0) + 1e-12)
#             base = seen("A_none")
#             for v in VARIANTS[1:]:
#                 diag.append({"fold": fi, "feature_set": fs_name, "variant": v,
#                              "max_abs_diff_model_input": float(np.max(np.abs(seen(v) - base)))})

#         for fs_name in FEATURE_SETS:
#             for model_kind in MODELS:
#                 for v in VARIANTS:
#                     Ftr, Fva = feats[v][0][fs_name], feats[v][1][fs_name]
#                     model = make_model(model_kind, args.seed).fit(Ftr, ytr)
#                     m = compute_metrics(model, Fva, yva, classes)
#                     rows.append({"fold": fi, "feature_set": fs_name,
#                                  "role": "primary" if fs_name in PRIMARY_SETS else "secondary_descriptive",
#                                  "model": model_kind, "variant": v,
#                                  "n_train_subjects": len(tr_s), "n_val_subjects": len(va_s),
#                                  "n_train_windows": len(tr_idx), "n_val_windows": len(va_idx), **m})
#                     print(f"fold {fi} {fs_name:10s} {model_kind:8s} {v:9s} "
#                           + " ".join(f"{k}={m[k]:.4f}" for k in METRICS))

#     fold_df = pd.DataFrame(rows)
#     fold_df.to_csv(os.path.join(out_dir, "fold_results.csv"), index=False)
#     diag_df = pd.DataFrame(diag)
#     diag_df.to_csv(os.path.join(out_dir, "equivalence_diagnostic.csv"), index=False)

#     srows = []
#     for (fs_name, model_kind, v), sub in fold_df.groupby(["feature_set", "model", "variant"], sort=False):
#         r = {"feature_set": fs_name, "role": sub.role.iloc[0], "model": model_kind, "variant": v, "n_folds": len(sub)}
#         for k in METRICS:
#             r[f"{k}_mean"], r[f"{k}_sd"] = sub[k].mean(), sub[k].std(ddof=1)
#             r[f"{k}_n_valid"] = int(sub[k].notna().sum())
#         srows.append(r)
#     summ = pd.DataFrame(srows)
#     summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

#     config = {
#         "created": datetime.now().isoformat(timespec="seconds"),
#         "paths": {"X": x_path, "metadata": meta_path, "splits": splits_path, "out_dir": out_dir},
#         "X_shape": list(X.shape), "subject_col": scol, "label_col": lcol, "classes": [str(c) for c in classes_raw],
#         "n_folds": len(folds), "n_dev_subjects": len(dev_all), "n_test_subjects_excluded": len(test_subj),
#         "variants": {"A_none": "no scaling",
#                      "B_zscore": "channel-wise (X-mean)/std over train windows x time",
#                      "C_robust": "channel-wise (X-median)/(1.4826*MAD) over train windows x time"},
#         "feature_sets": {"logabs": "log10(abs band power + eps), PRIMARY",
#                          "logabs_rms": "logabs + linear RMS per channel, PRIMARY",
#                          "relative": "relative band power, SECONDARY descriptive only"},
#         "welch": {"fs": args.fs, "nperseg": args.nperseg, "detrend": "constant", "bands_hz": BANDS,
#                   "relative_total_range_hz": [1, 45]},
#         "eps": args.eps,
#         "models": {"LR_std": "StandardScaler + LogisticRegression(max_iter=5000, class_weight=balanced, C=1)",
#                    "LR_nostd": "LogisticRegression(max_iter=5000, class_weight=balanced, C=1), no feature standardization"},
#         "seed": args.seed, "test_data_used": False, "windows_removed": 0,
#         "versions": {"python": platform.python_version(), "numpy": np.__version__,
#                      "sklearn": sklearn.__version__, "pandas": pd.__version__},
#     }
#     with open(os.path.join(out_dir, "config.json"), "w") as fh:
#         json.dump(config, fh, indent=2, default=str)

#     L = ["STEP 4 (REVISED) - NORMALIZATION COMPARISON, subject-wise CV on development subjects only",
#          "=" * 78,
#          f"Folds: {len(folds)} | dev subjects: {len(dev_all)} | test subjects untouched: {len(test_subj)} | windows removed: 0",
#          "Scaling constants fitted on each fold's training subjects only; validation only transformed.",
#          "Mean +/- SD across folds (ddof=1; folds where ROC-AUC is undefined are skipped).", ""]
#     for role, title in [("primary", "PRIMARY (amplitude-retaining features) - basis for the normalization decision"),
#                         ("secondary_descriptive", "SECONDARY / DESCRIPTIVE ONLY (relative band power) - not used for the decision")]:
#         L.append(title)
#         L.append("-" * len(title))
#         for (fs_name, model_kind), g in summ[summ.role == role].groupby(["feature_set", "model"], sort=False):
#             L.append(f"[{fs_name} | {model_kind}]")
#             for _, r in g.iterrows():
#                 L.append(f"  {r.variant:9s} " + "  ".join(f"{k}={r[k + '_mean']:.4f}+/-{r[k + '_sd']:.4f}" for k in METRICS))
#         L.append("")
#     L.append("Equivalence diagnostic: max |difference| in the features LR_std actually receives "
#              "(train-standardized validation features) vs A_none, max over folds:")
#     for (fs_name, v), g in diag_df.groupby(["feature_set", "variant"], sort=False):
#         L.append(f"  {fs_name:10s} {v:9s} {g.max_abs_diff_model_input.max():.3e}")
#     L += ["",
#           "How to read this:",
#           "  * Scaling a channel by s adds a constant (-2*log10 s) to all log10 band powers of that channel; "
#           "LR's intercept / feature standardization absorbs it. Differences near 0 in the diagnostic mean "
#           "LR_std cannot distinguish A/B/C for that feature set (only eps and median-centering can).",
#           "  * Linear RMS and LR_nostd are where scaling can genuinely change results; compare those rows.",
#           "  * Differences between variants smaller than the across-fold SD should not be called an improvement.",
#           "  * Relative band power is retained for description only; no test data was used."]
#     with open(os.path.join(out_dir, "report.txt"), "w") as fh:
#         fh.write("\n".join(L) + "\n")
#     print("\n" + "\n".join(L))
#     print(f"\nSaved to {out_dir}")


# if __name__ == "__main__":
#     main()