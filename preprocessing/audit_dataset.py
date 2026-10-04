"""
audit_dataset.py  -  STEP 3: amplitude, artifact and spectral AUDIT
===================================================================
READ-ONLY and DESCRIPTIVE. This script
  * does not modify X_filtered.npy, metadata.csv or splits.json
    (SHA-256 of all three is checked before and after the run),
  * does not normalize, extract model features, or train anything,
  * does not reject any window and does not choose an artifact threshold.

Groups used in every table (column "group"):
  development : train + validation subjects (38). The only group that may
                inform pipeline decisions in Step 4.
  test        : the 10 frozen test subjects. DESCRIPTIVE ONLY - shown so the
                report can describe them, never used for fitting/decisions.
  all         : all 48 subjects (descriptive).

Artifact thresholds below are EXPLORATORY, not decisions.

Run from the repo root:
    python preprocessing/audit_dataset.py
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import LogNorm
from scipy import signal, stats

# ----------------------------------------------------------------------
# Configuration (all descriptive / exploratory)
# ----------------------------------------------------------------------
FS = 128
DEFAULT_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
# EXPLORATORY peak-to-peak thresholds in the units of the data (assumed uV).
EXPLORATORY_PTP_THRESHOLDS = (100, 150, 200)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "low_gamma": (30, 40)}  # 40 Hz = filter cutoff
WELCH_NPERSEG = 256      # 2 s -> 0.5 Hz resolution at 128 Hz
WELCH_NOVERLAP = 128
CHUNK = 500              # windows processed at once (memory control)

DEV_SPLITS = ("train", "validation")
SPLIT_RANK = {"train": 0, "validation": 1, "test": 2}
SPLIT_COLOR = {"train": "#1f77b4", "validation": "#2ca02c", "test": "#d62728"}
CLASS_COLOR = {"low": "#4c78a8", "high": "#e45756"}
GROUPS = ("development", "test", "all")


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def subjects_of(group, split_of: pd.Series):
    if group == "development":
        return split_of.index[split_of.isin(DEV_SPLITS)]
    if group == "test":
        return split_of.index[split_of == "test"]
    return split_of.index


def window_group_masks(meta: pd.DataFrame):
    return {"development": meta["split"].isin(DEV_SPLITS).to_numpy(),
            "test": (meta["split"] == "test").to_numpy(),
            "all": np.ones(len(meta), dtype=bool)}


def color_ticklabels(ax, labels_subjects, split_of):
    for tick, subject in zip(ax.get_xticklabels(), labels_subjects):
        tick.set_color(SPLIT_COLOR[split_of[subject]])


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"   figure: {path.name}")


def save_table(df, path):
    df.to_csv(path, index=False)
    print(f"   table : {path.name}")


# ----------------------------------------------------------------------
# Load inputs (read-only)
# ----------------------------------------------------------------------
def load_inputs(data_dir: Path):
    x_path, meta_path = data_dir / "X_filtered.npy", data_dir / "metadata.csv"
    split_path, cfg_path = data_dir / "splits.json", data_dir / "config.json"
    for p in (x_path, meta_path, split_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    X = np.load(x_path, mmap_mode="r")  # read-only memory map
    meta = pd.read_csv(meta_path)
    with open(split_path) as f:
        splits = json.load(f)

    channels, config = DEFAULT_CHANNELS, None
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = json.load(f)
        channels = config.get("channel_names", DEFAULT_CHANNELS)

    if X.ndim != 3 or X.shape[2] != len(channels):
        raise ValueError(f"Unexpected X shape {X.shape}")
    if len(X) != len(meta):
        raise ValueError("X and metadata.csv have different lengths")
    if config is not None and file_sha256(x_path) != config["X_sha256"]:
        # Hash of the file bytes differs from hash of array bytes by the .npy
        # header, so compare the array content hash instead:
        h = hashlib.sha256()
        for s in range(0, len(X), CHUNK):
            h.update(np.ascontiguousarray(X[s:s + CHUNK]).tobytes())
        if h.hexdigest() != config["X_sha256"]:
            raise ValueError("X_filtered.npy does not match the hash in config.json")

    split_of = {}
    for name, key in (("train", "train_subjects"), ("validation", "validation_subjects"),
                      ("test", "test_subjects")):
        for s in splits[key]:
            split_of[int(s)] = name
    meta["split"] = meta["subject"].map(split_of)
    if meta["split"].isna().any():
        raise ValueError("Some subjects in metadata.csv are not in splits.json")
    split_series = meta.drop_duplicates("subject").set_index("subject")["split"].sort_index()
    return X, meta, splits, channels, split_series, config


# ----------------------------------------------------------------------
# PART A - amplitude spread across subjects
# ----------------------------------------------------------------------
def part_a_amplitude(X, meta, channels, split_of, tab, fig):
    print("\n[A] Amplitude spread")
    rec_rows, subj_rows = [], []
    for (subject, condition), idx in meta.groupby(["subject", "condition"]).indices.items():
        block = np.asarray(X[np.sort(idx)], dtype=np.float64)
        std = block.reshape(-1, block.shape[2]).std(axis=0)
        for ch, s in zip(channels, std):
            rec_rows.append(dict(subject=subject, split=split_of[subject],
                                 condition=condition, channel=ch, std_uv=s))
    for subject, idx in meta.groupby("subject").indices.items():
        block = np.asarray(X[np.sort(idx)], dtype=np.float64)
        std = block.reshape(-1, block.shape[2]).std(axis=0)
        for ch, s in zip(channels, std):
            subj_rows.append(dict(subject=subject, split=split_of[subject],
                                  channel=ch, std_uv=s))
    rec_df, subj_df = pd.DataFrame(rec_rows), pd.DataFrame(subj_rows)
    save_table(rec_df, tab / "A_std_per_recording_channel.csv")
    save_table(subj_df, tab / "A_std_per_subject_channel.csv")

    wide = subj_df.pivot(index="subject", columns="channel", values="std_uv")[channels]
    rec_wide = rec_df.pivot_table(index=["subject", "condition"], columns="channel",
                                  values="std_uv")[channels]
    ratio = rec_wide.xs("high", level="condition") / rec_wide.xs("low", level="condition")

    spread_rows, within_rows = [], []
    for group in GROUPS:
        subs = subjects_of(group, split_of)
        sub = wide.loc[subs]
        rat = ratio.loc[subs]
        for ch in channels:
            v = sub[ch].to_numpy()
            r = rat[ch].to_numpy()
            between_sd = float(np.std(np.log2(v), ddof=1))
            median_abs_within = float(np.median(np.abs(np.log2(r))))
            spread_rows.append(dict(
                group=group, channel=ch, n_subjects=len(v),
                median_std_uv=np.median(v), q25=np.percentile(v, 25),
                q75=np.percentile(v, 75), min=v.min(), max=v.max(),
                max_over_min=v.max() / v.min(),
                p90_over_p10=np.percentile(v, 90) / np.percentile(v, 10),
                between_subject_sd_log2=between_sd))
            within_rows.append(dict(
                group=group, channel=ch, n_subjects=len(r),
                median_ratio_high_over_low=np.median(r),
                frac_subjects_high_gt_low=float(np.mean(r > 1)),
                median_abs_log2_ratio=median_abs_within,
                between_subject_sd_log2=between_sd,
                between_over_within=(between_sd / median_abs_within
                                     if median_abs_within > 0 else np.nan)))
    spread_df, within_df = pd.DataFrame(spread_rows), pd.DataFrame(within_rows)
    save_table(spread_df, tab / "A_between_subject_spread_summary.csv")
    save_table(within_df, tab / "A_within_vs_between_summary.csv")

    # ---- Figure A1: heatmap (subjects x channels), rows grouped by split
    order = sorted(wide.index, key=lambda s: (SPLIT_RANK[split_of[s]], s))
    mat = wide.loc[order].to_numpy()
    f, ax = plt.subplots(figsize=(12, 13))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", norm=LogNorm())
    ax.set_xticks(range(len(channels)), channels)
    ax.set_yticks(range(len(order)), [f"S{s:02d} ({split_of[s]})" for s in order],
                  fontsize=7)
    for tick, s in zip(ax.get_yticklabels(), order):
        tick.set_color(SPLIT_COLOR[split_of[s]])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    fontsize=5.5, color="white")
    ax.set_title("Filtered-signal std per subject and channel (log colour scale)\n"
                 "test rows are descriptive only")
    f.colorbar(im, ax=ax, label="std (data units, assumed uV)")
    save_fig(f, fig / "A1_std_heatmap_subjects_x_channels.png")

    # ---- Figure A2: between-subject spread per channel
    f, ax = plt.subplots(figsize=(12, 5))
    rng = np.random.RandomState(0)  # jitter only; cosmetic
    for ci, ch in enumerate(channels):
        for s in wide.index:
            ax.scatter(ci + rng.uniform(-0.25, 0.25), wide.loc[s, ch], s=14,
                       color=SPLIT_COLOR["test"] if split_of[s] == "test" else "0.35",
                       alpha=0.75, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(channels)), channels)
    ax.set_ylabel("subject std (log scale)")
    ax.set_title("Between-subject amplitude spread per channel "
                 "(grey = development, red = test, descriptive)")
    ax.grid(alpha=0.3, which="both")
    save_fig(f, fig / "A2_between_subject_std_spread.png")

    # ---- Figure A3: within-subject class effect (high/low std ratio)
    f, ax = plt.subplots(figsize=(12, 5))
    dev_subs = subjects_of("development", split_of)
    ax.boxplot([np.log2(ratio.loc[dev_subs, ch].to_numpy()) for ch in channels],
               tick_labels=channels, showfliers=True)
    for ci, ch in enumerate(channels):
        for s in subjects_of("test", split_of):
            ax.scatter(ci + 1, np.log2(ratio.loc[s, ch]), s=18,
                       color=SPLIT_COLOR["test"], zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("log2( std_high / std_low ) per subject")
    ax.set_title("Within-subject class effect on amplitude "
                 "(box = development, red dots = test, descriptive)")
    ax.grid(alpha=0.3)
    save_fig(f, fig / "A3_within_subject_high_vs_low_std_ratio.png")
    return spread_df, within_df


# ----------------------------------------------------------------------
# PART B - exploratory artifact statistics (NO rejection)
# ----------------------------------------------------------------------
def compute_ptp(X):
    out = np.empty((len(X), X.shape[2]), dtype=np.float32)
    for s in range(0, len(X), CHUNK):
        blk = np.asarray(X[s:s + CHUNK])
        out[s:s + CHUNK] = blk.max(axis=1) - blk.min(axis=1)
    return out


def ecdf(values):
    v = np.sort(values)
    return v, np.arange(1, len(v) + 1) / len(v)


def part_b_artifacts(X, meta, channels, split_of, tab, fig):
    print("\n[B] Exploratory artifact statistics (no window is removed)")
    ptp = compute_ptp(X)
    ptp_df = meta[["subject", "split", "condition", "window"]].copy()
    for ci, ch in enumerate(channels):
        ptp_df[ch] = ptp[:, ci]
    save_table(ptp_df, tab / "B_window_ptp_per_channel.csv")

    masks = window_group_masks(meta)
    cond = meta["condition"].to_numpy()

    # distribution of peak-to-peak amplitude
    pct_rows = []
    for group, gmask in masks.items():
        for cls in ("low", "high"):
            sel = gmask & (cond == cls)
            for ci, ch in enumerate(channels):
                v = ptp[sel, ci]
                pct_rows.append(dict(group=group, condition=cls, channel=ch,
                                     n_windows=int(sel.sum()),
                                     **{f"p{p}": float(np.percentile(v, p))
                                        for p in (50, 75, 90, 95, 99)},
                                     max=float(v.max())))
    save_table(pd.DataFrame(pct_rows), tab / "B_ptp_percentiles.csv")

    # exceedance rates at EXPLORATORY thresholds
    half = len(channels) / 2
    overall, per_subject, per_channel = [], [], []
    for thr in EXPLORATORY_PTP_THRESHOLDS:
        exceed = ptp > thr
        any_ch = exceed.any(axis=1)
        half_ch = exceed.sum(axis=1) >= half
        for group, gmask in masks.items():
            for cls in ("low", "high", "both"):
                sel = gmask if cls == "both" else gmask & (cond == cls)
                overall.append(dict(
                    group=group, condition=cls, exploratory_threshold=thr,
                    n_windows=int(sel.sum()),
                    pct_windows_any_channel=100 * any_ch[sel].mean(),
                    pct_windows_ge_half_channels=100 * half_ch[sel].mean(),
                    mean_pct_per_channel=100 * exceed[sel].mean()))
                if cls != "both":
                    for ci, ch in enumerate(channels):
                        per_channel.append(dict(
                            group=group, condition=cls, channel=ch,
                            exploratory_threshold=thr,
                            pct_windows_exceeding=100 * exceed[sel, ci].mean()))
        for (subject, c), idx in meta.groupby(["subject", "condition"]).indices.items():
            per_subject.append(dict(
                subject=subject, split=split_of[subject], condition=c,
                exploratory_threshold=thr, n_windows=len(idx),
                pct_windows_any_channel=100 * any_ch[idx].mean()))
    overall_df = pd.DataFrame(overall)
    per_subject_df = pd.DataFrame(per_subject)
    per_channel_df = pd.DataFrame(per_channel)
    save_table(overall_df, tab / "B_artifact_rates_overall_EXPLORATORY.csv")
    save_table(per_subject_df, tab / "B_artifact_rates_per_subject_EXPLORATORY.csv")
    save_table(per_channel_df, tab / "B_artifact_rates_per_channel_EXPLORATORY.csv")

    # paired comparison high vs low across development subjects (exploratory)
    dev_subs = subjects_of("development", split_of)
    paired = []
    for thr in EXPLORATORY_PTP_THRESHOLDS:
        t = per_subject_df[(per_subject_df.exploratory_threshold == thr)
                           & per_subject_df.subject.isin(dev_subs)]
        w = t.pivot(index="subject", columns="condition",
                    values="pct_windows_any_channel")
        diff = (w["high"] - w["low"]).to_numpy()
        try:
            p = stats.wilcoxon(w["high"], w["low"]).pvalue if np.any(diff != 0) else np.nan
        except ValueError:
            p = np.nan
        paired.append(dict(group="development", exploratory_threshold=thr,
                           n_subjects=len(w), median_pct_low=w["low"].median(),
                           median_pct_high=w["high"].median(),
                           median_diff_high_minus_low=float(np.median(diff)),
                           n_subjects_high_gt_low=int((diff > 0).sum()),
                           n_subjects_high_lt_low=int((diff < 0).sum()),
                           wilcoxon_p_EXPLORATORY=p))
    save_table(pd.DataFrame(paired), tab / "B_paired_high_vs_low_development_EXPLORATORY.csv")

    # ---- Figure B1: ECDF of window peak-to-peak amplitude
    f, axes = plt.subplots(1, 2, figsize=(13, 5))
    summaries = {"max over channels": ptp.max(axis=1),
                 "median over channels": np.median(ptp, axis=1)}
    for ax, (name, vals) in zip(axes, summaries.items()):
        for cls in ("low", "high"):
            for group, ls, lw in (("development", "-", 2.0), ("test", "--", 1.2)):
                sel = masks[group] & (cond == cls)
                x, y = ecdf(vals[sel])
                ax.plot(x, y, ls=ls, lw=lw, color=CLASS_COLOR[cls],
                        label=f"{cls} ({group}{', descriptive' if group == 'test' else ''})")
        if name.startswith("max"):
            for thr in EXPLORATORY_PTP_THRESHOLDS:
                ax.axvline(thr, color="0.5", ls=":", lw=1)
                ax.text(thr, 0.02, f" {thr}\n exploratory", fontsize=7, color="0.3")
        ax.set_xscale("log")
        ax.set_xlabel(f"window peak-to-peak, {name}")
        ax.set_ylabel("empirical CDF")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig_title = "Window peak-to-peak amplitude distribution (no thresholds applied)"
    f.suptitle(fig_title)
    save_fig(f, fig / "B1_ptp_ecdf_low_vs_high.png")

    # ---- Figure B2: per-subject exceedance at each exploratory threshold
    order = sorted(split_of.index, key=lambda s: (SPLIT_RANK[split_of[s]], s))
    f, axes = plt.subplots(len(EXPLORATORY_PTP_THRESHOLDS), 1, figsize=(14, 9), sharex=True)
    xpos = np.arange(len(order))
    for ax, thr in zip(np.atleast_1d(axes), EXPLORATORY_PTP_THRESHOLDS):
        t = per_subject_df[per_subject_df.exploratory_threshold == thr]
        for k, cls in enumerate(("low", "high")):
            vals = [t[(t.subject == s) & (t.condition == cls)]
                    ["pct_windows_any_channel"].iloc[0] for s in order]
            ax.bar(xpos + (k - 0.5) * 0.4, vals, width=0.4,
                   color=CLASS_COLOR[cls], label=cls)
        ax.set_ylabel(f"% windows\n> {thr} (expl.)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xticks(xpos, [str(s) for s in order], fontsize=7)
    color_ticklabels(axes[-1], order, split_of)
    axes[-1].set_xlabel("subject (blue = train, green = validation, red = test/descriptive)")
    f.suptitle("Windows with >= 1 channel above an EXPLORATORY peak-to-peak threshold")
    save_fig(f, fig / "B2_per_subject_exceedance_exploratory.png")

    # ---- Figure B3: per-channel exceedance (development)
    f, axes = plt.subplots(1, len(EXPLORATORY_PTP_THRESHOLDS), figsize=(15, 4.5), sharey=False)
    for ax, thr in zip(np.atleast_1d(axes), EXPLORATORY_PTP_THRESHOLDS):
        t = per_channel_df[(per_channel_df.group == "development")
                           & (per_channel_df.exploratory_threshold == thr)]
        for k, cls in enumerate(("low", "high")):
            vals = t[t.condition == cls].set_index("channel").loc[channels,
                                                                  "pct_windows_exceeding"]
            ax.bar(np.arange(len(channels)) + (k - 0.5) * 0.4, vals, width=0.4,
                   color=CLASS_COLOR[cls], label=cls)
        ax.set_xticks(range(len(channels)), channels, rotation=60, fontsize=8)
        ax.set_title(f"> {thr} (exploratory)")
        ax.set_ylabel("% windows")
        ax.grid(alpha=0.3, axis="y")
    np.atleast_1d(axes)[0].legend()
    f.suptitle("Per-channel exceedance, development subjects")
    save_fig(f, fig / "B3_per_channel_exceedance_development.png")
    return overall_df, pd.DataFrame(paired)


# ----------------------------------------------------------------------
# PART C - PSD and relative band power
# ----------------------------------------------------------------------
def part_c_spectral(X, meta, channels, split_of, tab, fig):
    print("\n[C] Spectral view (Welch PSD, relative band power)")
    psd_list, rec_keys, band_rows = [], [], []
    freqs = None
    for (subject, condition), idx in meta.groupby(["subject", "condition"]).indices.items():
        block = np.asarray(X[np.sort(idx)], dtype=np.float64)       # (w, 512, ch)
        freqs, pxx = signal.welch(block, fs=FS, window="hann", nperseg=WELCH_NPERSEG,
                                  noverlap=WELCH_NOVERLAP, detrend="constant",
                                  axis=1, scaling="density")        # (w, f, ch)
        in_range = (freqs >= 1) & (freqs <= 40)
        total = pxx[:, in_range, :].sum(axis=1)                     # (w, ch)
        for name, (lo, hi) in BANDS.items():
            sel = (freqs >= lo) & ((freqs < hi) | ((hi == 40) & (freqs <= 40)))
            rel = pxx[:, sel, :].sum(axis=1) / total                # (w, ch)
            for ci, ch in enumerate(channels):
                band_rows.append(dict(subject=subject, split=split_of[subject],
                                      condition=condition, channel=ch, band=name,
                                      rel_power=float(rel[:, ci].mean())))
        psd_list.append(pxx.mean(axis=0))                           # (f, ch)
        rec_keys.append((subject, condition))
    psd = np.stack(psd_list)                                        # (rec, f, ch)
    band_df = pd.DataFrame(band_rows)
    save_table(band_df, tab / "C_relative_band_power_per_recording_channel.csv")
    np.savez_compressed(tab / "C_psd_per_recording.npz", freqs=freqs, psd=psd,
                        subject=np.array([k[0] for k in rec_keys]),
                        condition=np.array([k[1] for k in rec_keys]),
                        channels=np.array(channels))
    print("   table : C_psd_per_recording.npz")

    subj_arr = np.array([k[0] for k in rec_keys])
    cond_arr = np.array([k[1] for k in rec_keys])
    split_arr = np.array([split_of[s] for s in subj_arr])

    # subject-level band table (average channels first), per class and paired change
    sub_band = (band_df.groupby(["subject", "split", "condition", "band"])["rel_power"]
                .mean().reset_index())
    summ_rows, delta_rows = [], []
    for group in GROUPS:
        subs = subjects_of(group, split_of)
        sb = sub_band[sub_band.subject.isin(subs)]
        for band in BANDS:
            w = (sb[sb.band == band].pivot(index="subject", columns="condition",
                                           values="rel_power"))
            for cls in ("low", "high"):
                summ_rows.append(dict(group=group, band=band, condition=cls,
                                      n_subjects=len(w), mean_rel_power=w[cls].mean(),
                                      sd_across_subjects=w[cls].std(ddof=1)))
            d = (w["high"] - w["low"]) * 100  # percentage points
            delta_rows.append(dict(group=group, band=band, n_subjects=len(d),
                                   mean_change_pp_high_minus_low=d.mean(),
                                   sd_pp=d.std(ddof=1),
                                   n_subjects_increase=int((d > 0).sum()),
                                   n_subjects_decrease=int((d < 0).sum())))
    save_table(pd.DataFrame(summ_rows), tab / "C_band_power_summary_by_class.csv")
    delta_df = pd.DataFrame(delta_rows)
    save_table(delta_df, tab / "C_band_power_high_minus_low_summary.csv")

    # ---- Figure C1: grand-average PSD (mean over channels), dev mean +/- SEM
    def subject_psd_db(group, cls):
        subs = set(subjects_of(group, split_of))
        sel = np.array([(s in subs) and (c == cls) for s, c in zip(subj_arr, cond_arr)])
        return 10 * np.log10(psd[sel].mean(axis=2) + 1e-20)         # (n_sub, f)

    f, ax = plt.subplots(figsize=(10, 5.5))
    for cls in ("low", "high"):
        dev = subject_psd_db("development", cls)
        m, sem = dev.mean(0), dev.std(0, ddof=1) / np.sqrt(len(dev))
        ax.plot(freqs, m, color=CLASS_COLOR[cls], lw=2, label=f"{cls} (development, n={len(dev)})")
        ax.fill_between(freqs, m - sem, m + sem, color=CLASS_COLOR[cls], alpha=0.25)
        tst = subject_psd_db("test", cls)
        ax.plot(freqs, tst.mean(0), color=CLASS_COLOR[cls], ls="--", lw=1.2,
                label=f"{cls} (test, descriptive, n={len(tst)})")
    for name, (lo, hi) in BANDS.items():
        ax.axvline(lo, color="0.7", lw=0.6)
        ax.text(lo + 0.3, ax.get_ylim()[1], name, fontsize=7, va="top", color="0.35")
    ax.axvline(40, color="0.7", lw=0.6)
    ax.set_xlim(0, FS / 2)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD, mean over channels (dB, data units^2/Hz)")
    ax.set_title("Grand-average PSD by class (shaded = +/- SEM across development subjects)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(f, fig / "C1_grand_average_psd_low_vs_high.png")

    # ---- Figure C2: per-subject PSD curves (inter-subject variability)
    f, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, cls in zip(axes, ("low", "high")):
        for s, c, sp, curve in zip(subj_arr, cond_arr, split_arr,
                                   10 * np.log10(psd.mean(axis=2) + 1e-20)):
            if c != cls:
                continue
            is_test = sp == "test"
            ax.plot(freqs, curve, lw=0.9 if not is_test else 1.1,
                    color=SPLIT_COLOR["test"] if is_test else "0.45",
                    alpha=0.6, ls="--" if is_test else "-")
        ax.set_title(f"{cls}: one line per subject (grey = development, red = test)")
        ax.set_xlabel("frequency (Hz)")
        ax.set_xlim(0, FS / 2)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PSD, mean over channels (dB)")
    save_fig(f, fig / "C2_per_subject_psd.png")

    # ---- Figure C3: paired relative band power low -> high (development)
    dev_subs = subjects_of("development", split_of)
    f, axes = plt.subplots(1, len(BANDS), figsize=(16, 4.5))
    for ax, (band, (lo, hi)) in zip(axes, BANDS.items()):
        w = (sub_band[(sub_band.band == band) & sub_band.subject.isin(dev_subs)]
             .pivot(index="subject", columns="condition", values="rel_power"))
        for _, row in w.iterrows():
            ax.plot([0, 1], [row["low"], row["high"]], color="0.6", lw=0.7, alpha=0.7)
        ax.plot([0, 1], [w["low"].mean(), w["high"].mean()], color="k", lw=2.5, marker="o")
        ax.set_xticks([0, 1], ["low", "high"])
        ax.set_title(f"{band} ({lo}-{hi} Hz)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("relative power (subject mean over channels)")
    f.suptitle("Relative band power, one grey line per development subject, black = mean")
    save_fig(f, fig / "C3_relative_band_power_paired_development.png")

    # ---- Figure C4: mean change (high - low) in rel. power, band x channel
    dev_bc = band_df[band_df.subject.isin(dev_subs)]
    piv = (dev_bc.pivot_table(index=["band", "channel"], columns="condition",
                              values="rel_power", aggfunc="mean"))
    change = ((piv["high"] - piv["low"]) * 100).unstack("channel").loc[list(BANDS)][channels]
    f, ax = plt.subplots(figsize=(12, 4))
    lim = np.abs(change.to_numpy()).max()
    im = ax.imshow(change.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(channels)), channels)
    ax.set_yticks(range(len(BANDS)), list(BANDS))
    for i in range(change.shape[0]):
        for j in range(change.shape[1]):
            ax.text(j, i, f"{change.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
    f.colorbar(im, ax=ax, label="mean change, percentage points (high - low)")
    ax.set_title("Mean relative-power change high - low, development subjects")
    save_fig(f, fig / "C4_band_power_change_band_x_channel_development.png")
    return pd.DataFrame(summ_rows), delta_df


# ----------------------------------------------------------------------
# Summary text (numbers only - NO decisions)
# ----------------------------------------------------------------------
def write_summary(path, meta, within_df, overall_df, paired_df, band_summary, delta_df):
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    lines = []
    add = lines.append
    add("STEP 3 AUDIT SUMMARY  -  DESCRIPTIVE ONLY")
    add("No window was removed, nothing was normalized, no artifact threshold was")
    add("chosen and no pipeline decision was made by this script.")
    add("Groups: development = train+validation (decision-eligible in Step 4);")
    add("        test = descriptive only; all = descriptive.")
    add("")
    add(f"Windows: {len(meta)} | subjects: {meta['subject'].nunique()} | "
        f"recordings: {meta.groupby(['subject', 'condition']).ngroups}")
    add("")
    add("[A] Between-subject vs within-subject amplitude variation (per channel)")
    add("    between_subject_sd_log2 : spread of subject std across people")
    add("    median_abs_log2_ratio   : typical |high vs low| std change inside one person")
    add("    between_over_within     : >1 means people differ more than classes do")
    cols = ["channel", "between_subject_sd_log2", "median_abs_log2_ratio",
            "between_over_within", "median_ratio_high_over_low",
            "frac_subjects_high_gt_low"]
    for g in ("development", "test"):
        add(f"  group = {g}")
        add(within_df[within_df.group == g][cols].to_string(index=False))
    add("")
    add("[B] Windows with >=1 channel above an EXPLORATORY peak-to-peak threshold (%)")
    ob = overall_df[overall_df.condition.isin(["low", "high"])]
    for g in GROUPS:
        t = ob[ob.group == g].pivot(index="exploratory_threshold", columns="condition",
                                    values="pct_windows_any_channel")[["low", "high"]]
        add(f"  group = {g}")
        add(t.to_string())
    add("  Paired high-vs-low across development subjects (exploratory):")
    add(paired_df.to_string(index=False))
    add("")
    add("[C] Relative band power, mean over subjects (development)")
    bs = band_summary[band_summary.group == "development"]
    add(bs.pivot(index="band", columns="condition", values="mean_rel_power")[["low", "high"]]
        .loc[list(BANDS)].to_string())
    add("  Mean change high - low (percentage points), development:")
    add(delta_df[delta_df.group == "development"].set_index("band")
        [["mean_change_pp_high_minus_low", "sd_pp", "n_subjects_increase",
          "n_subjects_decrease"]].loc[list(BANDS)].to_string())
    add("")
    add("DECISIONS MADE BY THIS STEP: none.")
    add("Normalization, artifact handling and feature choices are decided in Step 4,")
    add("using development-subject results only.")
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    print("\n" + text)


# ----------------------------------------------------------------------
def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Step 3 descriptive audit.")
    parser.add_argument("--data-dir", type=Path,
                        default=repo_root / "processed_data" / "contract_v1")
    parser.add_argument("--out-dir", type=Path,
                        default=repo_root / "results" / "step3_audit")
    args = parser.parse_args()

    tab, fig = args.out_dir / "tables", args.out_dir / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    inputs = {n: args.data_dir / n for n in ("X_filtered.npy", "metadata.csv", "splits.json")}
    hashes_before = {n: file_sha256(p) for n, p in inputs.items()}

    X, meta, splits, channels, split_of, config = load_inputs(args.data_dir)
    print(f"X {X.shape} | subjects {meta['subject'].nunique()} | "
          f"groups: " + ", ".join(f"{k}={int((split_of == k).sum())}"
                                  for k in ('train', 'validation', 'test')))

    _, within_df = part_a_amplitude(X, meta, channels, split_of, tab, fig)
    overall_df, paired_df = part_b_artifacts(X, meta, channels, split_of, tab, fig)
    band_summary, delta_df = part_c_spectral(X, meta, channels, split_of, tab, fig)
    write_summary(args.out_dir / "audit_summary.txt", meta, within_df, overall_df,
                  paired_df, band_summary, delta_df)

    hashes_after = {n: file_sha256(p) for n, p in inputs.items()}
    assert hashes_before == hashes_after, "An input file changed during the audit!"
    run_info = {
        "step": "3 - descriptive audit (no decisions)",
        "input_sha256_before_and_after_identical": True,
        "input_sha256": hashes_after,
        "exploratory_ptp_thresholds": list(EXPLORATORY_PTP_THRESHOLDS),
        "assumed_units": "uV (not verified)",
        "bands_hz": BANDS,
        "welch": {"fs": FS, "nperseg": WELCH_NPERSEG, "noverlap": WELCH_NOVERLAP,
                  "window": "hann", "detrend": "constant"},
        "windows_removed": 0,
        "normalization_applied": False,
        "groups": {"development": "train+validation", "test": "descriptive only"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pandas": pd.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    (args.out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print("\n[OK] input files unchanged (SHA-256 identical before and after).")
    print(f"[OK] results saved under: {args.out_dir}")


if __name__ == "__main__":
    main()