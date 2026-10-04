"""
audit_robust.py  -  STEP 3b: gross-artifact concentration + robust amplitude audit
==================================================================================
READ-ONLY and DESCRIPTIVE (builds on audit_dataset.py, which it imports).
  * no window and no subject is removed, nothing is normalized, no model is trained
  * X_filtered.npy / metadata.csv / splits.json are hash-checked before and after
  * thresholds are the Step 3 EXPLORATORY ones (100 / 150 / 200), not validated limits
  * groups: development (train+validation) = decision-eligible; test = descriptive only

Questions answered (see audit3b_summary.txt):
  A. Are extreme amplitudes concentrated in a few subjects or widespread?
  B. Can ordinary mean/std scaling be strongly affected by extreme values?
  C. Is robust scaling worth carrying into Step 4 as a CANDIDATE?
  D. Is there a defensible reason to remove anything?  (nothing is removed here)

Run from the repo root:
    python preprocessing/audit_robust.py
"""

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import audit_dataset as ad   # reuse loader, hashing, plotting helpers, thresholds

SEED = 42
SUBSAMPLE_PER_SUBJECT = 10_000      # samples/subject for pooled median & MAD (+ LOSO)
MAD_TO_SIGMA = 1.4826               # MAD * 1.4826 = sigma for Gaussian data
IQR_TO_SIGMA = 1.349
TOP_SUBJECT_FRACTION = 0.10         # "top 10 % of units" for concentration shares
RANK_PERCENTS = (1, 5)              # rank-based extremes (NOT amplitude thresholds)

# Heuristic reading rules for the printed A-D answers. They only label the numbers;
# they are NOT decisions and are printed so they can be challenged.
CONC_STRONG, CONC_MODERATE = 2.0, 1.5      # top-10% share / fair share
INFLATION_STRONG, INFLATION_MILD = 1.5, 1.2  # pooled std / robust sigma (median over ch.)
REVIEW_RATIO = 3.0                          # subject-channel std/robust ratio worth a look


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def amplitude_stats(flat):
    """flat: (n_samples, n_channels) -> mean, std, median, MAD-sigma, IQR-sigma."""
    med = np.median(flat, axis=0)
    mad_sigma = MAD_TO_SIGMA * np.median(np.abs(flat - med), axis=0)
    q75, q25 = np.percentile(flat, [75, 25], axis=0)
    return flat.mean(axis=0), flat.std(axis=0), med, mad_sigma, (q75 - q25) / IQR_TO_SIGMA


def concentration_stats(counts, top_frac=TOP_SUBJECT_FRACTION):
    """How unevenly are 'extreme' counts spread over equally-sized units?"""
    desc = np.sort(np.asarray(counts, dtype=float))[::-1]
    n, total = len(desc), desc.sum()
    k = max(1, int(np.ceil(top_frac * n)))
    out = dict(n_units=n, top_k=k, total_extreme=int(total), fair_share=k / n)
    if total == 0:
        return {**out, "top_k_share": np.nan, "conc_ratio": np.nan,
                "gini": np.nan, "units_for_50pct": np.nan}
    asc = desc[::-1]
    gini = 2 * np.sum(np.arange(1, n + 1) * asc) / (n * total) - (n + 1) / n
    top_share = desc[:k].sum() / total
    return {**out, "top_k_share": top_share, "conc_ratio": top_share / (k / n),
            "gini": gini, "units_for_50pct": int(np.searchsorted(np.cumsum(desc) / total, 0.5) + 1)}


def lorenz(counts):
    asc = np.sort(np.asarray(counts, dtype=float))
    total = asc.sum()
    x = np.arange(len(asc) + 1) / len(asc)
    y = np.concatenate([[0.0], np.cumsum(asc) / total]) if total > 0 else x
    return x, y


# ----------------------------------------------------------------------
# PART 1 - gross artifact concentration
# ----------------------------------------------------------------------
def part1_concentration(X, meta, channels, split_of, tab, fig):
    print("\n[1] Gross-artifact concentration (exploratory thresholds; nothing removed)")
    ptp = ad.compute_ptp(X)                              # (N, ch)
    max_ptp = ptp.max(axis=1)
    cond = meta["condition"].to_numpy()
    subj = meta["subject"].to_numpy()
    dev_subjects = list(ad.subjects_of("development", split_of))
    test_subjects = list(ad.subjects_of("test", split_of))
    groups = {"development": dev_subjects, "test": test_subjects}

    subject_rows, summary_rows, cell_counts = [], [], {}
    for thr in ad.EXPLORATORY_PTP_THRESHOLDS:
        exceed = ptp > thr
        any_ch = exceed.any(axis=1)
        for group, subs in groups.items():
            for cls in ("low", "high", "both"):
                sel_cls = np.ones(len(meta), bool) if cls == "both" else cond == cls
                # window level, one unit = one subject (per class = one recording)
                counts = [int(any_ch[(subj == s) & sel_cls].sum()) for s in subs]
                n_win = [int(((subj == s) & sel_cls).sum()) for s in subs]
                if cls != "both":
                    for s, c, n in zip(subs, counts, n_win):
                        subject_rows.append(dict(
                            subject=s, split=split_of[s], condition=cls,
                            exploratory_threshold=thr, n_windows=n, n_extreme=c,
                            pct_extreme=100 * c / n))
                summary_rows.append(dict(level="window (subject/recording)", group=group,
                                         condition=cls, exploratory_threshold=thr,
                                         **concentration_stats(counts)))
                # cell level, one unit = one (subject, channel) pair
                cells = np.array([[int(exceed[(subj == s) & sel_cls, ci].sum())
                                   for ci in range(len(channels))] for s in subs])
                cell_counts[(group, cls, thr)] = cells
                summary_rows.append(dict(level="cell (subject x channel)", group=group,
                                         condition=cls, exploratory_threshold=thr,
                                         **concentration_stats(cells.ravel())))
    subj_df = pd.DataFrame(subject_rows)
    summ_df = pd.DataFrame(summary_rows)
    ad.save_table(subj_df, tab / "1_extreme_windows_per_recording_EXPLORATORY.csv")
    ad.save_table(summ_df, tab / "1_concentration_summary_EXPLORATORY.csv")

    # rank-based extremes in the development group (no amplitude threshold)
    dev_mask = meta["split"].isin(ad.DEV_SPLITS).to_numpy()
    rank_rows, rank_counts = [], {}
    for q in RANK_PERCENTS:
        cutoff = np.percentile(max_ptp[dev_mask], 100 - q)
        sel = dev_mask & (max_ptp >= cutoff)
        counts = [int((sel & (subj == s)).sum()) for s in dev_subjects]
        rank_counts[q] = np.array(counts)
        rank_rows.append(dict(top_percent_of_dev_windows=q, value_at_rank_cutoff=float(cutoff),
                              n_windows=int(sel.sum()),
                              pct_from_high_class=100 * float((cond[sel] == "high").mean()),
                              n_subjects_contributing=int(np.sum(np.array(counts) > 0)),
                              **concentration_stats(counts)))
    rank_df = pd.DataFrame(rank_rows)
    ad.save_table(rank_df, tab / "1_rank_based_extremes_development.csv")
    top_by_subject = pd.DataFrame({"subject": dev_subjects,
                                   **{f"top{q}pct_windows": rank_counts[q] for q in RANK_PERCENTS}})
    ad.save_table(top_by_subject, tab / "1_rank_based_extremes_per_subject_development.csv")

    # ---------------- figure 1
    f, axes = plt.subplots(2, 3, figsize=(16, 9))
    thr_colors = dict(zip(ad.EXPLORATORY_PTP_THRESHOLDS, ("#1b9e77", "#d95f02", "#7570b3")))
    for ci, cls in enumerate(("low", "high")):
        for row, level in enumerate(("window", "cell")):
            ax = axes[row, ci]
            for thr in ad.EXPLORATORY_PTP_THRESHOLDS:
                cells = cell_counts[("development", cls, thr)]
                if level == "window":
                    # reuse window-level counts from the summary table
                    c = subj_df[(subj_df.condition == cls) & (subj_df.exploratory_threshold == thr)
                                & subj_df.subject.isin(dev_subjects)].set_index("subject") \
                        .loc[dev_subjects, "n_extreme"].to_numpy()
                else:
                    c = cells.ravel()
                x, y = lorenz(c)
                g = concentration_stats(c)["gini"]
                ax.plot(x, y, color=thr_colors[thr], lw=2, label=f">{thr} (Gini {g:.2f})")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="equal share")
            ax.set_title(f"{cls}: {'subjects (recordings)' if level == 'window' else 'subject x channel cells'}")
            ax.set_xlabel("cumulative share of units (least -> most extreme)")
            ax.set_ylabel("cumulative share of extreme windows")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    ax = axes[0, 2]
    for q, col in zip(RANK_PERCENTS, ("#1f77b4", "#e45756")):
        x, y = lorenz(rank_counts[q])
        ax.plot(x, y, lw=2, color=col, label=f"top {q}% windows by max p-p "
                f"(Gini {concentration_stats(rank_counts[q])['gini']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("dev.: rank-based extremes (no threshold)", fontsize=10)
    ax.set_xlabel("cumulative share of subjects")
    ax.set_ylabel("cumulative share of extreme windows")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    order = np.argsort(-rank_counts[5])
    ax.bar(range(len(dev_subjects)), rank_counts[5][order], color="#e45756")
    ax.axhline(rank_counts[5].sum() / len(dev_subjects), color="k", ls="--", lw=1,
               label="equal share")
    ax.set_xticks(range(len(dev_subjects)), [str(dev_subjects[i]) for i in order],
                  rotation=90, fontsize=7)
    ax.set_title("dev.: top-5% windows per subject", fontsize=10)
    ax.set_xlabel("subject")
    ax.legend(fontsize=8)
    f.suptitle("Gross-artifact concentration (exploratory thresholds / rank-based; "
               "nothing removed)")
    ad.save_fig(f, fig / "1_artifact_concentration.png")
    return summ_df, rank_df, subj_df, ptp


# ----------------------------------------------------------------------
# PART 2 - robust amplitude statistics
# ----------------------------------------------------------------------
def part2_robust(X, meta, channels, split_of, tab, fig):
    print("\n[2] Robust amplitude statistics (median / MAD / IQR vs mean / std)")
    rec_rows, subj_rows = [], []
    moments, samples = {}, {}
    for (subject, cond), idx in meta.groupby(["subject", "condition"]).indices.items():
        flat = np.asarray(X[np.sort(idx)], dtype=np.float64).reshape(-1, X.shape[2])
        mean, std, med, mad, iqr = amplitude_stats(flat)
        for ci, ch in enumerate(channels):
            rec_rows.append(dict(subject=subject, split=split_of[subject], condition=cond,
                                 channel=ch, mean=mean[ci], std=std[ci], median=med[ci],
                                 mad_sigma=mad[ci], iqr_sigma=iqr[ci],
                                 std_over_mad_sigma=std[ci] / mad[ci]))
    rng_root = SEED
    for subject, idx in meta.groupby("subject").indices.items():
        flat = np.asarray(X[np.sort(idx)], dtype=np.float64).reshape(-1, X.shape[2])
        mean, std, med, mad, iqr = amplitude_stats(flat)
        moments[subject] = (len(flat), flat.sum(axis=0), (flat ** 2).sum(axis=0))
        if split_of[subject] in ad.DEV_SPLITS:
            rs = np.random.RandomState(rng_root + int(subject))
            samples[subject] = flat[rs.choice(len(flat), SUBSAMPLE_PER_SUBJECT, replace=False)]
        for ci, ch in enumerate(channels):
            subj_rows.append(dict(subject=subject, split=split_of[subject], channel=ch,
                                  mean=mean[ci], std=std[ci], median=med[ci],
                                  mad_sigma=mad[ci], iqr_sigma=iqr[ci],
                                  std_over_mad_sigma=std[ci] / mad[ci]))
    rec_df, subj_df = pd.DataFrame(rec_rows), pd.DataFrame(subj_rows)
    if (subj_df.mad_sigma <= 0).any() or (rec_df.mad_sigma <= 0).any():
        raise ValueError("A recording/subject has MAD = 0 on some channel (flat channel?)")
    ad.save_table(rec_df, tab / "2_robust_stats_per_recording_channel.csv")
    ad.save_table(subj_df, tab / "2_robust_stats_per_subject_channel.csv")

    # ---- spread of ordinary vs robust amplitude across people, and class effect
    std_w = subj_df.pivot(index="subject", columns="channel", values="std")[channels]
    mad_w = subj_df.pivot(index="subject", columns="channel", values="mad_sigma")[channels]
    rec_std = rec_df.pivot_table(index=["subject", "condition"], columns="channel",
                                 values="std")[channels]
    rec_mad = rec_df.pivot_table(index=["subject", "condition"], columns="channel",
                                 values="mad_sigma")[channels]
    ratio_std = rec_std.xs("high", level="condition") / rec_std.xs("low", level="condition")
    ratio_mad = rec_mad.xs("high", level="condition") / rec_mad.xs("low", level="condition")
    spread_rows = []
    for group in ("development", "test"):
        subs = ad.subjects_of(group, split_of)
        for ch in channels:
            s, m = std_w.loc[subs, ch].to_numpy(), mad_w.loc[subs, ch].to_numpy()
            rr = (s / m)
            b_std, b_mad = np.std(np.log2(s), ddof=1), np.std(np.log2(m), ddof=1)
            w_std = float(np.median(np.abs(np.log2(ratio_std.loc[subs, ch]))))
            w_mad = float(np.median(np.abs(np.log2(ratio_mad.loc[subs, ch]))))
            spread_rows.append(dict(
                group=group, channel=ch, n_subjects=len(s),
                between_sd_log2_std=b_std, between_sd_log2_robust=b_mad,
                spread_ratio_std_over_robust=b_std / b_mad,
                median_std_over_robust=float(np.median(rr)),
                p90_std_over_robust=float(np.percentile(rr, 90)),
                max_std_over_robust=float(rr.max()),
                within_median_abs_log2_ratio_std=w_std,
                within_median_abs_log2_ratio_robust=w_mad,
                median_high_over_low_std=float(np.median(ratio_std.loc[subs, ch])),
                median_high_over_low_robust=float(np.median(ratio_mad.loc[subs, ch])),
                between_over_within_std=b_std / w_std if w_std > 0 else np.nan,
                between_over_within_robust=b_mad / w_mad if w_mad > 0 else np.nan))
    spread_df = pd.DataFrame(spread_rows)
    ad.save_table(spread_df, tab / "2_spread_ordinary_vs_robust.csv")

    # ---- pooled development scale and leave-one-subject-out influence
    dev_subjects = [s for s in std_w.index if split_of[s] in ad.DEV_SPLITS]
    n_tot = sum(moments[s][0] for s in dev_subjects)
    sum_tot = sum(moments[s][1] for s in dev_subjects)
    ss_tot = sum(moments[s][2] for s in dev_subjects)
    mean_all = sum_tot / n_tot
    std_all = np.sqrt(ss_tot / n_tot - mean_all ** 2)
    pooled = np.concatenate([samples[s] for s in dev_subjects])
    labels = np.concatenate([[s] * SUBSAMPLE_PER_SUBJECT for s in dev_subjects])
    med_all = np.median(pooled, axis=0)
    mad_all = MAD_TO_SIGMA * np.median(np.abs(pooled - med_all), axis=0)

    loso_std = np.zeros((len(dev_subjects), len(channels)))
    loso_mad = np.zeros_like(loso_std)
    for i, s in enumerate(dev_subjects):
        n_s, sum_s, ss_s = moments[s]
        n, su, sq = n_tot - n_s, sum_tot - sum_s, ss_tot - ss_s
        loso_std[i] = np.sqrt(sq / n - (su / n) ** 2)
        rest = pooled[labels != s]
        m = np.median(rest, axis=0)
        loso_mad[i] = MAD_TO_SIGMA * np.median(np.abs(rest - m), axis=0)
    rel_std = np.abs(loso_std - std_all) / std_all
    rel_mad = np.abs(loso_mad - mad_all) / mad_all

    centred = np.abs(pooled - pooled.mean(axis=0))
    top1_share = np.empty(len(channels))
    for ci in range(len(channels)):
        v = centred[:, ci]
        cut = np.percentile(v, 99)
        top1_share[ci] = (v[v >= cut] ** 2).sum() / (v ** 2).sum()
    z = stats.norm.ppf(0.995)
    gaussian_ref = 2 * (z * stats.norm.pdf(z) + stats.norm.sf(z))

    pooled_df = pd.DataFrame(dict(
        channel=channels, pooled_mean=mean_all, pooled_std=std_all,
        pooled_median=med_all, pooled_mad_sigma=mad_all,
        inflation_std_over_mad_sigma=std_all / mad_all,
        energy_share_of_top1pct_samples=top1_share, gaussian_reference_share=gaussian_ref,
        loso_max_rel_change_std=rel_std.max(axis=0),
        loso_max_rel_change_mad_sigma=rel_mad.max(axis=0),
        most_influential_subject_for_std=[dev_subjects[i] for i in rel_std.argmax(axis=0)]))
    ad.save_table(pooled_df, tab / "2_pooled_development_scale_and_influence.csv")

    # review list: NOT an exclusion list
    cells = subj_df.copy()
    cells["group"] = np.where(cells.split.isin(ad.DEV_SPLITS), "development", "test")
    review_cells = (cells[cells.group == "development"]
                    .sort_values("std_over_mad_sigma", ascending=False).head(15))
    review_cells = review_cells[["subject", "split", "channel", "std", "mad_sigma",
                                 "std_over_mad_sigma"]].copy()
    review_cells.insert(0, "note", "FOR HUMAN REVIEW ONLY - not an exclusion list")
    ad.save_table(review_cells, tab / "2_review_list_NOT_exclusions.csv")

    # ---------------- figure 2: ordinary vs robust scale, per subject-channel
    f, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for group, color, lbl in (("development", "0.4", "development"),
                              ("test", "#d62728", "test (descriptive)")):
        sub = cells[cells.group == group]
        ax.scatter(sub.mad_sigma, sub["std"], s=14, alpha=0.65, color=color, label=lbl)
    lims = [cells.mad_sigma.min() * 0.8, cells["std"].max() * 1.2]
    ax.plot(lims, lims, "k--", lw=0.8, label="std = robust sigma")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("robust sigma (1.4826 x MAD)"); ax.set_ylabel("ordinary std")
    ax.set_title("per subject x channel: ordinary vs robust scale")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax = axes[1]
    for group, color, ls in (("development", "0.3", "-"), ("test", "#d62728", "--")):
        r = np.sort(cells[cells.group == group].std_over_mad_sigma.to_numpy())
        ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=color, ls=ls, lw=2, label=group)
    ax.axvline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("std / robust sigma (about 1 for Gaussian data)"); ax.set_ylabel("ECDF")
    ax.set_title("how far extreme values inflate the ordinary std")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ad.save_fig(f, fig / "2_std_vs_robust_scale.png")

    # ---------------- figure 3: spread, pooled scale, influence
    dev_spread = spread_df[spread_df.group == "development"].set_index("channel").loc[channels]
    pooled_i = pooled_df.set_index("channel").loc[channels]
    xs = np.arange(len(channels))
    f, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].bar(xs - 0.2, dev_spread.between_sd_log2_std, 0.4, label="ordinary std")
    axes[0].bar(xs + 0.2, dev_spread.between_sd_log2_robust, 0.4, label="robust sigma")
    axes[0].set_title("between-subject spread (development)")
    axes[0].set_ylabel("SD of log2(scale) across subjects")
    axes[1].bar(xs - 0.2, pooled_i.pooled_std, 0.4, label="pooled std")
    axes[1].bar(xs + 0.2, pooled_i.pooled_mad_sigma, 0.4, label="pooled robust sigma")
    axes[1].set_title("pooled development scale (what a global scaler would use)")
    axes[1].set_ylabel("data units")
    axes[2].bar(xs - 0.2, 100 * pooled_i.loso_max_rel_change_std, 0.4, label="std")
    axes[2].bar(xs + 0.2, 100 * pooled_i.loso_max_rel_change_mad_sigma, 0.4,
                label="robust sigma")
    axes[2].set_title("largest change when ONE subject is dropped")
    axes[2].set_ylabel("% change of pooled scale")
    for ax in axes:
        ax.set_xticks(xs, channels, rotation=60, fontsize=8)
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    ad.save_fig(f, fig / "3_spread_pooled_scale_and_influence.png")
    return spread_df, pooled_df, subj_df


# ----------------------------------------------------------------------
# summary: numbers + transparent heuristic reading (NO decisions)
# ----------------------------------------------------------------------
def write_summary(path, conc_df, rank_df, subj_ext_df, spread_df, pooled_df, subj_rob_df,
                  split_of):
    pd.set_option("display.width", 200)
    L = []
    add = L.append
    add("STEP 3b AUDIT SUMMARY - DESCRIPTIVE ONLY (nothing removed, normalized or trained)")
    add("Decision-eligible group: development (train+validation). Test = descriptive only.")
    add("Thresholds 100/150/200 are EXPLORATORY. 'Heuristic reading' lines use the rules")
    add(f"  printed here (concentration ratio >= {CONC_STRONG}/{CONC_MODERATE}, std/robust "
        f">= {INFLATION_STRONG}/{INFLATION_MILD}); they label numbers, they are not decisions.")
    add("")

    # ---- A
    add("A. CONCENTRATION OF EXTREME AMPLITUDES")
    w = conc_df[(conc_df.level == "window (subject/recording)") & (conc_df.group == "development")
                & (conc_df.condition.isin(["low", "high"]))]
    fair = float(w.fair_share.iloc[0])
    add("  Exploratory thresholds, development, subject/recording level "
        f"(equal-share expectation for the top {TOP_SUBJECT_FRACTION:.0%} of subjects = {fair:.1%}):")
    show = w[["exploratory_threshold", "condition", "total_extreme", "top_k_share",
              "conc_ratio", "gini", "units_for_50pct"]].round(2)
    add("  " + show.to_string(index=False).replace("\n", "\n  "))
    add("  Rank-based extremes (top q% of development windows by max peak-to-peak):")
    add("  " + rank_df[["top_percent_of_dev_windows", "value_at_rank_cutoff", "n_windows",
                         "n_subjects_contributing", "pct_from_high_class", "top_k_share",
                         "conc_ratio", "gini", "units_for_50pct"]].round(2)
        .to_string(index=False).replace("\n", "\n  "))
    ratio5 = float(rank_df.loc[rank_df.top_percent_of_dev_windows == 5, "conc_ratio"].iloc[0])
    cell = conc_df[(conc_df.level == "cell (subject x channel)") & (conc_df.group == "development")
                   & (conc_df.condition == "both")]
    cell_ratio = float(cell.conc_ratio.max())
    worst = max(ratio5, cell_ratio)
    verdict_a = ("CONCENTRATED in a few subjects/cells" if worst >= CONC_STRONG else
                 "MODERATELY concentrated" if worst >= CONC_MODERATE else
                 "WIDESPREAD (not dominated by a few subjects)")
    add(f"  Heuristic reading: rank-based top-5% ratio = {ratio5:.2f}, "
        f"largest cell-level ratio = {cell_ratio:.2f} -> {verdict_a}.")
    top5 = (subj_ext_df[(subj_ext_df.exploratory_threshold == max(ad.EXPLORATORY_PTP_THRESHOLDS))
                        & subj_ext_df.subject.isin(
                            [s for s in split_of.index if split_of[s] in ad.DEV_SPLITS])]
            .groupby("subject").n_extreme.sum().sort_values(ascending=False).head(5))
    add(f"  Development subjects with most windows >{max(ad.EXPLORATORY_PTP_THRESHOLDS)} "
        f"(both classes, of 148): " + ", ".join(f"S{s}={n}" for s, n in top5.items()))
    add("")

    # ---- B
    add("B. EFFECT OF EXTREME VALUES ON ORDINARY MEAN/STD SCALING (development)")
    p = pooled_df
    add("  " + p[["channel", "pooled_std", "pooled_mad_sigma", "inflation_std_over_mad_sigma",
                  "energy_share_of_top1pct_samples", "loso_max_rel_change_std",
                  "loso_max_rel_change_mad_sigma"]].round(3)
        .to_string(index=False).replace("\n", "\n  "))
    infl = float(p.inflation_std_over_mad_sigma.median())
    ratio_cells = subj_rob_df[subj_rob_df.split.isin(ad.DEV_SPLITS)].std_over_mad_sigma
    add(f"  (pooled std and its leave-one-out change are exact; pooled robust sigma uses "
        f"{SUBSAMPLE_PER_SUBJECT} random samples per subject, seed {SEED}.)")
    add(f"  Gaussian reference: top 1% of samples hold {float(p.gaussian_reference_share.iloc[0]):.3f}"
        " of the energy.")
    add(f"  Median (over channels) pooled std/robust sigma = {infl:.2f}; per subject-channel "
        f"std/robust sigma: median {ratio_cells.median():.2f}, p90 {ratio_cells.quantile(.9):.2f}, "
        f"max {ratio_cells.max():.2f}.")
    sd = spread_df[spread_df.group == "development"]
    add(f"  Between-subject spread (SD log2): ordinary median {sd.between_sd_log2_std.median():.2f}"
        f" vs robust median {sd.between_sd_log2_robust.median():.2f}.")
    add(f"  Largest single-subject change of pooled scale (median over channels): "
        f"std {100 * p.loso_max_rel_change_std.median():.1f}% vs "
        f"robust {100 * p.loso_max_rel_change_mad_sigma.median():.1f}%.")
    verdict_b = ("STRONGLY affected" if infl >= INFLATION_STRONG else
                 "MILDLY affected" if infl >= INFLATION_MILD else "NOT materially affected")
    add(f"  Heuristic reading: ordinary z-score scale is {verdict_b}.")
    add("")

    # ---- C
    add("C. IS ROBUST SCALING WORTH CARRYING INTO STEP 4?")
    keep = verdict_b != "NOT materially affected"
    add(f"  Heuristic reading: {'YES' if keep else 'OPTIONAL'} - include robust "
        "(median/MAD) scaling as a CANDIDATE next to global z-score and per-subject scaling.")
    add("  Whether it actually helps is a Step 4 question (subject-wise CV, development only).")
    add("")

    # ---- D
    add("D. IS THERE A DEFENSIBLE REASON TO REMOVE RECORDINGS/WINDOWS?")
    dev_rec = subj_ext_df[subj_ext_df.subject.isin(
        [s for s in split_of.index if split_of[s] in ad.DEV_SPLITS])]
    t_max = max(ad.EXPLORATORY_PTP_THRESHOLDS)
    heavy = dev_rec[(dev_rec.exploratory_threshold == t_max) & (dev_rec.pct_extreme >= 90)]
    n_cells_review = int((ratio_cells >= REVIEW_RATIO).sum())
    add(f"  Development recordings with >=90% of windows above {t_max} (exploratory): "
        f"{len(heavy)} of {len(dev_rec[dev_rec.exploratory_threshold == t_max])}.")
    add(f"  Development subject-channel cells with std/robust >= {REVIEW_RATIO}: {n_cells_review} "
        f"of {len(ratio_cells)} (listed in 2_review_list_NOT_exclusions.csv for human review).")
    add("  Nothing was removed. Any exclusion would need an a-priori, label-blind, documented")
    add("  rule fixed on development data only; the exploratory thresholds are not validated")
    add("  artifact limits and a rule that removes ~half of the windows would not be defensible")
    add("  without evidence that it helps under subject-wise CV (a Step 4 comparison).")
    text = "\n".join(L)
    path.write_text(text, encoding="utf-8")
    print("\n" + text)


# ----------------------------------------------------------------------
def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Step 3b robust / concentration audit.")
    parser.add_argument("--data-dir", type=Path,
                        default=repo_root / "processed_data" / "contract_v1")
    parser.add_argument("--out-dir", type=Path,
                        default=repo_root / "results" / "step3b_audit")
    args = parser.parse_args()
    tab, fig = args.out_dir / "tables", args.out_dir / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    inputs = {n: args.data_dir / n for n in ("X_filtered.npy", "metadata.csv", "splits.json")}
    before = {n: ad.file_sha256(p) for n, p in inputs.items()}

    X, meta, splits, channels, split_of, config = ad.load_inputs(args.data_dir)
    print(f"X {X.shape} | subjects {meta['subject'].nunique()} | groups: "
          + ", ".join(f"{k}={int((split_of == k).sum())}" for k in ("train", "validation", "test")))

    conc_df, rank_df, subj_ext_df, _ = part1_concentration(X, meta, channels, split_of, tab, fig)
    spread_df, pooled_df, subj_rob_df = part2_robust(X, meta, channels, split_of, tab, fig)
    write_summary(args.out_dir / "audit3b_summary.txt", conc_df, rank_df, subj_ext_df,
                  spread_df, pooled_df, subj_rob_df, split_of)

    after = {n: ad.file_sha256(p) for n, p in inputs.items()}
    assert before == after, "An input file changed during the audit!"
    (args.out_dir / "run_info.json").write_text(json.dumps({
        "step": "3b - concentration + robust statistics (descriptive, no decisions)",
        "input_sha256": after, "inputs_unchanged": True,
        "exploratory_ptp_thresholds": list(ad.EXPLORATORY_PTP_THRESHOLDS),
        "rank_based_percents": list(RANK_PERCENTS),
        "subsample_per_subject_for_pooled_robust": SUBSAMPLE_PER_SUBJECT, "seed": SEED,
        "windows_removed": 0, "subjects_removed": 0, "normalization_applied": False,
        "heuristic_rules": {"conc_strong": CONC_STRONG, "conc_moderate": CONC_MODERATE,
                            "inflation_strong": INFLATION_STRONG,
                            "inflation_mild": INFLATION_MILD, "review_ratio": REVIEW_RATIO},
        "python": platform.python_version()}, indent=2))
    print("\n[OK] input files unchanged (SHA-256 identical before and after).")
    print(f"[OK] results saved under: {args.out_dir}")


if __name__ == "__main__":
    main()