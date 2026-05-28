"""
analyze_latency.py
==================
All-in-one analysis script for the *Fake It Till You Make It* project.

Produces both:
  1. System-performance benchmark (descriptive stats + plots) from
     `performance_log.csv` written by the Python bridge.
  2. User-evaluation analysis (subscale stats, one-sample Wilcoxon vs.
     neutral, plots) from the Google-Forms responses CSV.

Outputs every figure and a `report_numbers.txt` summary into ./analysis_out/.

Usage:
    python analyze_latency.py
        --latency_csv ../performance_log.csv
        --user_csv    ../Fake_it_till_you_make_it__Responses__-_Form_Responses_1.csv

Both arguments are optional; defaults below assume both CSVs sit one
directory above this script. If either file is missing, that section is
skipped and the rest still runs.
"""

import argparse
import os
import sys
import textwrap
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #

LATENCY_COLUMNS_FRAME_LEVEL = [
    "fer_ms",
    "ser_ms",
    "fusion_ms",
    "udp_send_ms",
    "total_ms",
]
LATENCY_COLUMN_WHISPER = "whisper_inference_ms"  # logged once per real STT call

DEFAULT_OUT_DIR = "analysis_out"


# ------------------------------------------------------------------ #
# Latency analysis                                                    #
# ------------------------------------------------------------------ #

def summarize_series(name: str, s: pd.Series) -> dict:
    s = s.dropna()
    return {
        "metric": name,
        "n": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "p95": s.quantile(0.95),
        "p99": s.quantile(0.99),
        "max": s.max(),
    }


def analyze_latency(csv_path: str, out_dir: str, log, warmup: int = 50) -> Optional[pd.DataFrame]:
    if not os.path.exists(csv_path):
        log(f"[skip] latency CSV not found at: {csv_path}")
        return None

    df_raw = pd.read_csv(csv_path)
    log(f"[ok]  latency CSV loaded: {csv_path}  rows={len(df_raw)}  cols={list(df_raw.columns)}")

    # Drop the first `warmup` frames to exclude cold-start / model-init spikes.
    # PyTorch graph construction, CUDA/CPU kernel JIT, and MediaPipe session
    # initialization all happen on the very first inferences and are not
    # representative of steady-state behaviour.
    if warmup > 0 and len(df_raw) > warmup:
        df = df_raw.iloc[warmup:].reset_index(drop=True)
        log(f"[ok]  dropped first {warmup} frames as warmup "
            f"(kept {len(df)} of {len(df_raw)} for analysis)")
    else:
        df = df_raw
        if warmup > 0:
            log(f"[warn] warmup={warmup} >= total frames ({len(df_raw)}); "
                f"keeping all rows")

    rows = []

    # Frame-level columns: one row per loop iteration, treat directly
    for col in LATENCY_COLUMNS_FRAME_LEVEL:
        if col in df.columns:
            rows.append(summarize_series(col, df[col]))
        else:
            log(f"[warn] expected column missing: {col}")

    # Whisper inference: same value is cached across many main-loop frames, so
    # deduplicate consecutive identical values to recover one row per real
    # transcription call. The bridge sleeps WHISPER_INTERVAL = 3 s between
    # calls, so the *user-perceived* transcript refresh rate is bounded by
    # that interval and is separate from the model's inference time.
    if LATENCY_COLUMN_WHISPER in df.columns:
        whisper_calls = df[LATENCY_COLUMN_WHISPER].dropna()
        unique_calls = whisper_calls.loc[whisper_calls.shift() != whisper_calls]
        unique_calls = unique_calls[unique_calls > 0]
        rows.append(summarize_series(LATENCY_COLUMN_WHISPER, unique_calls))
        log(f"[ok]  whisper_inference_ms: {len(whisper_calls)} logged frames "
            f"\u2192 {len(unique_calls)} unique inference calls after dedup")
        log(f"      design note: Whisper sleeps 3 s between calls, so transcript "
            f"refresh rate \u2264 0.33 Hz regardless of inference time")
    else:
        log(f"[warn] {LATENCY_COLUMN_WHISPER} column not found "
            "(did you patch the bridge with the STT timing fix?)")

    stats = pd.DataFrame(rows).set_index("metric").round(2)
    log("\n=== Latency summary ===")
    log(stats.to_string())

    # CSV out
    stats.to_csv(os.path.join(out_dir, "latency_stats.csv"))

    # Per-component box+strip plot
    plot_cols = [c for c in LATENCY_COLUMNS_FRAME_LEVEL if c in df.columns]
    if plot_cols:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        data = [df[c].dropna().values for c in plot_cols]
        labels = [c.replace("_ms", "") for c in plot_cols]
        bp = ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#9ec5e0")
            patch.set_alpha(0.7)
        # Light jitter strip
        for i, vals in enumerate(data):
            if len(vals) == 0:
                continue
            x_jitter = np.random.normal(i + 1, 0.05, size=len(vals))
            ax.scatter(x_jitter, vals, s=4, alpha=0.15, color="#1f4e79")
        ax.set_yscale("log")
        ax.set_ylabel("Latency (ms, log scale)")
        ax.set_title("Per-component latency distribution")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "latency_components.png"), dpi=150)
        plt.close(fig)
        log(f"[ok]  wrote latency_components.png")

    # FPS proxy: 1000 / total_ms per frame
    if "total_ms" in df.columns:
        fps = 1000.0 / df["total_ms"].replace(0, np.nan).dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(fps, bins=40, color="#e08c63", edgecolor="white")
        ax.axvline(fps.median(), color="black", linestyle="--",
                   label=f"median = {fps.median():.1f} FPS")
        ax.axvline(30, color="red", linestyle=":", label="30 FPS target")
        ax.set_xlabel("Effective main-loop frame rate (1000 / total_ms)")
        ax.set_ylabel("Frames")
        ax.set_title("Effective frame rate of the Python sensing loop")
        ax.legend()
        fig.tight_layout()
        plt.close(fig)
        log(f"[ok]  wrote loop_fps.png   "
            f"median FPS = {fps.median():.2f}  "
            f"p5 FPS = {fps.quantile(0.05):.2f}")

    return stats


# ------------------------------------------------------------------ #
# User-evaluation analysis                                            #
# ------------------------------------------------------------------ #

# Map subscale -> list of (short_name, exact column substring used to locate
# the column in the Google-Forms CSV). Substrings are matched case-insensitively
# and must be unique enough to identify the column.
SUBSCALES = {
    "Understanding": [
        ("understood_concept",            "I understood that the game uses facial"),
        ("understood_modality_per_spell", "I understood which input modality"),
        ("spellbook_helped",              "The spellbook helped me understand"),
    ],
    "Reliability": [
        ("FER_reliable",        "Facial emotion recognition felt reliable"),
        ("SER_reliable",        "Speech emotion recognition felt reliable"),
        ("STT_reliable",        "Spoken keyword recognition felt reliable"),
        ("fusion_correct",      "correctly combined the different inputs"),
        ("failure_diagnosable", "I could understand which modality caused the failure"),
    ],
    "Experience": [
        ("engagement", "made the game more engaging"),
        ("in_control", "I felt in control while using multimodal"),
    ],
}

# Reverse-coded items: higher = worse. We keep these out of the subscales and
# report them separately.
REVERSE_CODED = [
    ("difficulty", "made the game more difficult"),
]

# Categorical preference columns. The OPTIONS list defines every category that
# *could* appear on each question, so the bar plot has the same number of bars
# for every subplot even when some options got zero responses.
CATEGORICAL = [
    ("easiest_input",     "Which input felt easiest to control"),
    ("least_reliable",    "Which input felt least reliable"),
    ("preferred_mode",    "Which casting mode did you prefer"),
]

CATEGORICAL_OPTIONS = {
    "easiest_input":   ["Facial emotion", "Speech emotion", "Spoken keyword",
                        "Manual mode", "Not sure"],
    "least_reliable":  ["Facial emotion", "Speech emotion", "Spoken keyword",
                        "Manual mode", "Not sure"],
    "preferred_mode":  ["FER + SER + STT", "FER + STT", "SER + STT",
                        "STT Only", "SER Only", "Manual"],
}


def _find_column(df: pd.DataFrame, needle: str) -> Optional[str]:
    needle_low = needle.lower()
    for c in df.columns:
        if needle_low in c.lower():
            return c
    return None


def rank_biserial_for_paired(d: pd.Series) -> float:
    """Matched-pairs rank-biserial r for a one-sample Wilcoxon on `d = x - midpoint`."""
    d = d.dropna()
    d = d[d != 0]  # Wilcoxon convention: drop zeros
    if len(d) == 0:
        return float("nan")
    abs_ranks = pd.Series(np.argsort(np.argsort(np.abs(d.values))) + 1, index=d.index)
    W_plus  = abs_ranks[d > 0].sum()
    W_minus = abs_ranks[d < 0].sum()
    denom = W_plus + W_minus
    return (W_plus - W_minus) / denom if denom > 0 else float("nan")


def analyze_user_study(csv_path: str, out_dir: str, log) -> Optional[dict]:
    if not os.path.exists(csv_path):
        log(f"[skip] user-study CSV not found at: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    log(f"\n[ok]  user-study CSV loaded: {csv_path}  n={len(df)} participants")

    # ---- Locate Likert columns ----
    likert_frame = {}
    for subscale, items in SUBSCALES.items():
        for short, needle in items:
            col = _find_column(df, needle)
            if col is None:
                log(f"[warn] column not found for '{short}' (needle={needle!r})")
                continue
            likert_frame[short] = pd.to_numeric(df[col], errors="coerce")

    for short, needle in REVERSE_CODED:
        col = _find_column(df, needle)
        if col is not None:
            likert_frame[short] = pd.to_numeric(df[col], errors="coerce")

    L = pd.DataFrame(likert_frame)

    # ---- Per-item descriptive stats ----
    desc = L.describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc.columns = ["n", "mean", "sd", "min", "median", "max"]
    desc = desc.round(2)
    log("\n=== Per-item descriptives ===")
    log(desc.to_string())
    desc.to_csv(os.path.join(out_dir, "user_study_per_item.csv"))

    # ---- Build subscale scores ----
    subscale_scores = pd.DataFrame()
    for subscale, items in SUBSCALES.items():
        cols_present = [s for s, _ in items if s in L.columns]
        if cols_present:
            subscale_scores[subscale] = L[cols_present].mean(axis=1)

    # ---- One-sample Wilcoxon vs neutral midpoint (3) ----
    log("\n=== Subscale stats (one-sample Wilcoxon vs neutral=3) ===")
    stat_rows = []
    for sub in subscale_scores.columns:
        s = subscale_scores[sub].dropna()
        d = s - 3.0
        d_nonzero = d[d != 0]
        if len(d_nonzero) < 1:
            log(f"  {sub}: all responses at neutral; skipped")
            continue
        stat, p = wilcoxon(d_nonzero)
        r = rank_biserial_for_paired(d)
        stat_rows.append({
            "subscale": sub,
            "n": len(s),
            "median": round(s.median(), 2),
            "mean": round(s.mean(), 2),
            "sd": round(s.std(), 2),
            "W": round(float(stat), 2),
            "p": round(float(p), 4),
            "rank_biserial_r": round(float(r), 3),
            "effect_size_label": _effect_label(r),
        })
    sub_stats = pd.DataFrame(stat_rows)
    log(sub_stats.to_string(index=False))
    sub_stats.to_csv(os.path.join(out_dir, "user_study_subscale_stats.csv"), index=False)

    # ---- Plot 1: per-item violin / strip ----
    item_order = list(L.columns)
    if item_order:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        data = [L[c].dropna().values for c in item_order]
        positions = np.arange(1, len(item_order) + 1)
        parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.7)
        for body in parts["bodies"]:
            body.set_facecolor("#9ec5e0")
            body.set_alpha(0.6)
        for i, vals in enumerate(data, start=1):
            x_jitter = np.random.normal(i, 0.05, size=len(vals))
            ax.scatter(x_jitter, vals, color="#1f4e79", s=18, alpha=0.7)
        ax.axhline(3, color="grey", linestyle="--", alpha=0.6, label="neutral (3)")
        ax.set_xticks(positions)
        ax.set_xticklabels(item_order, rotation=35, ha="right", fontsize=9)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylabel("Likert score (1\u20135)")
        ax.set_title(f"Per-item user-study responses (n = {len(df)})")
        ax.legend(loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "user_study_items.png"), dpi=150)
        plt.close(fig)
        log("[ok]  wrote user_study_items.png")

    # ---- Plot 2: subscale box+strip ----
    if not subscale_scores.empty:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sub_cols = list(subscale_scores.columns)
        data = [subscale_scores[c].dropna().values for c in sub_cols]
        bp = ax.boxplot(data, labels=sub_cols, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("#cce1c4")
            patch.set_alpha(0.7)
        for i, vals in enumerate(data, start=1):
            x_jitter = np.random.normal(i, 0.04, size=len(vals))
            ax.scatter(x_jitter, vals, color="#2c6f3a", s=28, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
        ax.axhline(3, color="grey", linestyle="--", alpha=0.6, label="neutral (3)")
        ax.set_ylim(1, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylabel("Subscale score (1\u20135)")
        ax.set_title(f"User-study subscale scores (n = {len(df)})")
        ax.legend(loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "user_study_subscales.png"), dpi=150)
        plt.close(fig)
        log("[ok]  wrote user_study_subscales.png")

    # ---- Plot 3: categorical preferences ----
    # Every subplot shares the same x-axis range (0..n_participants) and shows
    # ALL possible options, with missing ones rendered as zero-height bars.
    n_total = len(df)
    cat_data = {}
    for short, needle in CATEGORICAL:
        col = _find_column(df, needle)
        if col is None:
            continue
        options = CATEGORICAL_OPTIONS.get(short, [])
        raw_counts = df[col].fillna("(no answer)").value_counts()
        # Build a Series indexed by the canonical option list, then append any
        # extra responses we didn't predeclare (typos, "Other", etc.) at the end.
        full = pd.Series(0, index=options, dtype=int)
        for k, v in raw_counts.items():
            if k in full.index:
                full[k] = v
            else:
                full[k] = v   # unexpected category - keep it visible
        cat_data[short] = full

    if cat_data:
        fig, axes = plt.subplots(1, len(cat_data),
                                 figsize=(4.5 * len(cat_data), 4.2),
                                 sharex=True)
        if len(cat_data) == 1:
            axes = [axes]
        for ax, (label, counts) in zip(axes, cat_data.items()):
            colors = ["#e3a05b" if v > 0 else "#e6e6e6" for v in counts.values]
            counts.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
            ax.set_title(label.replace("_", " "))
            ax.set_xlabel(f"# of participants (n = {n_total})")
            ax.set_xlim(0, n_total)
            ax.set_xticks(range(0, n_total + 1))
            ax.invert_yaxis()
            for i, v in enumerate(counts.values):
                ax.text(v + 0.1, i, str(v), va="center", fontsize=9,
                        color="black" if v > 0 else "#999999")
        fig.suptitle(f"Categorical preferences (n = {n_total})", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "user_study_categorical.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("[ok]  wrote user_study_categorical.png")

    return {
        "n": len(df),
        "per_item": desc,
        "subscale_stats": sub_stats,
        "subscale_scores": subscale_scores,
    }


def _effect_label(r: float) -> str:
    if pd.isna(r):
        return "n/a"
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


# ------------------------------------------------------------------ #
# Report-ready text snippet                                           #
# ------------------------------------------------------------------ #

def write_report_snippet(latency_stats: Optional[pd.DataFrame],
                         user_results: Optional[dict],
                         out_dir: str) -> None:
    lines = []
    lines.append("REPORT-READY NUMBERS")
    lines.append("=" * 60)

    if latency_stats is not None:
        lines.append("\n## System latency (descriptive)\n")
        for metric, row in latency_stats.iterrows():
            lines.append(
                f"  {metric:24s}  n={int(row['n'])}  "
                f"median = {row['median']:.1f} ms  "
                f"p95 = {row['p95']:.1f} ms  "
                f"p99 = {row['p99']:.1f} ms"
            )
        if "total_ms" in latency_stats.index:
            t = latency_stats.loc["total_ms", "median"]
            lines.append(
                f"\n  Effective median loop rate \u2248 {1000.0 / t:.1f} FPS "
                f"(based on median total_ms = {t:.1f} ms)."
            )

    if user_results is not None:
        n = user_results["n"]
        lines.append(f"\n## User evaluation (n = {n})\n")
        for _, r in user_results["subscale_stats"].iterrows():
            lines.append(textwrap.dedent(f"""\
                  {r['subscale']}: median = {r['median']}, mean = {r['mean']} (SD = {r['sd']});
                    one-sample Wilcoxon vs neutral (3): W = {r['W']}, p = {r['p']}, r = {r['rank_biserial_r']} ({r['effect_size_label']} effect).
            """).rstrip())

    text = "\n".join(lines) + "\n"
    out_path = os.path.join(out_dir, "report_numbers.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\n[ok]  wrote {out_path}")


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency_csv",
                        default=os.path.join(here, "..", "performance_log.csv"))
    parser.add_argument("--user_csv",
                        default=os.path.join(here, "..",
                                             "qualitative.csv"))
    parser.add_argument("--out_dir", default=os.path.join(here, DEFAULT_OUT_DIR))
    parser.add_argument("--warmup", type=int, default=50,
                        help="Number of initial frames to drop from latency "
                             "analysis to exclude cold-start / model-init "
                             "spikes (default: 50). Set to 0 to keep all.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[ok]  output directory: {args.out_dir}")

    log_buf = []

    def log(msg: str) -> None:
        print(msg)
        log_buf.append(msg)

    latency_stats = analyze_latency(os.path.abspath(args.latency_csv),
                                    args.out_dir, log, warmup=args.warmup)
    user_results  = analyze_user_study(os.path.abspath(args.user_csv),
                                       args.out_dir, log)

    write_report_snippet(latency_stats, user_results, args.out_dir)

    # also flush the full console log so you have a record
    with open(os.path.join(args.out_dir, "analysis_console.log"), "w",encoding="utf-8") as f:
        f.write("\n".join(log_buf))


if __name__ == "__main__":
    main()