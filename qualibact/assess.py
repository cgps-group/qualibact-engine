#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from pathlib import Path
import argparse
import textwrap

# =========================
# CONFIG
# =========================

MAX_GMM_COMPONENTS = 3
MIN_N_FOR_MULTIVARIATE = 30
MIN_N_FOR_GC_GMM = 30
WEIRD_SUBGROUP_MIN_WEIGHT = 0.02
WEIRD_SUBGROUP_MAX_WEIGHT = 0.2
WEIRD_SUBGROUP_MIN_D2 = 15.0
GC_IQR_VARIATION_MIN = 0.0005
COMPLETENESS_FAIL_LT = 60.0
CONTAMINATION_FAIL_GT = 30.0

DEFAULT_PLOTS_DIRNAME = "qc_plots"
TITLE_WRAP_WIDTH = 44
LABEL_PERCENTILE = 90


# =========================
# ROBUST STATS
# =========================

# =========================
# MULTIVARIATE MIXTURE
# =========================

def _pick_first_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _fit_best_gmm_1d(x, max_components=3):
    x = x.reshape(-1, 1)
    best = None
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=0,
        ).fit(x)
        if best is None or gmm.bic(x) < best.bic(x):
            best = gmm
    return best


def _gc_gmm_flags(df):
    if "GC_Content" not in df.columns:
        return np.zeros(len(df), dtype=bool), {
            "gc_iqr": np.nan,
            "gc_variation_flag": False,
            "gc_gmm_flagged_frac": 0.0,
        }

    gc_values = df["GC_Content"].values
    gc_values = gc_values[np.isfinite(gc_values)]
    if len(gc_values) == 0:
        return np.zeros(len(df), dtype=bool), {
            "gc_iqr": np.nan,
            "gc_variation_flag": False,
            "gc_gmm_flagged_frac": 0.0,
        }

    q1, q3 = np.percentile(gc_values, [25, 75])
    gc_iqr = q3 - q1
    gc_variation_flag = gc_iqr > GC_IQR_VARIATION_MIN

    if not gc_variation_flag or len(gc_values) < MIN_N_FOR_GC_GMM:
        return np.zeros(len(df), dtype=bool), {
            "gc_iqr": gc_iqr,
            "gc_variation_flag": gc_variation_flag,
            "gc_gmm_flagged_frac": 0.0,
        }

    gmm = _fit_best_gmm_1d(gc_values, MAX_GMM_COMPONENTS)
    if gmm is None or gmm.n_components == 1:
        return np.zeros(len(df), dtype=bool), {
            "gc_iqr": gc_iqr,
            "gc_variation_flag": gc_variation_flag,
            "gc_gmm_flagged_frac": 0.0,
        }

    weights = gmm.weights_
    minor = int(np.argmin(weights))
    if not (WEIRD_SUBGROUP_MIN_WEIGHT < weights[minor] < WEIRD_SUBGROUP_MAX_WEIGHT):
        return np.zeros(len(df), dtype=bool), {
            "gc_iqr": gc_iqr,
            "gc_variation_flag": gc_variation_flag,
            "gc_gmm_flagged_frac": 0.0,
        }

    labels = gmm.predict(gc_values.reshape(-1, 1))
    minor_mask = labels == minor

    flags = np.zeros(len(df), dtype=bool)
    valid_idx = np.where(np.isfinite(df["GC_Content"].values))[0]
    flags[valid_idx] = minor_mask

    return flags, {
        "gc_iqr": gc_iqr,
        "gc_variation_flag": gc_variation_flag,
        "gc_gmm_flagged_frac": float(weights[minor]),
    }


def _multivariate_gmm_flags(df, x_col, y_col):
    if x_col not in df.columns or y_col not in df.columns:
        return np.zeros(len(df), dtype=bool), {
            "has_weird_subgroup": False,
            "weird_subgroup_frac": 0.0,
            "weird_subgroup_separation": np.nan,
            "x_col": x_col,
            "y_col": y_col,
            "gmm": None,
            "labels": None,
        }

    x1 = df[x_col].values
    x2 = df[y_col].values

    valid = (
        np.isfinite(x1) & np.isfinite(x2) &
        (x1 > 0) & (x2 > 0)
    )

    if valid.sum() < MIN_N_FOR_MULTIVARIATE:
        return np.zeros(len(df), dtype=bool), {
            "has_weird_subgroup": False,
            "weird_subgroup_frac": 0.0,
            "weird_subgroup_separation": np.nan,
            "x_col": x_col,
            "y_col": y_col,
            "gmm": None,
            "labels": None,
        }

    X = np.column_stack([np.log10(x1[valid]), np.log10(x2[valid])])

    best = None
    for k in range(1, MAX_GMM_COMPONENTS + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=0,
        ).fit(X)
        if best is None or gmm.bic(X) < best.bic(X):
            best = gmm

    if best is None or best.n_components == 1:
        return np.zeros(len(df), dtype=bool), {
            "has_weird_subgroup": False,
            "weird_subgroup_frac": 0.0,
            "weird_subgroup_separation": np.nan,
            "x_col": x_col,
            "y_col": y_col,
            "gmm": best,
            "labels": None,
        }

    weights = best.weights_
    means = best.means_
    covariances = best.covariances_

    dominant = int(np.argmax(weights))
    minor = int(np.argmin(weights))

    delta = means[dominant] - means[minor]
    try:
        sigma_inv = np.linalg.inv(covariances[dominant])
        d2 = float(delta.T @ sigma_inv @ delta)
    except np.linalg.LinAlgError:
        d2 = np.nan

    is_weird_subgroup = (
        best.n_components > 1
        and weights[minor] < WEIRD_SUBGROUP_MAX_WEIGHT
        and weights[minor] > WEIRD_SUBGROUP_MIN_WEIGHT
        and np.isfinite(d2)
        and d2 > WEIRD_SUBGROUP_MIN_D2
    )

    weird_flags = np.zeros(len(df), dtype=bool)
    labels = best.predict(X)
    if is_weird_subgroup:
        weird_labels = labels == minor
        weird_flags[np.where(valid)[0]] = weird_labels

    return weird_flags, {
        "has_weird_subgroup": bool(is_weird_subgroup),
        "weird_subgroup_frac": float(weights[minor]),
        "weird_subgroup_separation": d2,
        "x_col": x_col,
        "y_col": y_col,
        "gmm": best,
        "labels": labels,
    }


def plot_multivariate_gmm(df, info, outdir, species, label):
    if not info.get("has_weird_subgroup", False):
        return

    gmm = info.get("gmm")
    labels = info.get("labels")
    x_col = info.get("x_col")
    y_col = info.get("y_col")

    if gmm is None or labels is None or x_col is None or y_col is None:
        return

    x1 = df[x_col].values
    x2 = df[y_col].values
    valid = (
        np.isfinite(x1) & np.isfinite(x2) &
        (x1 > 0) & (x2 > 0)
    )
    X = np.column_stack([np.log10(x1[valid]), np.log10(x2[valid])])

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=8, alpha=0.7)
    ax.set_xlabel(f"log10({x_col})")
    ax.set_ylabel(f"log10({y_col})")

    weights = gmm.weights_
    dominant = int(np.argmax(weights))
    minor = int(np.argmin(weights))
    d2 = info.get("weird_subgroup_separation", np.nan)

    title = (
        f"{species} — {label} GMM "
        f"(minor={weights[minor]:.3f}, dominant={weights[dominant]:.3f}, d2={d2:.2f})"
    )
    ax.set_title(textwrap.fill(title, TITLE_WRAP_WIDTH), fontsize=10)

    fig.tight_layout()
    out = outdir / f"{species.replace(' ', '_')}__{label}_gmm.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


# =========================
# MAIN ASSESSMENT
# =========================

def assess_species(df, species, outdir):
    df = df.copy()

    # Pair 1: Genome_Size vs Total_Coding_Sequences
    size_col = _pick_first_column(df, ["Genome_Size", "total_length"])
    cds_col = _pick_first_column(df, ["Total_Coding_Sequences"])
    pair1_flags, pair1_info = _multivariate_gmm_flags(df, size_col, cds_col)
    df["bad_pair_genome_size_cds"] = pair1_flags

    # Pair 2: total_length / Genome_Size vs N50
    n50_col = _pick_first_column(df, ["N50", "Contig_N50"])
    pair2_flags, pair2_info = _multivariate_gmm_flags(df, size_col, n50_col)
    df["bad_pair_length_n50"] = pair2_flags

    # Hard bounds for completeness / contamination
    completeness_col = _pick_first_column(df, ["Completeness_Specific", "Completeness"])
    contamination_col = _pick_first_column(df, ["Contamination"])

    if completeness_col is not None:
        df["bad_completeness"] = df[completeness_col].values < COMPLETENESS_FAIL_LT
    else:
        df["bad_completeness"] = False

    if contamination_col is not None:
        df["bad_contamination"] = df[contamination_col].values > CONTAMINATION_FAIL_GT
    else:
        df["bad_contamination"] = False

    # GC content variation + GMM (species-level gate)
    gc_flags, gc_info = _gc_gmm_flags(df)
    df["bad_gc_gmm"] = gc_flags

    # QC score: stronger weighting for hard bounds and pair flags
    df["qc_score"] = (
        df["bad_pair_genome_size_cds"].astype(int)
        + df["bad_pair_length_n50"].astype(int)
        + df["bad_gc_gmm"].astype(int)
        + 3 * df["bad_completeness"].astype(int)
        + 3 * df["bad_contamination"].astype(int)
    )
    df["qc_class"] = np.where(
        df["qc_score"] == 0, "pass",
        np.where(df["qc_score"] == 1, "warn", "fail")
    )

    summary = {
        "n": len(df),
        "pair_genome_size_cds_has_weird_subgroup": pair1_info["has_weird_subgroup"],
        "pair_genome_size_cds_frac": pair1_info["weird_subgroup_frac"],
        "pair_genome_size_cds_separation": pair1_info["weird_subgroup_separation"],
        "pair_length_n50_has_weird_subgroup": pair2_info["has_weird_subgroup"],
        "pair_length_n50_frac": pair2_info["weird_subgroup_frac"],
        "pair_length_n50_separation": pair2_info["weird_subgroup_separation"],
        "gc_iqr": gc_info["gc_iqr"],
        "gc_variation_flag": gc_info["gc_variation_flag"],
        "gc_gmm_flagged_frac": gc_info["gc_gmm_flagged_frac"],
        "bad_pair_genome_size_cds_count": int(df["bad_pair_genome_size_cds"].sum()),
        "bad_pair_length_n50_count": int(df["bad_pair_length_n50"].sum()),
        "bad_gc_gmm_count": int(df["bad_gc_gmm"].sum()),
        "bad_completeness_count": int(df["bad_completeness"].sum()),
        "bad_contamination_count": int(df["bad_contamination"].sum()),
    }

    plot_multivariate_gmm(df, pair1_info, outdir, species, "genome_size_vs_cds")
    plot_multivariate_gmm(df, pair2_info, outdir, species, "length_vs_n50")

    metric_results = pd.DataFrame(
        [
            {
                "rule": "pair_genome_size_cds",
                "has_weird_subgroup": pair1_info["has_weird_subgroup"],
                "fraction_flagged": pair1_info["weird_subgroup_frac"],
                "separation": pair1_info["weird_subgroup_separation"],
            },
            {
                "rule": "pair_length_n50",
                "has_weird_subgroup": pair2_info["has_weird_subgroup"],
                "fraction_flagged": pair2_info["weird_subgroup_frac"],
                "separation": pair2_info["weird_subgroup_separation"],
            },
            {
                "rule": "gc_variation",
                "gc_iqr": gc_info["gc_iqr"],
                "gc_variation_flag": gc_info["gc_variation_flag"],
                "gc_gmm_flagged_frac": gc_info["gc_gmm_flagged_frac"],
            },
            {
                "rule": "completeness_lt_60",
                "count_flagged": int(df["bad_completeness"].sum()),
            },
            {
                "rule": "contamination_gt_30",
                "count_flagged": int(df["bad_contamination"].sum()),
            },
        ]
    )

    return df, metric_results, summary


# =========================
# EXAMPLE USAGE
# =========================

def _resolve_species_name(df, summary_df, fallback_name):
    for col in ("species", "species_sylph"):
        if col in df.columns:
            value = df[col].dropna().astype(str)
            if not value.empty:
                return value.iloc[0]
    if summary_df is not None and "species" in summary_df.columns:
        value = summary_df["species"].dropna().astype(str)
        if not value.empty:
            return value.iloc[0]
    return fallback_name


def _load_genomes_csv(input_dir, genomes_csv):
    if genomes_csv is not None:
        path = Path(genomes_csv)
        return path, pd.read_csv(path)

    matches = sorted(Path(input_dir).glob("*_high_quality_genomes.csv"))
    if matches:
        path = matches[0]
        return path, pd.read_csv(path)

    default_path = Path(input_dir) / "qc_genome_results.csv"
    return default_path, pd.read_csv(default_path)


def _load_summary_csv(input_dir, summary_csv):
    path = Path(summary_csv) if summary_csv is not None else Path(input_dir) / "summary.csv"
    if path.exists():
        return path, pd.read_csv(path)
    return path, None


def _normalize_columns(df):
    if "Completeness_Specific" not in df.columns and "Completeness" in df.columns:
        df = df.copy()
        df["Completeness_Specific"] = df["Completeness"]
    return df


def _find_species_dirs(root_dir):
    root = Path(root_dir)
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if path.name == "all_summary":
            continue
        genomes = list(path.glob("*_high_quality_genomes.csv"))
        summary = path / "summary.csv"
        if genomes and summary.exists():
            yield path


def _plot_species_scatter(df, outpath):
    if df.empty:
        return
    x = df["pair_genome_size_cds_separation"].values
    y = df["pair_length_n50_separation"].values
    flagged = (df["bad_completeness_count"] > 0) | (df["bad_contamination_count"] > 0)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    ax.scatter(x[~flagged], y[~flagged], s=24, alpha=0.7, label="no hard fails")
    ax.scatter(x[flagged], y[flagged], s=28, alpha=0.8, label="hard fails", color="tab:red")

    ax.set_xlabel("Genome_Size vs CDS separation (d2)")
    ax.set_ylabel("Genome_Size/total_length vs N50 separation (d2)")
    ax.legend(frameon=False)

    x_thresh = np.nanpercentile(x, LABEL_PERCENTILE)
    y_thresh = np.nanpercentile(y, LABEL_PERCENTILE)

    for _, row in df.iterrows():
        if row["pair_genome_size_cds_separation"] >= x_thresh or row["pair_length_n50_separation"] >= y_thresh:
            ax.annotate(
                row["species"],
                (row["pair_genome_size_cds_separation"], row["pair_length_n50_separation"]),
                fontsize=8,
                alpha=0.9,
            )

    ax.set_title(
        f"Species separations (label >= {LABEL_PERCENTILE}th percentile)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assess genome QC distributions from a species output directory."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Directory containing summary.csv and *_high_quality_genomes.csv",
    )
    parser.add_argument(
        "--genomes-csv",
        default=None,
        help="Path to per-genome metrics CSV (defaults to *_high_quality_genomes.csv)",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Path to summary.csv (used for species name fallback)",
    )
    parser.add_argument(
        "--qc-output",
        default=None,
        help="Output CSV for per-genome QC results (default: input_dir/qc_genome_results.csv)",
    )
    parser.add_argument(
        "--metric-output",
        default=None,
        help="Output CSV for metric summary (default: input_dir/qc_metric_summary.csv)",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help=f"Directory for plots (default: input_dir/{DEFAULT_PLOTS_DIRNAME})",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all species subdirectories under input_dir",
    )
    return parser.parse_args()


def run_assessment(args):
    """Run the QC assessment with a pre-built args namespace. Callable from qualibact-run.py."""
    if args.batch:
        root = Path(args.input_dir)
        summaries = []
        for species_dir in _find_species_dirs(root):
            genomes_path, df = _load_genomes_csv(species_dir, None)
            summary_path, summary_df = _load_summary_csv(species_dir, None)
            df = _normalize_columns(df)
            fallback_species = species_dir.name.replace("_", " ")
            species = _resolve_species_name(df, summary_df, fallback_species)
            plots_dir = species_dir / DEFAULT_PLOTS_DIRNAME
            plots_dir.mkdir(parents=True, exist_ok=True)
            qc_df, metric_info, summary = assess_species(df, species, plots_dir)
            (species_dir / "qc_genome_results.csv").write_text(qc_df.to_csv(index=False))
            (species_dir / "qc_metric_summary.csv").write_text(metric_info.to_csv(index=False))
            summaries.append({"species": species, **summary})
        summary_df = pd.DataFrame(summaries)
        if not summary_df.empty:
            summary_out = root / "qc_species_summary.csv"
            summary_df.to_csv(summary_out, index=False)
            _plot_species_scatter(summary_df, root / "qc_species_separation_scatter.png")
        print(f"Batch processed: {len(summaries)} species")
        return

    genomes_path, df = _load_genomes_csv(args.input_dir, args.genomes_csv)
    summary_path, summary_df = _load_summary_csv(args.input_dir, args.summary_csv)
    df = _normalize_columns(df)
    fallback_species = Path(args.input_dir).name.replace("_", " ")
    species = _resolve_species_name(df, summary_df, fallback_species)
    plots_dir = Path(args.plots_dir) if args.plots_dir else Path(args.input_dir) / DEFAULT_PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)
    qc_df, metric_info, summary = assess_species(df, species, plots_dir)
    qc_output = Path(args.qc_output) if args.qc_output else Path(args.input_dir) / "qc_genome_results.csv"
    metric_output = Path(args.metric_output) if args.metric_output else Path(args.input_dir) / "qc_metric_summary.csv"
    qc_df.to_csv(qc_output, index=False)
    metric_info.to_csv(metric_output, index=False)
    print(f"QC results: {qc_output}")
    print(f"Metric summary: {metric_output}")
    print(f"Plots: {plots_dir}")
    print(f"\n  n: {summary['n']}")
    print(f"  gc_variation_flag: {summary['gc_variation_flag']}")
    print(f"  bad_completeness: {summary['bad_completeness_count']}")
    print(f"  bad_contamination: {summary['bad_contamination_count']}")
    print(f"  bad_gc_gmm: {summary['bad_gc_gmm_count']}")
    print(f"  bad_pair_genome_size_cds: {summary['bad_pair_genome_size_cds_count']}")
    print(f"  bad_pair_length_n50: {summary['bad_pair_length_n50_count']}")


if __name__ == "__main__":
    args = parse_args()
    run_assessment(args)
