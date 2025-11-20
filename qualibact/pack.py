#!/usr/bin/env python3
"""
Lightweight QualiBact → Next.js packer.

This script expects a QualiBact results directory (with species folders and
`*_assembly_stats.parquet` files) and writes ready-to-serve assets into a
Next.js `public/` directory plus minimal MDX under the adjacent `content/`.

It keeps the logic deliberately simple:
* pick the preferred species/version directories (using preferred_version.yml when present)
* apply the published cut-offs to the assembly parquet to produce filtered/high‑quality CSV.xz
  files that include a rejection_reason column
* flatten the RefSeq JSON into a CSV.xz table
* copy the useful figures/tables while skipping logs and shell scripts
* build site_index.json, per-version manifests, genus summaries, and merged summary tables
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from rich.console import Console
from qualibact.summary_util import tidy_summary, plot_genus_boxplots_from_summary

console = Console()
LOG_VERBOSE = False

# File filters
ALLOWED_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".csv",
    ".xz",
    ".json",
}
SKIP_SUFFIXES = (".log", ".out", ".err", ".sh")
SKIP_DIR_NAMES = {"all_summary", "GENUS_SUMMARY", "filtered_plots"}


# ---------------------------------------------------------------------------
# small helpers

def configure_logging(verbose: bool) -> None:
    global LOG_VERBOSE
    LOG_VERBOSE = verbose


def log(msg: str, verbose: bool = False, level: str = "info") -> None:
    """Console logger with basic level handling."""
    effective_verbose = LOG_VERBOSE or verbose
    if level == "debug" and not effective_verbose:
        return
    styles = {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "debug": "dim",
    }
    console.log(msg, style=styles.get(level, "white"))


def load_yaml(path: Path) -> Dict[str, str]:
    if not path.exists():
        log(f"preferred_version.yml not found at {path}")
        return {}
    log(f"Loading preferred versions from {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "pyarrow or fastparquet is required to read parquet files. "
            "Install pyarrow to continue."
        ) from exc


def to_number(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_species_name(name: str) -> str:
    return name.replace("_", "").replace(" ", "").lower()


# ---------------------------------------------------------------------------
# discovery

def discover_species(input_dir: Path, version_override: Optional[str]) -> Dict[str, List[Tuple[Path, str]]]:
    """Return a mapping of species -> list of (species_dir, version_name)."""
    species_map: Dict[str, List[Tuple[Path, str]]] = {}
    added = set()

    def add_species(species_dir: Path, version: str) -> None:
        species = species_dir.name
        key = (species_dir.resolve(), version)
        if key in added:
            return
        species_map.setdefault(species, []).append((species_dir, version))
        added.add(key)

    for parquet in input_dir.rglob("*_assembly_stats.parquet"):
        # avoid pulling in aggregated outputs
        if any(part in SKIP_DIR_NAMES for part in parquet.parts):
            continue
        species_dir = parquet.parent
        species = species_dir.name
        parent = species_dir.parent
        if parent == input_dir:
            version = version_override or input_dir.name
        else:
            version = parent.name
        add_species(species_dir, version)

    # Fallback: include species directories even without assembly stats so assets still get copied
    for version_dir in input_dir.iterdir():
        if not version_dir.is_dir() or version_dir.name in SKIP_DIR_NAMES:
            continue
        # when a version override is provided, treat input_dir as the version container
        if version_override:
            version = version_override
            species_root = version_dir if version_dir.is_dir() else input_dir
            if version_dir.is_dir():
                species_dir = version_dir
                add_species(species_dir, version)
            continue
        for species_dir in version_dir.iterdir():
            if not species_dir.is_dir() or species_dir.name in SKIP_DIR_NAMES:
                continue
            add_species(species_dir, version_dir.name)
    log(f"Discovered {len(species_map)} species directories")
    return species_map


def select_species_dirs(
    species_map: Dict[str, List[Tuple[Path, str]]], preferred: Dict[str, str]
) -> List[Tuple[str, Path, str]]:
    """
    Return all directories per species; preferred_version just affects ordering (preferred first).
    """
    selected: List[Tuple[str, Path, str]] = []
    for species, options in species_map.items():
        preferred_version = preferred.get(species)
        # sort with preferred first, then by version name for stability
        def sort_key(opt: Tuple[Path, str]) -> Tuple[bool, str]:
            directory, version = opt
            return (0 if preferred_version and version == preferred_version else 1, version)

        ordered = sorted(options, key=sort_key)
        for directory, version in ordered:
            selected.append((species, directory, version))
        log(f"Queued {species} versions {[v for _, v in ordered]} ({len(ordered)} total)")
    return selected


# ---------------------------------------------------------------------------
# data preparation

def load_filtered_metrics(input_dir: Path) -> Optional[pd.DataFrame]:
    # Aggregated filtered_metrics.csv is no longer used; rely on per-species *_metrics.csv instead.
    return None


def load_species_thresholds(
    species: str, metrics_df: Optional[pd.DataFrame], summary_path: Path
) -> pd.DataFrame:
    """
    Return a dataframe with columns metric / lower_bounds / upper_bounds for a species.
    Falls back to summary.csv when aggregated filtered_metrics.csv is missing.
    """
    metrics_path = summary_path.parent / f"{species}_metrics.csv"
    species_norm = normalize_species_name(species)
    if metrics_df is not None:
        if "species" in metrics_df.columns:
            subset = metrics_df[
                metrics_df["species"].apply(lambda s: normalize_species_name(str(s))) == species_norm
            ]
        else:
            subset = metrics_df
        if not subset.empty:
            log(f"Loaded thresholds for {species} from aggregated filtered_metrics")
            return subset[["metric", "lower_bounds", "upper_bounds"]]
    if metrics_path.exists():
        log(f"Loaded thresholds for {species} from {metrics_path}")
        try:
            metrics_df = pd.read_csv(metrics_path)
            if "species" in metrics_df.columns:
                subset = metrics_df[
                    metrics_df["species"].apply(lambda s: normalize_species_name(str(s))) == species_norm
                ]
            else:
                metrics_df["species"] = species
                subset = metrics_df
            if not subset.empty:
                return subset[["metric", "lower_bounds", "upper_bounds"]]
        except Exception as exc:
            log(f"Failed reading metrics CSV for {species}: {exc}", level="warning")
    if summary_path.exists():
        log(f"Deriving thresholds for {species} from summary at {summary_path}")
        summary_df = pd.read_csv(summary_path)
        rounded_df = tidy_summary(
            summary_df,
            summary_path.parent,
            output_filename=f"{species}_metrics.csv",
            plot=False,
        )
        if "species" in rounded_df.columns:
            subset = rounded_df[
                rounded_df["species"].apply(lambda s: normalize_species_name(str(s))) == species_norm
            ]
            # if still empty, allow using the whole rounded_df with normalized species column
            if subset.empty:
                rounded_df["species"] = rounded_df["species"].apply(lambda s: s.replace(" ", "_"))
                subset = rounded_df[
                    rounded_df["species"].apply(lambda s: normalize_species_name(str(s))) == species_norm
                ]
        else:
            rounded_df["species"] = species
            subset = rounded_df
        if not subset.empty:
            return subset[["metric", "lower_bounds", "upper_bounds"]]
    log(f"No thresholds found for {species}; proceeding with empty thresholds")
    return pd.DataFrame(columns=["metric", "lower_bounds", "upper_bounds"])


def apply_filters(
    assembly_stats: Path, thresholds: pd.DataFrame, verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply thresholds to the parquet and return (full_df_with_reasons, filtered_out, high_quality)."""
    df = read_parquet(assembly_stats)
    df = df.copy()
    df.rename(
        columns={"number": "no_of_contigs", "Completeness_Specific": "Completeness"}, inplace=True
    )
    if "GC_Content" in df.columns and df["GC_Content"].max() <= 1:
        df["GC_Content"] = df["GC_Content"] * 100

    reasons: List[List[str]] = [[] for _ in range(len(df))]
    index_positions = {idx: pos for pos, idx in enumerate(df.index)}
    for _, row in thresholds.iterrows():
        metric = row["metric"]
        if metric not in df.columns:
            continue
        lower = to_number(row.get("lower_bounds"))
        upper = to_number(row.get("upper_bounds"))
        if lower is not None:
            mask = df[metric] < lower
            for idx in df.index[mask]:
                pos = index_positions.get(idx)
                if pos is not None:
                    reasons[pos].append(f"{metric} < {lower}")
        if upper is not None:
            mask = df[metric] > upper
            for idx in df.index[mask]:
                pos = index_positions.get(idx)
                if pos is not None:
                    reasons[pos].append(f"{metric} > {upper}")

    rejection_reason = ["; ".join(r) for r in reasons]
    df["rejection_reason"] = rejection_reason
    filtered_out = df[df["rejection_reason"].astype(bool)].copy()
    high_quality = df[df["rejection_reason"] == ""].copy()
    high_quality = high_quality.drop(columns=["rejection_reason"])
    log(
        f"Applied filters to {assembly_stats.name}: {len(filtered_out)} rejected / {len(high_quality)} kept",
        verbose,
    )
    return df, filtered_out, high_quality


def write_xz_csv(df: pd.DataFrame, path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    ensure_dir(path.parent, dry_run=False)
    df.to_csv(path, index=False, compression="xz")


def _flatten_value(value, prefix: str, result: Dict[str, object], sep: str = ".") -> None:
    """
    Recursively flatten dict/list values into key/value pairs.

    Lists of primitives get joined into a semicolon-delimited string to keep a single column.
    Lists containing dict/list values are flattened with positional suffixes.
    """
    key_name = prefix if prefix else "_value"
    if isinstance(value, dict):
        for key, nested in value.items():
            new_prefix = f"{prefix}{sep}{key}" if prefix else key
            _flatten_value(nested, new_prefix, result, sep=sep)
    elif isinstance(value, list):
        if not value:
            result[key_name] = ""
            return
        if all(not isinstance(item, (dict, list)) for item in value):
            result[key_name] = "; ".join("" if item is None else str(item) for item in value)
            return
        for idx, item in enumerate(value):
            new_prefix = f"{prefix}{sep}{idx}" if prefix else str(idx)
            _flatten_value(item, new_prefix, result, sep=sep)
    else:
        result[key_name] = value


def flatten_records(records: List[Dict[str, object]], sep: str = ".") -> List[Dict[str, object]]:
    """Turn a list of nested dict records into fully flattened dicts suitable for CSV export."""
    flattened: List[Dict[str, object]] = []
    for record in records:
        flat_record: Dict[str, object] = {}
        _flatten_value(record, "", flat_record, sep=sep)
        flattened.append(flat_record)
    return flattened


def convert_refseq_json(
    species_dir: Path, species: str, dest_dir: Path, dry_run: bool, verbose: bool
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Flatten the RefSeq JSON into CSV.xz. Returns (public_path, count, last_updated_iso).
    """
    def safe_int(val: Optional[object]) -> Optional[int]:
        try:
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    candidates = [
        species_dir / f"{species}.json",
        species_dir / f"{species.replace('_', ' ')}.json",
    ]
    # Fall back to any json whose stem matches species ignoring underscores/spaces and case
    if not any(p.exists() for p in candidates):
        normalized = species.lower().replace("_", "")
        for json_file in species_dir.glob("*.json"):
            stem_norm = json_file.stem.lower().replace("_", "").replace(" ", "")
            if stem_norm == normalized:
                candidates.append(json_file)
                break
    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        log(f"No RefSeq JSON found for {species} in {species_dir}")
        return None, None, None
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw_reports = data.get("reports")
    raw_genomes = data.get("genomes")
    records = None
    if isinstance(raw_reports, list) and raw_reports:
        records = raw_reports
    elif isinstance(raw_genomes, list) and raw_genomes:
        records = raw_genomes
    count = safe_int(data.get("total_count"))
    last_updated = datetime.fromtimestamp(json_path.stat().st_mtime).isoformat()
    if not records:
        log(f"RefSeq JSON for {species} contained no genomes")
        return None, count, last_updated
    flattened_records = flatten_records(records, sep=".")
    if not flattened_records:
        log(f"RefSeq JSON for {species} contained no genomes")
        return None, count, last_updated
    refseq_df = pd.DataFrame(flattened_records)
    if refseq_df.empty:
        log(f"RefSeq JSON for {species} contained no genomes")
        if count is None:
            count = 0
        return None, count, last_updated
    if count is None:
        count = len(refseq_df)
    out_path = dest_dir / f"{species}_refseq_genomes.csv.xz"
    write_xz_csv(refseq_df, out_path, dry_run)
    log(f"Wrote RefSeq table for {species} -> {out_path}", verbose)
    return f"/{dest_dir.relative_to(dest_dir.parents[1])}/{out_path.name}", count, last_updated


def copy_assets(source_dir: Path, dest_dir: Path, dry_run: bool, verbose: bool) -> None:
    """Copy figures/tables while skipping logs and hidden files."""
    copied = 0
    species_norm = source_dir.name.replace("_", "").lower()
    for root, dirs, files in os.walk(source_dir):
        rel = Path(root).relative_to(source_dir)
        target = dest_dir / rel
        ensure_dir(target, dry_run)
        for file in files:
            if file.startswith("."):
                continue
            if file.endswith(SKIP_SUFFIXES):
                continue
            ext = Path(file).suffix.lower()
            if ext == ".parquet":
                continue
            if ext == ".json":
                stem_norm = Path(file).stem.replace("_", "").replace(" ", "").lower()
                if stem_norm == species_norm:
                    # Skip raw RefSeq JSON copies; CSV.xz is emitted separately
                    continue
            if file.endswith(("_filtered_out_genomes.csv", "_high_quality_genomes.csv")):
                continue
            if ext not in ALLOWED_EXTS and not file.endswith(".csv.xz"):
                continue
            src_path = Path(root) / file
            dst_path = target / file.replace(" ", "_")
            if dry_run:
                log(f"[dry-run] copy {src_path} -> {dst_path}", verbose)
                continue
            ensure_dir(dst_path.parent, dry_run=False)
            shutil.copy2(src_path, dst_path)
            copied += 1
    log(f"Copied {copied} asset files from {source_dir} to {dest_dir}", verbose)


# ---------------------------------------------------------------------------
# outputs

def write_manifest(
    species: str,
    version: str,
    dest_dir: Path,
    *,
    is_latest: bool,
    genome_counts: Optional[Dict[str, int]],
    refseq_path: Optional[str],
    filtered_path: Optional[str],
    highq_path: Optional[str],
    summary_exists: bool,
    selected_summary_exists: bool,
    verbose: bool,
    dry_run: bool,
) -> None:
    manifest = {
        "species": species,
        "version_name": version,
        "timestamp": datetime.now().isoformat(),
        "is_latest": is_latest,
        "modules": [],
        "genome_counts": genome_counts,
        "refseq_table_path": refseq_path,
        "summary_csv_path": f"/{species}/{version}/summary.csv" if summary_exists else None,
        "selected_summary_csv_path": f"/{species}/{version}/selected_summary.csv"
        if selected_summary_exists
        else None,
        "filtered_genomes_path": filtered_path,
        "high_quality_genomes_path": highq_path,
    }
    if dry_run:
        log(f"[dry-run] would write manifest for {species} {version}", verbose)
        return
    ensure_dir(dest_dir, dry_run=False)
    out_path = dest_dir / f"{species}__{version}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    log(f"Wrote manifest -> {out_path}", verbose)


def write_species_content(
    species: str,
    version: str,
    genome_counts: Optional[Dict[str, int]],
    content_dir: Path,
    dry_run: bool,
    overwrite: bool,
    *,
    filtered_out_count: Optional[int] = None,
    high_quality_count: Optional[int] = None,
    metrics_exists: bool = False,
    filtered_path: Optional[str] = None,
    highq_path: Optional[str] = None,
) -> None:
    """Generate minimal MDX files so pages render without manual edits."""
    title = species.replace("_", " ")
    species_dir = content_dir / species
    ensure_dir(species_dir, dry_run)
    index_path = species_dir / "index.mdx"
    version_path = species_dir / f"{version}.mdx"

    header = [
        "---",
        f'title: "{title}"',
        "generated_by: qualibact-pack",
        "---",
        "",
        f"# {title}",
        "",
    ]
    body: List[str] = []
    desc_lines: List[str] = []
    desc_lines.append(
        f"For detailed methods on how these thresholds were calculated, please see [Methods](/methods/{version})."
    )
    if metrics_exists:
        desc_lines.append("The suggested thresholds are in the table below.")

    refseq_count = genome_counts.get("refseq_count") if genome_counts else None
    all_count = genome_counts.get("allthebacteria_count") if genome_counts else None
    other_sources = None
    if refseq_count is not None and all_count is not None:
        try:
            other_sources = max(0, int(all_count) - int(refseq_count))
        except Exception:
            other_sources = None
    refseq_display = refseq_count if refseq_count is not None else 0
    other_display = other_sources if other_sources is not None else 0
    desc_lines.append(
        f"These thresholds are based on **{refseq_display}** genomes from RefSeq and **{other_display}** genomes from other sources."
    )

    if (
        filtered_out_count is not None
        and high_quality_count is not None
        and filtered_path is not None
        and highq_path is not None
    ):
        desc_lines.append(
            f"These thresholds were applied to all the bacteria dataset, which resulted in removing **{filtered_out_count}** "
            f"and retaining **{high_quality_count}**. The list of genomes retained (i.e. high quality) and the list of genomes "
            f"rejected (filtered) can be downloaded below. These files are in .xz format. The rejected genomes file also includes the reason why."
        )

    version_body = body + [""] + desc_lines + [""]

    if not dry_run:
        if not index_path.exists() or overwrite:
            with open(index_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(header + body) + "\n")
        if not version_path.exists() or overwrite:
            with open(version_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(version_body))
    log(f"Wrote content stubs for {species} version {version}")


def write_site_index(
    species_versions: Dict[str, List[str]],
    output_public: Path,
    mark_latest: Optional[str],
    dry_run: bool,
    overwrite: bool,
) -> None:
    site_index = {"species": {}, "last_updated": datetime.now().isoformat()}
    for species, versions in species_versions.items():
        versions_sorted = sorted(set(versions))
        preferred = None
        if mark_latest:
            if mark_latest is True or str(mark_latest).lower() == "true":
                preferred = versions_sorted[-1]
            elif mark_latest in versions_sorted:
                preferred = mark_latest
        if preferred is None and versions_sorted:
            preferred = versions_sorted[-1]
        site_index["species"][species] = {
            "qc_schemes": versions_sorted,
            "preferred_qc_scheme": preferred,
        }
    if dry_run:
        return
    out_path = output_public / "site_index.json"
    ensure_dir(out_path.parent, dry_run=False)
    if out_path.exists() and not overwrite:
        log("site_index.json already exists and overwrite is False; skipping write")
        return
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(site_index, handle, indent=2)
    log(f"Wrote site_index.json with {len(site_index['species'])} species")


def write_genus_summaries(
    species_metrics: Dict[str, pd.DataFrame],
    species_summaries: Dict[str, pd.DataFrame],
    species_counts: Dict[str, Dict[str, int]],
    output_public: Path,
    dry_run: bool,
) -> None:
    if not species_metrics and not species_counts and not species_summaries:
        log("No metrics or counts available; skipping genus summaries")
        return
    genus_groups: Dict[str, List[str]] = {}
    for species in set(species_metrics.keys()) | set(species_counts.keys()) | set(species_summaries.keys()):
        genus = species.split("_")[0]
        genus_groups.setdefault(genus, []).append(species)

    for genus, species_list in genus_groups.items():
        metrics_frames = []
        for species in species_list:
            df = species_metrics.get(species)
            if df is not None and not df.empty:
                metrics_frames.append(df.copy())
        genus_path = output_public / genus
        ensure_dir(genus_path, dry_run)
        if metrics_frames:
            merged_metrics = pd.concat(metrics_frames, ignore_index=True)
            metrics_out = genus_path / f"{genus}_metrics.csv"
            if not dry_run:
                merged_metrics.to_csv(metrics_out, index=False)
            log(f"Wrote genus metrics for {genus} with {len(merged_metrics)} rows")

        summary_frames = []
        for species in species_list:
            summary_df = species_summaries.get(species)
            if summary_df is not None and not summary_df.empty:
                summary_frames.append(summary_df.copy())
        if summary_frames and len(species_list) > 1:
            merged_summary = pd.concat(summary_frames, ignore_index=True)
            plot_genus_boxplots_from_summary(merged_summary, genus_path)

        genus_counts_rows = []
        for species in species_list:
            counts = species_counts.get(species)
            if counts:
                genus_counts_rows.append(counts)
        if genus_counts_rows:
            counts_df = pd.DataFrame(genus_counts_rows)
            counts_df = counts_df.fillna("NA")
            counts_out = genus_path / f"{genus}_counts.csv"
            if not dry_run:
                counts_df.to_csv(counts_out, index=False)
            log(f"Wrote genus counts for {genus}")


def write_summary_downloads(
    latest_metrics: Dict[str, pd.DataFrame],
    latest_summaries: Dict[str, pd.DataFrame],
    latest_counts: Dict[str, Dict[str, int]],
    fallback_counts: Optional[pd.DataFrame],
    output_public: Path,
    dry_run: bool,
) -> None:
    """Emit merged CSVs for the summary page (preferred QC schemes only)."""
    summary_dir = output_public / "summary"

    def ensure_summary_dir() -> None:
        ensure_dir(summary_dir, dry_run)

    def write_df(df: pd.DataFrame, filename: str) -> None:
        if df is None or df.empty:
            return
        ensure_summary_dir()
        out_path = summary_dir / filename
        if dry_run:
            log(f"[dry-run] write summary CSV -> {out_path}", verbose=True)
            return
        ensure_dir(summary_dir, dry_run=False)
        df.to_csv(out_path, index=False)
        log(f"Wrote {filename} for summary page with {len(df)} rows")

    metrics_frames: List[pd.DataFrame] = []
    for species, df in latest_metrics.items():
        if df is None or df.empty:
            continue
        frame = df.copy()
        if "species" not in frame.columns:
            frame["species"] = species
        metrics_frames.append(frame)
    if metrics_frames:
        metrics_merged = pd.concat(metrics_frames, ignore_index=True)
        if "species" in metrics_merged.columns:
            metrics_merged.sort_values(["species", "metric"], inplace=True, ignore_index=True, na_position="last")
        write_df(metrics_merged, "filtered_metrics.csv")

    summary_frames: List[pd.DataFrame] = []
    for species, df in latest_summaries.items():
        if df is None or df.empty:
            continue
        frame = df.copy()
        if "species" not in frame.columns:
            frame["species"] = species
        summary_frames.append(frame)
    if summary_frames:
        summary_merged = pd.concat(summary_frames, ignore_index=True)
        if "species" in summary_merged.columns:
            summary_merged.sort_values(["species"], inplace=True, ignore_index=True)
        write_df(summary_merged, "summary_statistics.csv")

    counts_df: Optional[pd.DataFrame] = None
    if latest_counts:
        counts_df = pd.DataFrame(list(latest_counts.values()))
    elif fallback_counts is not None:
        counts_df = fallback_counts.copy()
    if counts_df is not None and not counts_df.empty:
        if "species" in counts_df.columns:
            counts_df.sort_values(["species"], inplace=True, ignore_index=True)
        write_df(counts_df, "species_counts.csv")


# ---------------------------------------------------------------------------
# main entry point

def package(
    calculate_dir: str,
    output_path: str,
    *,
    mark_latest: Optional[str] = None,
    version_name: Optional[str] = None,
    dry_run: bool = False,
    overwrite: bool = False,  # kept for API compatibility; not used aggressively
    verbose: bool = False,
) -> int:
    """
    Primary entry point used by qualibact-run.

    calculate_dir: QualiBact output directory containing species folders.
    output_path:   Destination base directory; outputs go under output_path/public and output_path/content.
    """
    input_dir = Path(calculate_dir)
    output_base = Path(output_path)
    output_public = output_base / "public"
    content_dir = output_base / "content"

    configure_logging(verbose)
    log(f"Packaging {input_dir} -> {output_public}", verbose)

    ensure_dir(output_base, dry_run)
    ensure_dir(output_public, dry_run)
    ensure_dir(content_dir, dry_run)

    preferred_versions = load_yaml(input_dir / "preferred_version.yml")
    filtered_metrics = load_filtered_metrics(input_dir)

    species_map = discover_species(input_dir, version_name)
    selected = select_species_dirs(species_map, preferred_versions)

    species_versions: Dict[str, List[str]] = {}
    computed_counts: List[Dict[str, int]] = []
    latest_metrics: Dict[str, pd.DataFrame] = {}
    latest_counts: Dict[str, Dict[str, int]] = {}
    latest_summaries: Dict[str, pd.DataFrame] = {}
    methods_dir = content_dir / "methods"
    ensure_dir(methods_dir, dry_run)

    latest_per_species: Dict[str, str] = {}
    for species, options in species_map.items():
        preferred_version = preferred_versions.get(species)
        versions = [v for _, v in options]
        if preferred_version and preferred_version in versions:
            latest_per_species[species] = preferred_version
        else:
            latest_per_species[species] = sorted(versions)[-1]

    for species, species_dir, version in selected:
        dest_dir = output_public / species / version
        ensure_dir(dest_dir, dry_run)
        species_versions.setdefault(species, []).append(version)
        log(f"Processing {species} version {version} from {species_dir}")

        thresholds = load_species_thresholds(species, filtered_metrics, species_dir / "summary.csv")

        metrics_path = species_dir / f"{species}_metrics.csv"
        metrics_df_current: Optional[pd.DataFrame] = None
        if metrics_path.exists():
            try:
                metrics_df_current = pd.read_csv(metrics_path)
                if "species" not in metrics_df_current.columns:
                    metrics_df_current["species"] = species
            except Exception as exc:
                log(f"Failed to read metrics for {species}: {exc}", level="warning")
        if metrics_df_current is None and metrics_path.exists():
            try:
                metrics_df_current = pd.read_csv(metrics_path)
                if "species" not in metrics_df_current.columns:
                    metrics_df_current["species"] = species
            except Exception as exc:
                log(f"Failed to read metrics for {species}: {exc}", level="warning")
        if version == latest_per_species.get(species):
            summary_path = species_dir / "summary.csv"
            if summary_path.exists():
                try:
                    summary_df = pd.read_csv(summary_path)
                    summary_df["species"] = species
                    latest_summaries[species] = summary_df
                except Exception as exc:
                    log(f"Failed to read summary for {species}: {exc}", level="warning")

        filtered_path = None
        highq_path = None
        filtered_count: Optional[int] = None
        highq_count: Optional[int] = None
        genome_counts = {"allthebacteria_count": None, "refseq_count": None}

        assembly_path = species_dir / f"{species}_assembly_stats.parquet"
        if assembly_path.exists():
            try:
                full_df, filtered_out, high_quality = apply_filters(
                    assembly_path, thresholds, verbose=verbose
                )
                filtered_count = len(filtered_out)
                highq_count = len(high_quality)
                if not filtered_out.empty:
                    filtered_dest = dest_dir / f"{species}_filtered_out_genomes.csv.xz"
                    write_xz_csv(filtered_out, filtered_dest, dry_run)
                    filtered_path = f"/{species}/{version}/{filtered_dest.name}"
                if not high_quality.empty:
                    highq_dest = dest_dir / f"{species}_high_quality_genomes.csv.xz"
                    write_xz_csv(high_quality, highq_dest, dry_run)
                    highq_path = f"/{species}/{version}/{highq_dest.name}"
                genome_counts["allthebacteria_count"] = len(full_df)
                computed_counts.append(
                    {
                        "species": species,
                        "original_count": len(full_df),
                        "filtered_out_count": len(filtered_out),
                        "final_count": len(high_quality),
                    }
                )
                if version == latest_per_species.get(species):
                    latest_counts[species] = {
                        "species": species,
                        "original_count": len(full_df),
                        "filtered_out_count": len(filtered_out),
                        "final_count": len(high_quality),
                    }
            except Exception as exc:
                log(f"Failed filtering for {species}: {exc}", verbose)
        else:
            log(f"No assembly parquet found for {species} in {species_dir}", verbose)
            if version == latest_per_species.get(species):
                latest_counts.setdefault(
                    species,
                    {"species": species, "original_count": None, "filtered_out_count": None, "final_count": None},
                )

        metrics_csv = species_dir / f"{species}_metrics.csv"
        if not metrics_csv.exists():
            log(f"Missing metrics CSV for {species} at {metrics_csv}", level="warning")
        elif version == latest_per_species.get(species) and metrics_df_current is not None:
            latest_metrics[species] = metrics_df_current

        refseq_path, refseq_count, refseq_updated = convert_refseq_json(
            species_dir, species, dest_dir, dry_run, verbose
        )
        if refseq_count is not None:
            genome_counts["refseq_count"] = refseq_count

        # Copy original assets after derived files are written
        copy_assets(species_dir, dest_dir, dry_run, verbose)

        summary_exists = (species_dir / "summary.csv").exists()
        selected_summary_exists = (species_dir / "selected_summary.csv").exists()

        write_manifest(
            species,
            version,
            dest_dir,
            is_latest=version == latest_per_species.get(species),
            genome_counts=genome_counts,
            refseq_path=refseq_path,
            filtered_path=filtered_path,
            highq_path=highq_path,
            summary_exists=summary_exists,
            selected_summary_exists=selected_summary_exists,
            verbose=verbose,
            dry_run=dry_run,
        )
        write_species_content(
            species,
            version,
            genome_counts,
            content_dir,
            dry_run,
            overwrite,
            filtered_out_count=filtered_count,
            high_quality_count=highq_count,
            metrics_exists=metrics_path.exists(),
            filtered_path=filtered_path,
            highq_path=highq_path,
        )
        # ensure per-scheme methods stub exists
        method_candidates = [
            methods_dir / f"{version}.mdx",
            methods_dir / f"{version}.md",
        ]
        existing_method = next((p for p in method_candidates if p.exists()), None)
        target_method = existing_method or method_candidates[0]
        if overwrite or existing_method is None:
            if not dry_run:
                ensure_dir(target_method.parent, dry_run=False)
                if overwrite or not target_method.exists():
                    with open(target_method, "w", encoding="utf-8") as handle:
                        handle.write(f"{version} methods\n\n_Todo: add methods for {version}._\n")

    # species_counts: prefer existing aggregated file, else use computed counts
    species_counts_df = None
    existing_counts = input_dir / "all_summary" / "species_counts.csv"
    if existing_counts.exists():
        log(f"Using existing species_counts.csv from {existing_counts}")
        species_counts_df = pd.read_csv(existing_counts)
    elif computed_counts:
        log(f"Using computed species counts for {len(computed_counts)} species")
        species_counts_df = pd.DataFrame(computed_counts)

    write_site_index(species_versions, output_public, mark_latest, dry_run, overwrite)
    write_genus_summaries(latest_metrics, latest_summaries, latest_counts, output_public, dry_run)
    write_summary_downloads(
        latest_metrics,
        latest_summaries,
        latest_counts,
        species_counts_df,
        output_public,
        dry_run,
    )

    log("Packaging run complete", verbose)
    return 0


__all__ = ["package"]
