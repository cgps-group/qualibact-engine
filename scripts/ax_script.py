"""Helper script to build a QualiBact-ready parquet from local AX data."""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from rich.console import Console

# Ensure the repo root (containing the `qualibact` package) is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qualibact.species_run import species_run  # noqa: E402

console = Console()


def first_existing(paths: Iterable[Path]) -> Path:
    """
    Return the first existing path from a list or raise a helpful error.

    Args:
        paths: Candidate paths to check in order.
    """
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of the paths exist: {', '.join(str(p) for p in paths)}"
    )


def read_tsv(path: Path) -> pd.DataFrame:
    """
    Read a TSV or TSV.GZ into a dataframe.

    Uses gzip when the suffix is .gz and otherwise lets pandas infer.
    """
    compression = "gzip" if path.suffix == ".gz" else "infer"
    return pd.read_csv(path, sep="\t", compression=compression)


def create_quast_df(quast_dir: Path) -> pd.DataFrame:
    """
    Read QUAST reports and return one row per report.

    Each report is key/value tab-separated lines; numeric fields are coerced
    to numbers and rows are deduplicated. Adds `assembly_accession` from the
    Assembly field or filename.
    """
    if not quast_dir.exists():
        raise FileNotFoundError(f"QUAST directory not found: {quast_dir}")
    rows: List[dict] = []
    for report_path in sorted(quast_dir.glob("*.tsv*")):
        opener = gzip.open if report_path.suffix == ".gz" else open
        with opener(report_path, "rt") as f:
            record = {}
            for raw_line in f:
                line = raw_line.strip()
                if not line or "\t" not in line:
                    continue
                key, value = line.split("\t", 1)
                record[key.strip()] = value.strip()
        if not record:
            continue
        stem = report_path.stem
        if stem.endswith(".tsv"):
            stem = stem[:-4]
        assembly_accession = record.get("Assembly") or stem.replace("_report", "")
        # Normalize to match checkm/metadata ids (strip trailing _filtered and whitespace)
        assembly_accession = assembly_accession.strip().replace("_filtered", "")
        record["assembly_accession"] = assembly_accession
        rows.append(record)
    quast_df = pd.DataFrame(rows).drop_duplicates()
    for col in quast_df.columns:
        if col == "assembly_accession":
            continue
        quast_df[col] = pd.to_numeric(quast_df[col], errors="coerce")
    return quast_df


def create_ax_dataframe(data_dir: Path) -> pd.DataFrame:
    """
    Merge CheckM2, metadata, and nucleotide composition tables.

    Normalizes assembly_accession across sources, strips whitespace, coerces
    numerics where possible, and drops duplicate accessions.
    """
    checkm_path = first_existing(
        [data_dir / "checkm2_results_ax.tsv.gz", data_dir / "checkm2_results_ax.tsv"]
    )
    metadata_path = first_existing(
        [data_dir / "metadata_ax.tsv.gz", data_dir / "metadata_ax.tsv"]
    )
    nuc_path = first_existing(
        [
            data_dir / "nucleotide_composition_ax.tsv.gz",
            data_dir / "nucleotide_composition_ax.tsv",
        ]
    )

    checkm_df = read_tsv(checkm_path)
    metadata_df = read_tsv(metadata_path)
    nuc_df = read_tsv(nuc_path)

    checkm_df["assembly_accession"] = (
        checkm_df["Name"]
        .astype(str)
        .str.replace("_filtered$", "", regex=True)
        .str.strip()
    )
    metadata_df["assembly_accession"] = metadata_df["filename"].astype(str).str.strip()
    nuc_df["assembly_accession"] = nuc_df["Filename"].astype(str).str.strip()

    merged = (
        checkm_df.merge(
            metadata_df,
            on="assembly_accession",
            how="inner",
            suffixes=("_checkm", "_metadata"),
        )
        .merge(nuc_df, on="assembly_accession", how="inner", suffixes=("", "_nuc"))
        .drop_duplicates(subset=["assembly_accession"])
    )
    text_cols = {
        "assembly_accession",
        "Name",
        "filename",
        "Filename",
        "platform",
        "instrument",
        "species",
        "accession",
        "software",
        "version",
        "custom_parameters",
        "Notes",
        "Additional_Notes",
        "Completeness_Model_Used",
        "Translation_Table_Used",
    }
    for col in merged.columns:
        if merged[col].dtype == object:
            merged[col] = merged[col].astype(str).str.strip()
        if col not in text_cols:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def build_parquet(
    data_dir: Path,
    quast_dir: Path,
    species: str,
    parquet_path: Path,
    run_species: bool,
) -> Path:
    """
    Build merged parquet and optionally run QualiBact for the species.

    Merges AX tables with QUAST reports, normalizes key columns expected by
    qualibact, writes parquet, and calls `species_run` unless skipped.
    """
    console.print(f"[bold cyan]Reading AX tables from[/bold cyan] {data_dir}")
    ax_df = create_ax_dataframe(data_dir)
    console.print(
        f"[bold cyan]Loaded[/bold cyan] {len(ax_df)} records from CheckM/metadata/nucleotide tables"
    )

    console.print(f"[bold cyan]Reading QUAST reports from[/bold cyan] {quast_dir}")
    quast_df = create_quast_df(quast_dir)
    console.print(f"[bold cyan]Loaded[/bold cyan] {len(quast_df)} QUAST reports")

    merged_df = ax_df.merge(
        quast_df, on="assembly_accession", how="left", suffixes=("", "_quast")
    )
    # Normalize columns to what qualibact expects and coalesce quast variants
    rename_map = {
        "Total length": "total_length",
        "Total length (>= 0 bp)": "total_length",
        "# contigs": "number",
        "# contigs (>= 0 bp)": "number",
        "Largest contig": "longest",
        "Contig_N50": "N50",
        "Completeness": "Completeness_Specific",
    }
    merged_df = merged_df.rename(columns=rename_map)
    # Coalesce renamed quast columns into base columns
    for original, target in rename_map.items():
        quast_col = f"{original}_quast"
        if quast_col in merged_df.columns:
            merged_df[target] = merged_df[target].fillna(merged_df[quast_col])
            merged_df = merged_df.drop(columns=[quast_col])
    if "GC (%)" in merged_df.columns and "GC_Content" not in merged_df.columns:
        merged_df["GC_Content"] = (
            pd.to_numeric(merged_df["GC (%)"], errors="coerce") / 100.0
        )
    if "N50" not in merged_df.columns and "Contig_N50" in merged_df.columns:
        merged_df["N50"] = merged_df["Contig_N50"]
    # Drop any remaining duplicate-named columns (keep first occurrence)
    duplicated_cols = merged_df.columns[merged_df.columns.duplicated()].unique()
    if len(duplicated_cols) > 0:
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    console.print(f"[bold cyan]Merged dataset rows[/bold cyan]: {len(merged_df)}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(parquet_path, index=False)
    csv_path = parquet_path.with_suffix(".csv")
    merged_df.to_csv(csv_path, index=False)
    console.print(f"[bold green]Wrote parquet[/bold green]: {parquet_path}")
    console.print(f"[bold green]Wrote CSV[/bold green]: {csv_path}")

    if run_species:
        console.print(
            f"[bold cyan]Running QualiBact species_run for[/bold cyan] {species}"
        )
        species_run(species=species, species_dir=str(data_dir))
    return parquet_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the AX parquet builder."""
    parser = argparse.ArgumentParser(
        description="Build QualiBact parquet from Achromobacter local data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("AX_data"),
        help="Directory containing AX TSV inputs.",
    )
    parser.add_argument(
        "--quast-dir",
        type=Path,
        default=Path("AX_data/assembly_stats_ax"),
        help="Directory with QUAST report TSV/TSV.GZ files.",
    )
    parser.add_argument(
        "--species",
        default="Achromobacter xylosoxidans",
        help="Species name to pass to species_run.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("AX_data") / "Achromobacter_xylosoxidans_assembly_stats.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Build parquet but skip calling species_run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_parquet(
        data_dir=args.data_dir,
        quast_dir=args.quast_dir,
        species=args.species,
        parquet_path=args.output_parquet,
        run_species=not args.skip_run,
    )


if __name__ == "__main__":
    main()
