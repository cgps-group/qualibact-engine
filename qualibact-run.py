#!/usr/bin/env python3
import typer
from typing import Optional
import os
import logging
from dotenv import load_dotenv
from qualibact.prepare import prepare_from_csv, prepare_qualibact_dataframes
from qualibact.species_run import species_run
from qualibact.summary import summary as final_summary

app = typer.Typer(help="qualibact-engine: species-specific QC thresholds for bacterial genome assemblies.")


def load_environment(dotenv_path: str = ".qualibact.env"):
    """Load environment variables from a .env file if present."""
    if os.path.exists(dotenv_path):
        return load_dotenv(dotenv_path)


@app.command()
def prepare(
    workdir: str = typer.Argument(..., help="Output working directory."),
    input_csv: Optional[str] = typer.Option(
        None, "--input", "-i",
        help="Path to assembly stats CSV/TSV. Required unless --atb-dir is given."
    ),
    atb_dir: Optional[str] = typer.Option(
        None, "--atb-dir",
        help="ATB mode: path to AllTheBacteria metadata directory "
             "(assembly-stats.tsv.gz, checkm2.tsv.gz, sylph.tsv.gz, file_list.all.latest.tsv.gz)."
    ),
    min_genome_count: int = typer.Option(100, help="Minimum genomes per species."),
    submit: bool = typer.Option(False, help="Submit SLURM batch scripts automatically."),
    dotenv_path: str = typer.Option(".qualibact.env", hidden=True),
):
    """
    Prepare per-species parquet files and SLURM batch scripts for threshold calculation.

    Generic mode (default): provide --input with a CSV/TSV of assembly statistics.
    ATB mode: provide --atb-dir pointing to an AllTheBacteria metadata directory.
    """
    load_environment(dotenv_path)
    if atb_dir:
        typer.echo(f"ATB mode: reading from {atb_dir}")
        prepare_qualibact_dataframes(workdir, atb_dir, submit=submit)
    elif input_csv:
        typer.echo(f"Generic mode: reading from {input_csv}")
        prepare_from_csv(workdir, input_csv, min_genome_count=min_genome_count, submit=submit)
    else:
        typer.echo("Error: provide --input <file.csv> or --atb-dir <path>.", err=True)
        raise typer.Exit(code=1)


@app.command()
def calculate(
    species: str = typer.Argument(..., help="Species name (e.g. 'Escherichia coli')."),
    workdir: str = typer.Argument(..., help="Species working directory containing the parquet file."),
    dotenv_path: str = typer.Option(".qualibact.env", hidden=True),
):
    """Calculate QC thresholds for a single species."""
    load_environment(dotenv_path)
    typer.echo(f"Running QualiBact for {species}...")
    species_run(species, workdir)


@app.command()
def summary(
    calculate_dir: str = typer.Argument(..., help="Directory of per-species calculate outputs."),
    dotenv_path: str = typer.Option(".qualibact.env", hidden=True),
):
    """Aggregate per-species threshold results into a combined summary."""
    load_environment(dotenv_path)
    typer.echo("Summarising results...")
    final_summary(calculate_dir)


@app.command()
def report(
    input_dir: str = typer.Argument(
        ".", help="Directory containing summary.csv and *_high_quality_genomes.csv from a calculate run."
    ),
    genomes_csv: Optional[str] = typer.Option(None, "--genomes-csv", help="Per-genome metrics CSV."),
    summary_csv: Optional[str] = typer.Option(None, "--summary-csv", help="Summary CSV for species name."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output CSV for per-genome QC results."),
    plots_dir: Optional[str] = typer.Option(None, "--plots-dir", help="Directory for QC plots."),
    batch: bool = typer.Option(False, help="Process all species subdirectories under input_dir."),
):
    """
    Generate a human-readable QC report from a calculate output directory.

    Produces per-genome pass/fail results, metric summaries, and distribution plots.
    Use --batch to process all species subdirectories at once.
    """
    import argparse
    from qualibact.assess import run_assessment

    ns = argparse.Namespace(
        input_dir=input_dir,
        genomes_csv=genomes_csv,
        summary_csv=summary_csv,
        qc_output=output,
        metric_output=None,
        plots_dir=plots_dir,
        batch=batch,
    )
    run_assessment(ns)


@app.command()
def make_docs(
    calculate_dir: str = typer.Argument(..., help="Directory of per-species calculate outputs."),
    docs_dir: str = typer.Argument("docs", help="Output directory for website documentation (MDX)."),
    dotenv_path: str = typer.Option(".qualibact.env", hidden=True),
):
    """
    [INTERNAL / WEBSITE] Generate MDX documentation for qualibact.org.

    This command is for the qualibact.org publishing pipeline, not for end-user analysis.
    """
    load_environment(dotenv_path)
    from qualibact.docs import generate_docs
    typer.echo("Generating documentation...")
    generate_docs(calculate_dir, docs_dir)


@app.command()
def package(
    calculate_dir: str = typer.Argument(..., help="Directory of per-species calculate outputs."),
    output_path: str = typer.Argument("qualibact_package", help="Output package directory."),
    mark_latest: Optional[str] = typer.Option(None, help="Mark this version as latest."),
    overwrite: bool = typer.Option(False, help="Overwrite existing MDX files."),
    force: bool = typer.Option(False, help="Overwrite existing static assets."),
    prefer: bool = typer.Option(False, help="Mark schemes as preferred in site_index."),
    dry_run: bool = typer.Option(False, help="Print actions without writing files."),
    verbose: bool = typer.Option(False, help="Enable verbose logging."),
    dotenv_path: str = typer.Option(".qualibact.env", hidden=True),
):
    """
    [INTERNAL / WEBSITE] Package results for deployment to qualibact.org.

    This command is for the qualibact.org publishing pipeline, not for end-user analysis.
    """
    import qualibact.pack as pack
    load_environment(dotenv_path)
    typer.echo("Packaging results...")
    pack.package(
        calculate_dir,
        output_path,
        mark_latest=mark_latest,
        prefer=prefer,
        overwrite=overwrite,
        force=force,
        dry_run=dry_run,
        verbose=verbose,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    app()
