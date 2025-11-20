#!/usr/bin/env python3
import typer
from typing import Optional
import os
import logging
from dotenv import load_dotenv
from qualibact.prepare import prepare_qualibact_dataframes
from qualibact.species_run import species_run
from qualibact.summary import summary as final_summary

app = typer.Typer()

def load_environment(dotenv_path: str = ".qualibact.env"):
    """Load the environment variables and configurations."""
    if os.path.exists(dotenv_path):
        return load_dotenv(dotenv_path)
    else:
        raise FileNotFoundError(f"Environment file {dotenv_path} not found.")

@app.command()
def prepare(dotenv_path: str = ".qualibact.env", submit: bool = False):
    """Run the ATB fetch script."""
    load_environment(dotenv_path)
    logging.info("Environment loaded and logging set up.")
    typer.echo("Preparing qualibact dataframes...")
    metadata_path = os.environ["GENOMEQC_ATB_METADATA"] 
    GENOMEQC_OUTPUT_DIR = os.environ["GENOMEQC_OUTPUT_DIR"]
    prepare_qualibact_dataframes(metadata_path=metadata_path, workdir=GENOMEQC_OUTPUT_DIR, submit=submit)


@app.command()
def calculate(species: str,
              workdir: str,
              dotenv_path: str = ".qualibact.env", ):
    """calculate QualiBact for a specific species."""
    load_environment(dotenv_path)
    typer.echo("Running QualiBact for species...")
    species_run(species, workdir)

@app.command()
def summary(
    calculate_dir = typer.Argument(..., help="Path to the species directory containing qualibact results."),
    dotenv_path: str = ".qualibact.env"
):
    """Perform final summary of qualibact results."""
    load_environment(dotenv_path)
    typer.echo("Final summary...")
    final_summary(calculate_dir)

@app.command()
def make_docs(
    calculate_dir = typer.Argument(..., help="Path to the species directory containing qualibact results."),
    docs_dir: str = typer.Argument('docs', help="Path to the directory where documentation will be generated."),
    dotenv_path: str = ".qualibact.env"
):
    """Generate documentation for qualibact results."""
    load_environment(dotenv_path)
    from qualibact.docs import generate_docs
    typer.echo("Generating documentation...")
    generate_docs(calculate_dir, docs_dir)

@app.command()
def package(
    calculate_dir = typer.Argument(..., help="Path to the species directory containing qualibact results."),
    output_path: str = typer.Argument('qualibact_package', help="Path to the output package directory."),
    mark_latest: Optional[str] = typer.Option(None, help="Mark specified version as latest (true for current, or explicit version name)"),
    overwrite: bool = typer.Option(False, help="Overwrite existing mdx files in the destination"),
    dry_run: bool = typer.Option(False, help='Print what would happen; do not write any files'),
    verbose: bool = typer.Option(False, help='Enable verbose logging'),
    dotenv_path: str = ".qualibact.env"):
    """Package qualibact results into a zip file."""
    import qualibact.pack as pack
    load_environment(dotenv_path)
    typer.echo("Improved summary...")
    # Call programmatic wrapper so that CLI-layer is in qualibact-run; pack package() does the real work
    pack.package(calculate_dir, output_path, mark_latest=mark_latest, overwrite=overwrite, dry_run=dry_run, verbose=verbose)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    app()
