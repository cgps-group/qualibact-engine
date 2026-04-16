# This method will prepare the dataframes required for the qualibact module
import os
import logging
from qualibact.prepare_util import get_df, load_generic_csv, filter_assembly_data
from qualibact.refseq import run_datasets_summary
from tqdm import tqdm


def _write_species_files(workdir, assembly_filtered_df, submit=False):
    """Shared logic: split a filtered dataframe by species, write parquets + SLURM scripts."""
    species_list = assembly_filtered_df["species_sylph"].unique().tolist()
    for species in tqdm(species_list, desc="Processing species"):
        safe_species = species.replace(" ", "_").replace("/", "_")
        species_dir = os.path.join(workdir, safe_species)
        os.makedirs(species_dir, exist_ok=True)

        species_df = assembly_filtered_df[
            assembly_filtered_df["species_sylph"] == species
        ].copy()
        species_df.to_parquet(
            os.path.join(species_dir, f"{safe_species}_assembly_stats.parquet"),
            index=False,
        )

        run_datasets_summary(
            taxon=species,
            outdir=species_dir,
            assembly_source="RefSeq",
            assembly_level="complete",
        )

        batch_script_path = os.path.join(
            species_dir, f"run_qualibact_{safe_species}.sh"
        )
        python_path = os.path.abspath(os.environ.get("QUALIBACT_PYTHONPATH", "python3"))
        with open(batch_script_path, "w") as f:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_path = os.path.join(parent_dir, "qualibact-run.py")
            abs_species_dir = os.path.abspath(species_dir)
            f.write(f"""#!/bin/bash
#SBATCH --job-name=qualibact{safe_species}
#SBATCH --output={abs_species_dir}/qualibact_{safe_species}.out
#SBATCH --error={abs_species_dir}/qualibact_{safe_species}.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1

{python_path} {script_path} calculate "{species}" {abs_species_dir}
""")
        os.chmod(batch_script_path, 0o755)
        if submit:
            os.system(f"sbatch {batch_script_path}")


def prepare_from_csv(workdir, input_csv, min_genome_count=100, submit=False):
    """
    Generic mode: prepare from a user-supplied CSV/TSV of assembly statistics.

    The input file must contain at minimum a sample identifier column and a species
    column. See prepare_util.load_generic_csv for recognised column names.
    """
    os.makedirs(workdir, exist_ok=True)
    df = load_generic_csv(input_csv)
    logging.info("Loaded %d rows from %s", len(df), input_csv)
    _, assembly_filtered_df = filter_assembly_data(df, min_genome_count=min_genome_count)
    logging.info("Retained %d rows after filtering (min %d genomes per species)",
                 len(assembly_filtered_df), min_genome_count)
    _write_species_files(workdir, assembly_filtered_df, submit=submit)


def prepare_qualibact_dataframes(workdir, metadata_path, submit=False):
    """
    ATB mode: prepare from AllTheBacteria metadata directory.

    Expects assembly-stats.tsv.gz, checkm2.tsv.gz, sylph.tsv.gz and
    file_list.all.latest.tsv.gz inside metadata_path.
    """
    pq_table_path = os.path.join(workdir, "assembly_stats.parquet")
    os.makedirs(workdir, exist_ok=True)
    assembly_full_df, assembly_filtered_df = get_df(
        pq_table_path=pq_table_path,
        workdir=workdir,
        metadata_path=metadata_path,
        min_genome_count=100,
    )
    logging.info("Dataframe loaded with %d rows", len(assembly_filtered_df))
    logging.info("Ignoring full dataframe with %d rows", len(assembly_full_df))
    _write_species_files(workdir, assembly_filtered_df, submit=submit)
