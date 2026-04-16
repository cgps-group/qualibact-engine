# qualibact-engine

Species-specific quality control thresholds for bacterial genome assemblies.

Website: https://qualibact.org  
Used by: [SpecCheck](https://github.com/happykhan/speccheck)

## Installation

```bash
pixi install
```

Or with pip:

```bash
pip install -e .
```

`pixi.lock` is committed to the repository to ensure reproducible environments. If you add or update dependencies, run `pixi install` to regenerate it and commit the result.

## Quick start

### Generate a QC report from your own assembly stats

```bash
python qualibact-run.py report /path/to/species_output_dir
```

Or for multiple species at once:

```bash
python qualibact-run.py report /path/to/workdir --batch
```

Produces per-genome pass/fail results (`qc_genome_results.csv`), metric summaries, and distribution plots.

## Full threshold calculation workflow

Use this if you want to calculate thresholds from scratch for a new set of assemblies.

### 1. Prepare per-species data

**From your own assembly stats CSV/TSV:**

```bash
python qualibact-run.py prepare /path/to/workdir --input assemblies.csv
```

The input file must contain at minimum a sample identifier column and a species column. Recognised column names for species: `species`, `species_name`, `organism`, `taxon`. Recognised metric columns: `N50`, `Genome_Size`, `GC_Content`, `num_contigs`, `Completeness_Specific`, `Contamination`, `Total_Coding_Sequences`.

**From AllTheBacteria (ATB) metadata:**

```bash
python qualibact-run.py prepare /path/to/workdir --atb-dir /path/to/atb_metadata
```

The ATB metadata directory must contain: `assembly-stats.tsv.gz`, `checkm2.tsv.gz`, `sylph.tsv.gz`, `file_list.all.latest.tsv.gz`.

### 2. Calculate thresholds per species

Locally:

```bash
python qualibact-run.py calculate "Escherichia coli" /path/to/workdir/Escherichia_coli
```

On an HPC cluster: `prepare` generates SLURM batch scripts in each species directory. Submit with `--submit` or run manually.

### 3. Summarise results

```bash
python qualibact-run.py summary /path/to/workdir
```

## Commands

| Command | Description |
|---------|-------------|
| `prepare` | Prepare per-species parquet files from assembly stats (generic CSV or ATB) |
| `calculate` | Calculate QC thresholds for a single species |
| `summary` | Aggregate per-species results into a combined summary |
| `report` | Generate a human-readable QC report and plots |
| `make_docs` | **[Internal/website]** Generate MDX for qualibact.org |
| `package` | **[Internal/website]** Package results for qualibact.org deployment |

## Contributing

Issues and pull requests welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
