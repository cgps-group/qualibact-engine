"""Unit tests for prepare_util: column normalisation and generic CSV loading."""
import pytest
import pandas as pd
import os
from pathlib import Path

from qualibact.prepare_util import normalize_columns, load_generic_csv, filter_assembly_data

FIXTURES = Path(__file__).parent / "fixtures"


class TestNormalizeColumns:
    def test_canonical_names_unchanged(self):
        df = pd.DataFrame(columns=["species_sylph", "sample", "N50", "Genome_Size"])
        result = normalize_columns(df)
        assert list(result.columns) == ["species_sylph", "sample", "N50", "Genome_Size"]

    def test_common_aliases_resolved(self):
        df = pd.DataFrame(columns=["sample_id", "species", "n50", "genome_size", "gc_content"])
        result = normalize_columns(df)
        assert "sample" in result.columns
        assert "species_sylph" in result.columns
        assert "N50" in result.columns
        assert "Genome_Size" in result.columns
        assert "GC_Content" in result.columns

    def test_case_insensitive(self):
        df = pd.DataFrame(columns=["SPECIES", "SAMPLE_ID", "GC"])
        result = normalize_columns(df)
        assert "species_sylph" in result.columns
        assert "sample" in result.columns
        assert "GC_Content" in result.columns

    def test_unknown_columns_preserved(self):
        df = pd.DataFrame(columns=["sample_id", "species", "custom_column"])
        result = normalize_columns(df)
        assert "custom_column" in result.columns


class TestLoadGenericCSV:
    def test_loads_sample_fixture(self):
        df = load_generic_csv(str(FIXTURES / "sample_assemblies.csv"))
        assert "sample" in df.columns
        assert "species_sylph" in df.columns
        assert len(df) == 10

    def test_metric_columns_present(self):
        df = load_generic_csv(str(FIXTURES / "sample_assemblies.csv"))
        for col in ("N50", "Genome_Size", "GC_Content", "Completeness_Specific", "Contamination"):
            assert col in df.columns, f"Expected column {col} not found"

    def test_missing_species_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("sample_id,N50\nA,50000\n")
        with pytest.raises(ValueError, match="species"):
            load_generic_csv(str(bad_csv))

    def test_missing_sample_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("species,N50\nEscherichia coli,50000\n")
        with pytest.raises(ValueError, match="sample"):
            load_generic_csv(str(bad_csv))

    def test_tsv_support(self, tmp_path):
        tsv = tmp_path / "assemblies.tsv"
        tsv.write_text("sample_id\tspecies\tN50\nS1\tEscherichia coli\t80000\n")
        df = load_generic_csv(str(tsv))
        assert len(df) == 1


class TestFilterAssemblyData:
    def test_filters_low_count_species(self):
        df = load_generic_csv(str(FIXTURES / "sample_assemblies.csv"))
        _, filtered = filter_assembly_data(df, min_genome_count=4)
        # Escherichia coli has 4 samples, Klebsiella 3, Staphylococcus 3
        # Only Escherichia coli should survive min_genome_count=4
        remaining = filtered["species_sylph"].unique().tolist()
        assert "Escherichia coli" in remaining
        assert "Klebsiella pneumoniae" not in remaining

    def test_no_min_count_keeps_all_species(self):
        df = load_generic_csv(str(FIXTURES / "sample_assemblies.csv"))
        _, filtered = filter_assembly_data(df, min_genome_count=None)
        assert len(filtered["species_sylph"].unique()) == 3

    def test_removes_nan_species(self):
        df = load_generic_csv(str(FIXTURES / "sample_assemblies.csv"))
        df.loc[0, "species_sylph"] = None
        _, filtered = filter_assembly_data(df, min_genome_count=None)
        assert filtered["species_sylph"].isna().sum() == 0
