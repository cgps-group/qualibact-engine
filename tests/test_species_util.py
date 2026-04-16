"""Unit and regression tests for species_util: outlier filtering and metric stats."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from qualibact.species_util import apply_outlier_filter, make_metric_stats

FIXTURES = Path(__file__).parent / "fixtures"


def _make_species_df(n=200, seed=42):
    """Synthetic assembly stats dataframe for a single species."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "N50": rng.normal(80000, 10000, n).clip(10000),
        "total_length": rng.normal(5_000_000, 200_000, n).clip(1_000_000),
        "GC_Content": rng.normal(0.508, 0.005, n).clip(0.3, 0.8),
        "number": rng.integers(30, 100, n),
        "Completeness_Specific": rng.uniform(95, 100, n),
        "Contamination": rng.uniform(0, 2, n),
        "Total_Coding_Sequences": rng.normal(4900, 200, n).clip(1000),
        "longest": rng.normal(200_000, 50_000, n).clip(10_000),
    })


class TestApplyOutlierFilter:
    def test_returns_anomaly_column(self):
        df = _make_species_df()
        result = apply_outlier_filter(df)
        assert "anomaly" in result.columns

    def test_anomaly_values_are_1_or_minus1(self):
        df = _make_species_df()
        result = apply_outlier_filter(df)
        assert set(result["anomaly"].unique()).issubset({1, -1})

    def test_majority_labelled_inlier(self):
        df = _make_species_df(n=500)
        result = apply_outlier_filter(df)
        inlier_frac = (result["anomaly"] == 1).mean()
        assert inlier_frac > 0.7, f"Expected >70% inliers, got {inlier_frac:.1%}"

    def test_obvious_outliers_flagged(self):
        df = _make_species_df(n=300)
        # Add 5 obvious outliers with very small genomes (use total_length, not Genome_Size)
        outliers = _make_species_df(n=5, seed=99)
        outliers["total_length"] = 100_000  # far from the mean of 5M
        df = pd.concat([df, outliers], ignore_index=True)
        result = apply_outlier_filter(df)
        # At least 1 of the 5 outlier rows should be flagged
        flagged = result.iloc[-5:]["anomaly"] == -1
        assert flagged.sum() >= 1


class TestMakeMetricStats:
    def test_returns_expected_keys(self):
        df = _make_species_df()
        stats = make_metric_stats("N50", df)
        for key in ("metric", "median", "q1", "q3", "min", "max", "upper_bound", "lower_bound"):
            assert key in stats, f"Missing key: {key}"

    def test_median_within_range(self):
        df = _make_species_df()
        stats = make_metric_stats("N50", df)
        assert stats["min"] <= stats["median"] <= stats["max"]

    def test_bounds_ordered(self):
        df = _make_species_df()
        stats = make_metric_stats("N50", df)
        assert stats["lower_bound"] <= stats["upper_bound"]

    def test_regression_ecoli_n50(self):
        """Regression: metric stats for N50 on a fixed synthetic E. coli dataset."""
        df = _make_species_df(n=1000, seed=0)
        stats = make_metric_stats("N50", df)
        # Values computed from the same seed — update if method changes intentionally
        assert 75_000 < stats["median"] < 85_000, f"Median N50 out of expected range: {stats['median']}"
        assert stats["lower_bound"] > 0
