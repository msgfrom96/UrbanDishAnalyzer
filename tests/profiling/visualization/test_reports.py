"""Tests for the report generation module."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from profiling.core.profile import TasteProfile
from profiling.visualization.reports import ReportGenerator


# Sample data for testing
@pytest.fixture
def profiles():
    """Create sample taste profiles."""
    profiles = {}

    # High-end Italian restaurant
    p1 = TasteProfile("business_1")
    p1.price_category = "$$$"
    p1.dining_style = "fine_dining"
    p1.taste_quality = 0.9
    p1.service_quality = 0.85
    p1.ambiance = 0.8
    p1.value_ratio = 0.7
    p1.authenticity = 0.9
    p1.portion_size = 0.6
    p1.confidence_scores = {
        "taste_quality": 0.9,
        "service_quality": 0.85,
        "ambiance": 0.8,
        "value_ratio": 0.7,
        "authenticity": 0.9,
        "portion_size": 0.6,
    }
    profiles["business_1"] = p1

    # Casual Mexican restaurant
    p2 = TasteProfile("business_2")
    p2.price_category = "$$"
    p2.dining_style = "casual"
    p2.taste_quality = 0.8
    p2.service_quality = 0.75
    p2.ambiance = 0.7
    p2.value_ratio = 0.8
    p2.authenticity = 0.85
    p2.portion_size = 0.9
    p2.confidence_scores = {
        "taste_quality": 0.8,
        "service_quality": 0.75,
        "ambiance": 0.7,
        "value_ratio": 0.8,
        "authenticity": 0.85,
        "portion_size": 0.9,
    }
    profiles["business_2"] = p2

    return profiles


@pytest.fixture
def clusters():
    """Create sample cluster data."""
    return {"cluster_1": ["business_1"], "cluster_2": ["business_2"]}


@pytest.fixture
def report_gen(tmp_path):
    """Create a ReportGenerator instance."""
    return ReportGenerator(str(tmp_path))


def test_generate_cluster_report(report_gen, profiles, clusters, tmp_path):
    """Test cluster report generation."""
    report_gen.generate_cluster_report(clusters, profiles, "cluster_report.html")

    report_file = tmp_path / "cluster_report.html"
    assert report_file.exists()

    # Check report content
    content = report_file.read_text()
    assert "Cluster Analysis Report" in content
    assert "cluster_1" in content
    assert "cluster_2" in content
    assert "Price Distribution" in content
    assert "Dining Styles" in content
    assert "Dominant Aspects" in content


def test_plot_aspect_distributions(report_gen, profiles, tmp_path):
    """Test aspect distribution plotting."""
    aspects = ["taste_quality", "service_quality", "ambiance", "value_ratio"]

    report_gen.plot_aspect_distributions(profiles, aspects, "aspect_distributions.png")

    plot_file = tmp_path / "aspect_distributions.png"
    assert plot_file.exists()

    # Clean up plot
    plt.close()


def test_plot_price_analysis(report_gen, profiles, tmp_path):
    """Test price analysis plotting."""
    report_gen.plot_price_analysis(profiles, "price_analysis.png")

    plot_file = tmp_path / "price_analysis.png"
    assert plot_file.exists()

    # Clean up plot
    plt.close()


def test_plot_correlation_matrix(report_gen, profiles, tmp_path):
    """Test correlation matrix plotting."""
    aspects = [
        "taste_quality",
        "service_quality",
        "ambiance",
        "value_ratio",
        "authenticity",
        "portion_size",
    ]

    report_gen.plot_correlation_matrix(profiles, aspects, "correlation_matrix.png")

    plot_file = tmp_path / "correlation_matrix.png"
    assert plot_file.exists()

    # Clean up plot
    plt.close()


def test_export_summary_stats(report_gen, profiles, tmp_path):
    """Test summary statistics export."""
    report_gen.export_summary_stats(profiles, "summary_stats.csv")

    stats_file = tmp_path / "summary_stats.csv"
    assert stats_file.exists()

    # Check stats content
    df = pd.read_csv(stats_file)
    assert not df.empty
    assert "count" in df.columns
    assert "mean" in df.columns
    assert "std" in df.columns
    assert "min" in df.columns
    assert "max" in df.columns
    assert "confidence" in df.columns


def test_empty_data_handling(report_gen, tmp_path):
    """Test handling of empty data."""
    empty_profiles = {}
    empty_clusters = {}

    # Cluster report with empty data
    report_gen.generate_cluster_report(
        empty_clusters, empty_profiles, "empty_cluster_report.html"
    )
    empty_report = tmp_path / "empty_cluster_report.html"
    assert empty_report.exists()

    # Aspect distributions with empty data
    report_gen.plot_aspect_distributions(
        empty_profiles, ["taste_quality"], "empty_aspects.png"
    )
    empty_plot = tmp_path / "empty_aspects.png"
    assert empty_plot.exists()

    # Clean up plot
    plt.close()


def test_invalid_data_handling(report_gen, profiles, tmp_path):
    """Test handling of invalid data."""
    # Invalid aspects
    report_gen.plot_aspect_distributions(
        profiles, ["invalid_aspect"], "invalid_aspects.png"
    )
    invalid_plot = tmp_path / "invalid_aspects.png"
    assert invalid_plot.exists()

    # Invalid clusters
    invalid_clusters = {"cluster_1": ["invalid_business"]}
    report_gen.generate_cluster_report(
        invalid_clusters, profiles, "invalid_cluster_report.html"
    )
    invalid_report = tmp_path / "invalid_cluster_report.html"
    assert invalid_report.exists()

    # Clean up plot
    plt.close()


def test_style_customization(report_gen, profiles, tmp_path):
    """Test plot style customization."""
    # Custom style for aspect distributions
    plt.style.use("seaborn-darkgrid")
    report_gen.plot_aspect_distributions(
        profiles, ["taste_quality"], "custom_style.png"
    )
    style_plot = tmp_path / "custom_style.png"
    assert style_plot.exists()

    # Reset style
    plt.style.use("default")
    plt.close()
