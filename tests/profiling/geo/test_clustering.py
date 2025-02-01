"""Tests for the location clustering module."""

import numpy as np
import pytest

from profiling.core.analyzer import TasteProfileAnalyzer
from profiling.geo.clustering import LocationClusterer

# Sample data for testing
SAMPLE_LOCATIONS = [
    ("business_1", 37.7749, -122.4194),  # San Francisco
    ("business_2", 37.7833, -122.4167),  # 1km away
    ("business_3", 37.7855, -122.4130),  # 1.5km away
    ("business_4", 37.8000, -122.4000),  # 3km away
    ("business_5", 38.0000, -122.0000),  # Far away
]

SAMPLE_REVIEWS = {
    "business_1": [
        "Amazing Italian food with authentic flavors.",
        "Great pasta and excellent service.",
    ],
    "business_2": ["Authentic Italian cuisine, but pricey.", "Delicious pasta dishes."],
    "business_3": ["Best Mexican food in the area!", "Spicy and flavorful dishes."],
    "business_4": [
        "Decent Italian food but nothing special.",
        "Average service and atmosphere.",
    ],
    "business_5": [
        "Good Chinese food and friendly staff.",
        "Nice ambiance but small portions.",
    ],
}


@pytest.fixture
def clusterer():
    """Create a LocationClusterer instance with sample data."""
    analyzer = TasteProfileAnalyzer()
    clusterer = LocationClusterer(analyzer=analyzer)

    # Add restaurants
    for business_id, lat, lon in SAMPLE_LOCATIONS:
        reviews = SAMPLE_REVIEWS[business_id]
        clusterer.add_restaurant(business_id, lat, lon, reviews)

    return clusterer


def test_cluster_by_location(clusterer):
    """Test clustering based on location only."""
    clusters = clusterer.cluster_by_location(max_distance=2.0, min_samples=2)  # 2km

    # Should find at least one cluster
    assert len(clusters) > 0

    # First three businesses should be in same cluster
    cluster_with_business_1 = next(
        label for label, businesses in clusters.items() if "business_1" in businesses
    )
    assert "business_2" in clusters[cluster_with_business_1]
    assert "business_3" in clusters[cluster_with_business_1]

    # Business 5 should be noise
    noise_points = next(
        businesses for label, businesses in clusters.items() if label == "cluster_-1"
    )
    assert "business_5" in noise_points


def test_cluster_by_profile(clusterer):
    """Test clustering based on taste profiles."""
    clusters = clusterer.cluster_by_profile(min_confidence=0.3)

    # Should find clusters based on cuisine type
    assert len(clusters) > 0

    # Find Italian restaurants cluster
    italian_cluster = None
    for label, businesses in clusters.items():
        if "business_1" in businesses:
            italian_cluster = businesses
            break

    assert italian_cluster is not None
    assert "business_2" in italian_cluster  # Similar Italian restaurant
    assert "business_3" not in italian_cluster  # Mexican restaurant


def test_cluster_combined(clusterer):
    """Test combined location and profile clustering."""
    clusters = clusterer.cluster_combined(
        max_distance=2.0, min_samples=2, location_weight=0.5
    )

    # Should find clusters
    assert len(clusters) > 0

    # Businesses 1 and 2 should be together (close + similar cuisine)
    cluster_with_business_1 = next(
        label for label, businesses in clusters.items() if "business_1" in businesses
    )
    assert "business_2" in clusters[cluster_with_business_1]

    # Business 3 might be separate (different cuisine)
    assert "business_3" not in clusters[cluster_with_business_1]

    # Business 5 should be noise (far away)
    noise_points = next(
        businesses for label, businesses in clusters.items() if label == "cluster_-1"
    )
    assert "business_5" in noise_points


def test_analyze_cluster(clusterer):
    """Test cluster analysis functionality."""
    # Create a cluster
    cluster_businesses = ["business_1", "business_2"]
    analysis = clusterer.analyze_cluster(cluster_businesses)

    # Check analysis results
    assert "center" in analysis
    assert "radius" in analysis
    assert "dominant_aspects" in analysis
    assert "price_distribution" in analysis
    assert "dining_styles" in analysis

    # Check center coordinates
    center_lat, center_lon = analysis["center"]
    assert 37.7 < center_lat < 37.8
    assert -122.5 < center_lon < -122.4

    # Check radius
    assert 0 < analysis["radius"] < 2.0  # Should be less than 2km

    # Check aspects
    assert len(analysis["dominant_aspects"]) > 0

    # Check distributions
    assert len(analysis["price_distribution"]) > 0
    assert len(analysis["dining_styles"]) > 0


def test_generate_map(clusterer):
    """Test map generation functionality."""
    # Generate clusters
    clusters = clusterer.cluster_combined(
        max_distance=2.0, min_samples=2, location_weight=0.5
    )

    # Generate map
    map_obj = clusterer.generate_map(clusters)

    # Basic checks on the map object
    assert hasattr(map_obj, "_name")
    assert hasattr(map_obj, "save")

    # Check that the map contains markers
    assert len(map_obj._children) > 0


def test_edge_cases(clusterer):
    """Test edge cases and error handling."""
    # Empty cluster
    empty_analysis = clusterer.analyze_cluster([])
    assert empty_analysis == {}

    # Single point cluster
    single_analysis = clusterer.analyze_cluster(["business_1"])
    assert "center" in single_analysis
    assert single_analysis["radius"] == 0

    # Invalid business ID
    invalid_analysis = clusterer.analyze_cluster(["invalid_id"])
    assert invalid_analysis == {}

    # All points as noise
    far_clusters = clusterer.cluster_by_location(
        max_distance=0.1, min_samples=5  # Very small distance  # More than total points
    )
    assert all(label == "cluster_-1" for label in far_clusters.keys())
