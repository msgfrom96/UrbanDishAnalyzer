"""Tests for the hotspot detection module."""

import numpy as np
import pytest

from profiling.core.analyzer import TasteProfileAnalyzer
from profiling.geo.hotspots import HotspotDetector

# Sample data for testing
SAMPLE_LOCATIONS = [
    # Cluster 1: Italian restaurants
    (
        "italian_1",
        37.7749,
        -122.4194,
        [
            "Excellent authentic Italian cuisine.",
            "Amazing pasta and great wine selection.",
        ],
    ),
    (
        "italian_2",
        37.7755,
        -122.4190,
        ["Best Italian food in the area.", "Fantastic pasta dishes and service."],
    ),
    # Cluster 2: Asian restaurants
    (
        "asian_1",
        37.7920,
        -122.4100,
        ["Delicious sushi and great atmosphere.", "Fresh fish and excellent service."],
    ),
    (
        "asian_2",
        37.7925,
        -122.4095,
        ["Amazing Thai curry and friendly staff.", "Authentic Asian flavors."],
    ),
    (
        "asian_3",
        37.7930,
        -122.4090,
        ["Best Chinese food in town.", "Great dim sum selection."],
    ),
    # Scattered restaurants
    (
        "mexican_1",
        37.7800,
        -122.4150,
        ["Authentic Mexican tacos.", "Great margaritas and atmosphere."],
    ),
    (
        "burger_1",
        37.7850,
        -122.4000,
        ["Delicious burgers and fries.", "Casual atmosphere, good value."],
    ),
]


@pytest.fixture
def detector():
    """Create a HotspotDetector instance with sample data."""
    analyzer = TasteProfileAnalyzer()
    detector = HotspotDetector(
        min_samples=2, max_distance=0.5, analyzer=analyzer  # 500m
    )

    # Add restaurants
    for business_id, lat, lon, reviews in SAMPLE_LOCATIONS:
        detector.add_restaurant(business_id, lat, lon, reviews)

    return detector


def test_detect_hotspots(detector):
    """Test basic hotspot detection."""
    hotspots = detector.detect_hotspots()

    # Should find at least two hotspots
    assert len(hotspots) >= 2

    # Find Italian cluster
    italian_hotspot = next(h for h in hotspots if "italian_1" in h["businesses"])
    assert "italian_2" in italian_hotspot["businesses"]

    # Find Asian cluster
    asian_hotspot = next(h for h in hotspots if "asian_1" in h["businesses"])
    assert "asian_2" in asian_hotspot["businesses"]
    assert "asian_3" in asian_hotspot["businesses"]

    # Check hotspot properties
    for hotspot in hotspots:
        assert "center" in hotspot
        assert "radius" in hotspot
        assert "businesses" in hotspot
        assert "dominant_aspects" in hotspot
        assert "potential_gaps" in hotspot


def test_hotspot_characteristics(detector):
    """Test analysis of hotspot characteristics."""
    hotspots = detector.detect_hotspots()

    # Find Asian cluster
    asian_hotspot = next(h for h in hotspots if "asian_1" in h["businesses"])

    # Check center location
    center_lat, center_lon = asian_hotspot["center"]
    assert 37.79 < center_lat < 37.80
    assert -122.41 < center_lon < -122.40

    # Check radius (should be less than max_distance)
    assert asian_hotspot["radius"] < 0.5

    # Check dominant aspects
    assert any(
        "asian" in aspect.lower()
        or "sushi" in aspect.lower()
        or "thai" in aspect.lower()
        for aspect, _ in asian_hotspot["dominant_aspects"]
    )

    # Check potential gaps
    gaps = asian_hotspot["potential_gaps"]
    assert isinstance(gaps, list)
    assert len(gaps) > 0


def test_potential_opportunities(detector):
    """Test identification of potential opportunities."""
    hotspots = detector.detect_hotspots()

    for hotspot in hotspots:
        gaps = hotspot["potential_gaps"]

        # Should identify some opportunities
        assert len(gaps) > 0

        # Gaps should be reasonable
        for gap in gaps:
            assert isinstance(gap, str)
            assert len(gap) > 0

        # Should not suggest existing dominant cuisines
        dominant_aspects = [a for a, _ in hotspot["dominant_aspects"]]
        for gap in gaps:
            assert not any(aspect.lower() in gap.lower() for aspect in dominant_aspects)


def test_generate_map(detector):
    """Test map generation functionality."""
    hotspots = detector.detect_hotspots()
    map_obj = detector.generate_map(hotspots)

    # Basic map object checks
    assert hasattr(map_obj, "_name")
    assert hasattr(map_obj, "save")

    # Should have markers and circles
    assert len(map_obj._children) > 0

    # Should have at least one circle per hotspot
    circles = [
        child for child in map_obj._children.values() if "Circle" in str(type(child))
    ]
    assert len(circles) >= len(hotspots)


def test_edge_cases(detector):
    """Test edge cases and error handling."""
    # Empty detector
    empty_detector = HotspotDetector()
    assert empty_detector.detect_hotspots() == []

    # Single restaurant
    single_detector = HotspotDetector()
    single_detector.add_restaurant(
        "single_1", 37.7749, -122.4194, ["Good food and service"]
    )
    assert single_detector.detect_hotspots() == []

    # Very small max_distance
    small_detector = HotspotDetector(max_distance=0.01)
    for data in SAMPLE_LOCATIONS:
        small_detector.add_restaurant(*data)
    small_hotspots = small_detector.detect_hotspots()
    assert len(small_hotspots) == 0

    # Very large min_samples
    large_detector = HotspotDetector(min_samples=10)
    for data in SAMPLE_LOCATIONS:
        large_detector.add_restaurant(*data)
    large_hotspots = large_detector.detect_hotspots()
    assert len(large_hotspots) == 0
