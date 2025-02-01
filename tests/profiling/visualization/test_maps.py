"""Tests for the map visualization module."""

from pathlib import Path

import folium
import pytest

from profiling.core.profile import TasteProfile
from profiling.visualization.maps import MapVisualizer

# Sample data for testing
SAMPLE_LOCATIONS = {
    "business_1": (37.7749, -122.4194),
    "business_2": (37.7833, -122.4167),
    "business_3": (37.7855, -122.4130),
}


@pytest.fixture
def profiles():
    """Create sample taste profiles."""
    profiles = {}

    # Business 1: High-end Italian
    p1 = TasteProfile("business_1")
    p1.price_category = "$$$"
    p1.dining_style = "fine_dining"
    p1.taste_quality = 0.9
    p1.service_quality = 0.85
    p1.confidence_scores = {"taste_quality": 0.9, "service_quality": 0.85}
    profiles["business_1"] = p1

    # Business 2: Casual Mexican
    p2 = TasteProfile("business_2")
    p2.price_category = "$$"
    p2.dining_style = "casual"
    p2.taste_quality = 0.8
    p2.service_quality = 0.75
    p2.confidence_scores = {"taste_quality": 0.8, "service_quality": 0.75}
    profiles["business_2"] = p2

    # Business 3: Budget-friendly Asian
    p3 = TasteProfile("business_3")
    p3.price_category = "$"
    p3.dining_style = "casual"
    p3.taste_quality = 0.7
    p3.service_quality = 0.8
    p3.confidence_scores = {"taste_quality": 0.7, "service_quality": 0.8}
    profiles["business_3"] = p3

    return profiles


@pytest.fixture
def map_viz():
    """Create a MapVisualizer instance."""
    return MapVisualizer(default_location=(37.7749, -122.4194), default_zoom=13)


def test_create_base_map(map_viz):
    """Test base map creation."""
    # Default settings
    m = map_viz.create_base_map()
    assert isinstance(m, folium.Map)
    assert m.location == [37.7749, -122.4194]
    assert m.zoom_start == 13

    # Custom settings
    custom_m = map_viz.create_base_map(center=(37.8, -122.4), zoom=15)
    assert custom_m.location == [37.8, -122.4]
    assert custom_m.zoom_start == 15


def test_add_restaurants(map_viz, profiles):
    """Test adding restaurant markers."""
    m = map_viz.create_base_map()
    m = map_viz.add_restaurants(m, SAMPLE_LOCATIONS, profiles)

    # Count markers
    markers = [
        child for child in m._children.values() if isinstance(child, folium.Marker)
    ]
    assert len(markers) == len(SAMPLE_LOCATIONS)

    # Check marker properties
    for marker in markers:
        assert marker.icon.color == "blue"
        assert marker.icon.icon == "info-sign"
        assert marker.popup is not None


def test_add_clusters(map_viz, profiles):
    """Test adding cluster visualization."""
    m = map_viz.create_base_map()

    clusters = {"cluster_1": ["business_1", "business_2"], "cluster_2": ["business_3"]}

    m = map_viz.add_clusters(m, clusters, SAMPLE_LOCATIONS, profiles)

    # Check circles for clusters
    circles = [
        child for child in m._children.values() if isinstance(child, folium.Circle)
    ]
    assert len(circles) == len(clusters)

    # Check markers for businesses
    markers = [
        child for child in m._children.values() if isinstance(child, folium.Marker)
    ]
    assert len(markers) == len(SAMPLE_LOCATIONS)


def test_add_heatmap(map_viz):
    """Test adding heatmap layer."""
    m = map_viz.create_base_map()

    # Basic heatmap
    m = map_viz.add_heatmap(m, SAMPLE_LOCATIONS)

    # Weighted heatmap
    weights = {"business_1": 1.0, "business_2": 0.8, "business_3": 0.6}
    m = map_viz.add_heatmap(m, SAMPLE_LOCATIONS, weights)

    # Check heatmap layers
    heatmaps = [
        child for child in m._children.values() if "HeatMap" in str(type(child))
    ]
    assert len(heatmaps) == 2


def test_add_choropleth(map_viz):
    """Test adding choropleth layer."""
    m = map_viz.create_base_map()

    # Sample GeoJSON data
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "area_1",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.42, 37.77],
                            [-122.41, 37.77],
                            [-122.41, 37.78],
                            [-122.42, 37.78],
                            [-122.42, 37.77],
                        ]
                    ],
                },
            }
        ],
    }

    values = {"area_1": 0.8}

    m = map_viz.add_choropleth(
        m, geojson_data, values, "Test Layer", legend_name="Test Values"
    )

    # Check choropleth layer
    choropleths = [
        child for child in m._children.values() if isinstance(child, folium.Choropleth)
    ]
    assert len(choropleths) == 1
    assert choropleths[0].name == "Test Layer"


def test_add_value_markers(map_viz):
    """Test adding value markers."""
    m = map_viz.create_base_map()

    values = {"business_1": 0.9, "business_2": 0.7, "business_3": 0.5}

    m = map_viz.add_value_markers(m, SAMPLE_LOCATIONS, values, radius=15)

    # Check circle markers
    markers = [
        child
        for child in m._children.values()
        if isinstance(child, folium.CircleMarker)
    ]
    assert len(markers) == len(SAMPLE_LOCATIONS)

    # Check colorbar
    colorbars = [
        child for child in m._children.values() if "ColorMap" in str(type(child))
    ]
    assert len(colorbars) == 1


def test_save_map(map_viz, tmp_path):
    """Test map saving functionality."""
    m = map_viz.create_base_map()
    m = map_viz.add_restaurants(m, SAMPLE_LOCATIONS, {})

    output_file = tmp_path / "test_map.html"
    map_viz.save_map(m, str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0
