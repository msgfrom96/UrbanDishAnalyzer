"""Test configuration and fixtures."""

import json
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from profiling.core.analyzer import TasteProfileAnalyzer
from profiling.core.profile import TasteProfile
from profiling.data.loader import DataLoader
from profiling.data.storage import DataStorage

# Sample data for testing
SAMPLE_REVIEWS = [
    "The food was delicious and very authentic. Great service!",
    "Amazing flavors, but the portions were a bit small.",
    "The atmosphere is cozy and the staff is friendly.",
]

SAMPLE_BUSINESSES = [
    {
        "business_id": "test_business_1",
        "name": "Test Restaurant 1",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "stars": 4.5,
        "review_count": 100,
        "categories": ["Restaurants", "Italian"],
        "attributes": {"RestaurantsAttire": "casual", "NoiseLevel": "average"},
    },
    {
        "business_id": "test_business_2",
        "name": "Test Restaurant 2",
        "latitude": 37.7833,
        "longitude": -122.4167,
        "stars": 4.0,
        "review_count": 50,
        "categories": ["Restaurants", "Mexican"],
        "attributes": {"RestaurantsAttire": "casual", "NoiseLevel": "loud"},
    },
]


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory with test data files."""
    # Create business data
    business_file = tmp_path / "business.json"
    with open(business_file, "w") as f:
        for business in SAMPLE_BUSINESSES:
            f.write(json.dumps(business) + "\n")

    # Create review data
    review_file = tmp_path / "review.json"
    with open(review_file, "w") as f:
        for i, review in enumerate(SAMPLE_REVIEWS):
            review_data = {
                "review_id": f"review_{i}",
                "business_id": "test_business_1",
                "text": review,
                "stars": 4,
                "date": "2024-01-01",
            }
            f.write(json.dumps(review_data) + "\n")

    # Create metro areas data
    metro_file = tmp_path / "metro_areas.geojson"
    metro_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-122.4194, 37.7749]},
                "properties": {"name": "San Francisco", "population": 874961},
            }
        ],
    }
    with open(metro_file, "w") as f:
        json.dump(metro_data, f)

    return tmp_path


@pytest.fixture
def data_loader(temp_data_dir):
    """Create a DataLoader instance with test data."""
    loader = DataLoader()
    loader.load_businesses(str(temp_data_dir / "business.json"))
    loader.load_reviews(str(temp_data_dir / "review.json"))
    loader.load_metro_areas(str(temp_data_dir / "metro_areas.geojson"))
    return loader


@pytest.fixture
def data_storage(tmp_path):
    """Create a DataStorage instance with temporary directories."""
    return DataStorage(
        cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results")
    )


@pytest.fixture
def profile_analyzer():
    """Create a TasteProfileAnalyzer instance."""
    return TasteProfileAnalyzer()


@pytest.fixture
def sample_profile():
    """Create a sample TasteProfile instance."""
    profile = TasteProfile(business_id="test_business_1")

    # Set some test values
    profile.taste_quality = 0.8
    profile.service_quality = 0.7
    profile.ambiance = 0.6
    profile.value_ratio = 0.75
    profile.price_level = 2
    profile.dining_style = "casual"

    # Set confidence scores
    profile.confidence_scores = {
        "taste_quality": 0.9,
        "service_quality": 0.8,
        "ambiance": 0.7,
        "value_ratio": 0.85,
    }

    return profile


@pytest.fixture
def sample_businesses_df():
    """Create a sample DataFrame of businesses."""
    return pd.DataFrame(SAMPLE_BUSINESSES)


@pytest.fixture
def sample_reviews_df():
    """Create a sample DataFrame of reviews."""
    reviews = []
    for i, review in enumerate(SAMPLE_REVIEWS):
        reviews.append(
            {
                "review_id": f"review_{i}",
                "business_id": "test_business_1",
                "text": review,
                "stars": 4,
                "date": "2024-01-01",
            }
        )
    return pd.DataFrame(reviews)
