"""Tests for the data loading module."""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import YelpDataLoader


@pytest.fixture
def sample_business_data():
    """Create sample business data for testing."""
    return [
        {
            "business_id": "b1",
            "name": "Restaurant A",
            "categories": "Restaurants, Italian",
            "latitude": 34.0522,
            "longitude": -118.2437,
            "stars": 4.5,
            "review_count": 100,
        },
        {
            "business_id": "b2",
            "name": "Restaurant B",
            "categories": "Food, Mexican",
            "latitude": 34.0500,
            "longitude": -118.2400,
            "stars": 3.5,
            "review_count": 50,
        },
        {
            "business_id": "b3",
            "name": "Auto Shop",  # Non-restaurant business
            "categories": "Automotive",
            "latitude": 34.0600,
            "longitude": -118.2500,
            "stars": 4.0,
            "review_count": 75,
        },
    ]


@pytest.fixture
def sample_review_data():
    """Create sample review data for testing."""
    return [
        {
            "review_id": "r1",
            "business_id": "b1",
            "text": "Great food!",
            "stars": 5,
            "date": "2021-01-01",
        },
        {
            "review_id": "r2",
            "business_id": "b2",
            "text": "Decent service",
            "stars": 3,
            "date": "2021-01-02",
        },
        {
            "review_id": "r3",
            "business_id": "b1",
            "text": None,  # Missing text
            "stars": 4,
            "date": "2021-01-03",
        },
    ]


def test_init():
    """Test YelpDataLoader initialization."""
    loader = YelpDataLoader("/test/path")
    assert loader.data_path == Path("/test/path")
    assert loader.business_df is None
    assert loader.reviews_df is None
    assert loader.users_df is None


@patch("builtins.open", new_callable=mock_open)
def test_load_business_data(mock_file, sample_business_data):
    """Test loading business data."""
    # Mock file content
    mock_file.return_value.__iter__ = lambda self: iter(
        [str(business) for business in sample_business_data]
    )

    loader = YelpDataLoader("/test/path")
    df = loader.load_business_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Only restaurants should remain
    assert all(df["business_id"].isin(["b1", "b2"]))
    assert "stars" in df.columns
    assert "review_count" in df.columns


@patch("builtins.open", new_callable=mock_open)
def test_load_reviews(mock_file, sample_review_data):
    """Test loading review data."""
    # Mock file content
    mock_file.return_value.__iter__ = lambda self: iter(
        [str(review) for review in sample_review_data]
    )

    loader = YelpDataLoader("/test/path")
    df = loader.load_reviews(chunk_size=2)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Reviews with text only
    assert all(df["review_id"].isin(["r1", "r2"]))
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


@patch("geopy.geocoders.Nominatim")
def test_filter_metropolitan_area(mock_nominatim):
    """Test filtering businesses by metropolitan area."""
    # Mock geocoder
    mock_location = Mock()
    mock_location.latitude = 34.0522
    mock_location.longitude = -118.2437
    mock_nominatim.return_value.geocode.return_value = mock_location

    loader = YelpDataLoader("/test/path")
    loader.business_df = pd.DataFrame(
        [
            {"business_id": "b1", "latitude": 34.0522, "longitude": -118.2437},
            {
                "business_id": "b2",
                "latitude": 40.7128,  # New York coordinates
                "longitude": -74.0060,
            },
        ]
    )

    filtered_df = loader.filter_metropolitan_area("Los Angeles", radius_km=10)

    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["business_id"] == "b1"


def test_clean_business_data(sample_business_data):
    """Test business data cleaning."""
    loader = YelpDataLoader("/test/path")
    loader.business_df = pd.DataFrame(sample_business_data)

    cleaned_df = loader._clean_business_data()

    assert len(cleaned_df) == 2  # Only restaurants
    assert all(cleaned_df["business_id"].isin(["b1", "b2"]))
    assert not cleaned_df["stars"].isna().any()
    assert not cleaned_df["review_count"].isna().any()


def test_clean_review_data(sample_review_data):
    """Test review data cleaning."""
    loader = YelpDataLoader("/test/path")
    loader.reviews_df = pd.DataFrame(sample_review_data)

    cleaned_df = loader._clean_review_data()

    assert len(cleaned_df) == 2  # Reviews with text only
    assert all(cleaned_df["review_id"].isin(["r1", "r2"]))
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["date"])


def test_merge_business_reviews(sample_business_data, sample_review_data):
    """Test merging business and review data."""
    loader = YelpDataLoader("/test/path")
    loader.business_df = pd.DataFrame(sample_business_data)
    loader.reviews_df = pd.DataFrame(sample_review_data)

    merged_df = loader.merge_business_reviews()

    assert len(merged_df) == 2  # Only valid reviews for restaurants
    assert "business_id" in merged_df.columns
    assert "review_id" in merged_df.columns
    assert "name" in merged_df.columns
    assert "text" in merged_df.columns


def test_error_handling():
    """Test error handling in data loader."""
    loader = YelpDataLoader("/test/path")

    # Test operations without loading data
    with pytest.raises(ValueError):
        loader._clean_business_data()

    with pytest.raises(ValueError):
        loader._clean_review_data()

    with pytest.raises(ValueError):
        loader.merge_business_reviews()

    with pytest.raises(ValueError):
        loader.filter_metropolitan_area("Los Angeles")


def test_haversine_distance():
    """Test haversine distance calculation."""
    loader = YelpDataLoader("/test/path")
    loader.business_df = pd.DataFrame(
        [{"business_id": "b1", "latitude": 34.0522, "longitude": -118.2437}]
    )

    # Test distance calculation (Los Angeles to New York)
    distance = loader.filter_metropolitan_area._haversine_distance(
        34.0522, -118.2437, 40.7128, -74.0060  # Los Angeles  # New York
    )

    assert abs(distance - 3935.75) < 1.0  # Distance should be ~3936 km
