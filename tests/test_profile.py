"""Tests for the TasteProfile class."""

import pytest

from profiling import TasteProfile


@pytest.fixture
def profile():
    """Create a TasteProfile instance for testing."""
    return TasteProfile(business_id="test_business")


def test_initialization(profile):
    """Test profile initialization."""
    assert profile.business_id == "test_business"
    assert profile.price_level == 0.0
    assert profile.price_value_ratio == 0.0
    assert profile.price_category == "$"
    assert profile.dining_style == "casual"
    assert not profile.kosher_certified
    assert not profile.halal_certified


def test_aspect_initialization(profile):
    """Test that all aspects are properly initialized."""
    # Check taste aspects
    assert hasattr(profile, "sweet")
    assert hasattr(profile, "salty")
    assert hasattr(profile, "spicy")
    assert hasattr(profile, "savory")
    assert hasattr(profile, "bitter")
    assert hasattr(profile, "sour")

    # Check texture aspects
    assert hasattr(profile, "crunchiness")
    assert hasattr(profile, "smoothness")
    assert hasattr(profile, "chewiness")
    assert hasattr(profile, "creaminess")
    assert hasattr(profile, "firmness")
    assert hasattr(profile, "juiciness")
    assert hasattr(profile, "softness")

    # Check dietary aspects
    assert hasattr(profile, "gluten_free")
    assert hasattr(profile, "dairy_free")
    assert hasattr(profile, "vegan")
    assert hasattr(profile, "vegetarian")
    assert hasattr(profile, "nut_free")
    assert hasattr(profile, "shellfish_free")
    assert hasattr(profile, "kosher")
    assert hasattr(profile, "halal")

    # Check ambiance aspects
    assert hasattr(profile, "lighting_quality")
    assert hasattr(profile, "noise_level")
    assert hasattr(profile, "seating_comfort")
    assert hasattr(profile, "service_speed")
    assert hasattr(profile, "cleanliness")
    assert hasattr(profile, "accessibility")
    assert hasattr(profile, "friendly_staff")
    assert hasattr(profile, "family_friendly")
    assert hasattr(profile, "romantic_ambiance")


def test_update_aspect(profile):
    """Test updating aspect scores."""
    # Update with single value
    profile.update_aspect("spicy", 0.8, 0.9)
    assert profile.spicy == 0.8
    assert profile.confidence_scores["spicy"] > 0.8
    assert profile.mention_counts["spicy"] == 1

    # Update with another value
    profile.update_aspect("spicy", 0.6, 0.7)
    assert 0.6 < profile.spicy < 0.8  # Should be weighted average
    assert profile.mention_counts["spicy"] == 2

    # Update with negative sentiment
    profile.update_aspect("spicy", -0.5, 0.8)
    assert profile.spicy < 0.6  # Should decrease
    assert profile.mention_counts["spicy"] == 3


def test_price_level_updates(profile):
    """Test price level and category updates."""
    # Start with low price
    profile.update_aspect("price_level", -0.5, 0.8)
    assert profile.price_level < 0.5
    assert profile.price_category == "$"

    # Update to medium price
    profile.update_aspect("price_level", 0.0, 0.9)
    assert 0.25 <= profile.price_level <= 0.75
    assert profile.price_category in ["$$", "$$$"]

    # Update to high price
    profile.update_aspect("price_level", 0.8, 0.9)
    assert profile.price_level > 0.5
    assert profile.price_category in ["$$$", "$$$$"]


def test_value_ratio_calculation(profile):
    """Test value ratio calculations."""
    # Set high price level
    profile.update_aspect("price_level", 0.8, 0.9)

    # Add quality indicators
    profile.update_aspect("freshness", 0.9, 0.8)
    profile.update_aspect("authenticity", 0.8, 0.8)
    profile.update_aspect("plating_aesthetics", 0.7, 0.8)
    profile.update_aspect("cleanliness", 0.9, 0.8)

    # Value ratio should reflect quality vs price
    assert profile.price_value_ratio > 0
    assert profile.price_value_ratio < 2.0  # Should be reasonable


def test_significant_aspects(profile):
    """Test getting significant aspects."""
    # Add some aspects with varying confidence
    profile.update_aspect("spicy", 0.8, 0.9)
    profile.update_aspect("sweet", 0.7, 0.4)
    profile.update_aspect("salty", 0.6, 0.2)  # Below threshold

    significant = profile.get_significant_aspects()

    # Check structure
    assert isinstance(significant, dict)
    assert "taste" in significant

    # Check filtering
    aspects = significant["taste"]
    assert "spicy" in aspects
    assert "sweet" in aspects
    assert "salty" not in aspects  # Below confidence threshold


def test_sentiment_variance(profile):
    """Test sentiment variance calculation."""
    # Add consistent sentiments
    for _ in range(3):
        profile.update_aspect("spicy", 0.8, 0.9)

    # Add mixed sentiments
    for _ in range(3):
        profile.update_aspect("sweet", 0.8, 0.9)
        profile.update_aspect("sweet", -0.8, 0.9)

    summary1 = profile.get_aspect_summary("spicy")
    summary2 = profile.get_aspect_summary("sweet")

    assert summary1["variance"] < summary2["variance"]
    assert not summary1["mixed_sentiment"]
    assert summary2["mixed_sentiment"]


def test_weighted_distance(profile):
    """Test distance calculation between profiles."""
    other_profile = TasteProfile(business_id="other_business")

    # Make profiles similar in some aspects
    for p in [profile, other_profile]:
        p.update_aspect("spicy", 0.8, 0.9)
        p.update_aspect("freshness", 0.7, 0.8)

    # Make them different in other aspects
    profile.update_aspect("sweet", 0.8, 0.9)
    other_profile.update_aspect("sweet", -0.8, 0.9)

    distance = profile.get_weighted_distance(other_profile)
    assert distance > 0  # Should be some difference
    assert distance < float("inf")  # Should be calculable


def test_to_dict(profile):
    """Test dictionary conversion."""
    # Add some aspects
    profile.update_aspect("spicy", 0.8, 0.9)
    profile.update_aspect("sweet", 0.7, 0.8)
    profile.price_level = 0.6

    data = profile.to_dict()

    assert isinstance(data, dict)
    assert "spicy" in data
    assert "sweet" in data
    assert "price_level" in data
    assert "confidence_scores" not in data  # Should be excluded
