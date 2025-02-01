"""Tests for the TasteProfileAnalyzer class."""

import pytest

from profiling import TasteProfileAnalyzer


@pytest.fixture
def analyzer():
    """Create a TasteProfileAnalyzer instance for testing."""
    return TasteProfileAnalyzer()


@pytest.fixture
def sample_reviews():
    """Get sample reviews for testing."""
    return {
        "restaurant_1": [
            "The food was extremely spicy with authentic flavors.",
            "Very fresh ingredients and quick service.",
            "The ambiance was romantic but prices were high.",
        ],
        "restaurant_2": [
            "Excellent vegan and gluten-free options.",
            "The plant-based menu is creative and healthy.",
            "100% organic ingredients and peaceful environment.",
        ],
    }


def test_analyze_review(analyzer):
    """Test analyzing a single review."""
    review = "The food was extremely spicy with authentic flavors."
    analyzer.analyze_review("test_business", review)

    profile = analyzer.get_business_profile("test_business")
    assert profile is not None
    assert profile.business_id == "test_business"

    # Check spiciness aspect
    assert hasattr(profile, "spicy")
    assert profile.confidence_scores["spicy"] > 0.3
    assert profile.spicy > 0  # Should be positive sentiment

    # Check authenticity aspect
    assert hasattr(profile, "authenticity")
    assert profile.confidence_scores["authenticity"] > 0.3
    assert profile.authenticity > 0


def test_multiple_reviews(analyzer, sample_reviews):
    """Test analyzing multiple reviews for multiple businesses."""
    # Process all reviews
    for business_id, reviews in sample_reviews.items():
        for review in reviews:
            analyzer.analyze_review(business_id, review)

    # Check first restaurant (spicy focus)
    profile1 = analyzer.get_business_profile("restaurant_1")
    assert profile1 is not None
    assert profile1.spicy > 0
    assert profile1.service_speed > 0
    assert profile1.romantic_ambiance > 0
    assert profile1.price_level > 0.5  # Should be on the expensive side

    # Check second restaurant (vegan focus)
    profile2 = analyzer.get_business_profile("restaurant_2")
    assert profile2 is not None
    assert profile2.vegan > 0
    assert profile2.gluten_free > 0
    assert profile2.organic > 0
    assert profile2.health_consciousness > 0


def test_clustering(analyzer, sample_reviews):
    """Test business clustering functionality."""
    # Process reviews
    for business_id, reviews in sample_reviews.items():
        for review in reviews:
            analyzer.analyze_review(business_id, review)

    # Get clusters
    clusters = analyzer.cluster_businesses()

    # Basic validation
    assert isinstance(clusters, dict)
    assert len(clusters) == len(sample_reviews)

    # Check cluster assignments
    for business_id in sample_reviews:
        assert business_id in clusters
        info = clusters[business_id]
        assert "cluster_label" in info
        assert "price_category" in info
        assert "dining_style" in info
        assert "dietary" in info


def test_aggregate_profile(analyzer, sample_reviews):
    """Test aggregating profiles from multiple businesses."""
    # Process reviews
    for business_id, reviews in sample_reviews.items():
        for review in reviews:
            analyzer.analyze_review(business_id, review)

    # Get aggregate profile
    aggregate = analyzer.get_aggregate_profile(list(sample_reviews.keys()))
    assert aggregate is not None

    # Check that aggregate has meaningful values
    significant = aggregate.get_significant_aspects()
    assert len(significant) > 0

    # At least some aspects should be significant
    total_aspects = sum(len(aspects) for aspects in significant.values())
    assert total_aspects > 0


def test_error_handling(analyzer):
    """Test error handling in the analyzer."""
    # Test with empty review
    analyzer.analyze_review("test_business", "")
    profile = analyzer.get_business_profile("test_business")
    assert profile is not None

    # Test with non-existent business
    profile = analyzer.get_business_profile("non_existent")
    assert profile is None

    # Test clustering with too few businesses
    clusters = analyzer.cluster_businesses()
    assert clusters == {}

    # Test aggregate profile with invalid business IDs
    aggregate = analyzer.get_aggregate_profile(["non_existent"])
    assert aggregate is None


def test_multilingual_support(analyzer):
    """Test analyzing reviews in different languages."""
    reviews = {
        "spanish": "La comida estaba muy picante y auténtica.",
        "french": "Le service était excellent et rapide.",
        "italian": "Il cibo era delizioso e fresco.",
    }

    for lang, review in reviews.items():
        analyzer.analyze_review(f"restaurant_{lang}", review)
        profile = analyzer.get_business_profile(f"restaurant_{lang}")
        assert profile is not None

        # Each review should have some significant aspects
        significant = profile.get_significant_aspects()
        assert len(significant) > 0


def test_confidence_scores(analyzer):
    """Test confidence score calculation."""
    # Single review should have lower confidence
    analyzer.analyze_review("test_business", "The food was spicy.")
    profile1 = analyzer.get_business_profile("test_business")
    conf1 = profile1.confidence_scores["spicy"]

    # Multiple similar reviews should increase confidence
    for _ in range(3):
        analyzer.analyze_review("test_business", "Very spicy food.")
    profile2 = analyzer.get_business_profile("test_business")
    conf2 = profile2.confidence_scores["spicy"]

    assert conf2 > conf1  # Confidence should increase with more reviews


def test_compound_aspects(analyzer):
    """Test detection of compound aspects."""
    review = "The dish had a perfect sweet and spicy balance."
    analyzer.analyze_review("test_business", review)
    profile = analyzer.get_business_profile("test_business")

    # Both individual aspects should be detected
    assert profile.sweet > 0
    assert profile.spicy > 0

    # Compound aspect should also be detected
    assert hasattr(profile, "sweet_spicy")
    assert profile.sweet_spicy > 0


def test_value_ratio_calculation(analyzer):
    """Test price level and value ratio calculations."""
    # High price, high quality
    analyzer.analyze_review(
        "expensive_good",
        "Expensive but excellent quality. Premium ingredients and perfect service.",
    )
    profile1 = analyzer.get_business_profile("expensive_good")

    # High price, low quality
    analyzer.analyze_review(
        "expensive_bad", "Overpriced and mediocre quality. Not worth the high prices."
    )
    profile2 = analyzer.get_business_profile("expensive_bad")

    # Price levels should be similar (both expensive)
    assert abs(profile1.price_level - profile2.price_level) < 0.3

    # But value ratios should differ significantly
    assert profile1.price_value_ratio > profile2.price_value_ratio
