"""Tests for the aspect extraction module."""

import pytest

from profiling.extraction.extractor import AspectExtractor


def test_extract_aspects():
    """Test basic aspect extraction."""
    extractor = AspectExtractor()
    text = "The food was delicious and authentic. Great service!"

    aspects = extractor.extract_aspects(text)
    assert aspects
    assert any(a for a in aspects if "taste" in a)
    assert any(a for a in aspects if "service" in a)


def test_multilingual_extraction():
    """Test aspect extraction in different languages."""
    extractor = AspectExtractor()

    # English
    en_text = "The food was delicious and authentic. Great service!"
    en_aspects = extractor.extract_aspects(en_text)

    # Spanish
    es_text = "La comida estaba deliciosa y auténtica. ¡Excelente servicio!"
    es_aspects = extractor.extract_aspects(es_text)

    # French
    fr_text = "La nourriture était délicieuse et authentique. Excellent service!"
    fr_aspects = extractor.extract_aspects(fr_text)

    # Check that similar aspects are extracted regardless of language
    assert len(en_aspects) > 0
    assert len(es_aspects) > 0
    assert len(fr_aspects) > 0

    # Check for common aspects across languages
    common_categories = set.intersection(
        {a["category"] for a in en_aspects},
        {a["category"] for a in es_aspects},
        {a["category"] for a in fr_aspects},
    )
    assert len(common_categories) > 0


def test_aspect_confidence():
    """Test confidence scores for extracted aspects."""
    extractor = AspectExtractor()

    # Strong signal
    strong_text = "The food was absolutely delicious! Amazing flavors!"
    strong_aspects = extractor.extract_aspects(strong_text)

    # Weak signal
    weak_text = "I had dinner here."
    weak_aspects = extractor.extract_aspects(weak_text)

    # Check confidence scores
    strong_confidence = max(a["confidence"] for a in strong_aspects)
    weak_confidence = max(a["confidence"] for a in weak_aspects)
    assert strong_confidence > weak_confidence


def test_compound_aspects():
    """Test extraction of compound aspects."""
    extractor = AspectExtractor()
    text = "The spicy Thai curry had authentic flavors but small portions."

    aspects = extractor.extract_aspects(text)

    # Check for both individual and compound aspects
    categories = [a["category"] for a in aspects]
    assert "taste" in categories
    assert "portion_size" in categories

    # Check for specific compound aspects
    aspect_names = [a["aspect"] for a in aspects]
    assert any("spicy" in a for a in aspect_names)
    assert any("authentic" in a for a in aspect_names)


def test_negation_handling():
    """Test handling of negated aspects."""
    extractor = AspectExtractor()

    # Positive statement
    pos_text = "The food was delicious"
    pos_aspects = extractor.extract_aspects(pos_text)

    # Negated statement
    neg_text = "The food was not delicious"
    neg_aspects = extractor.extract_aspects(neg_text)

    # Check that sentiment is reversed for negated aspects
    pos_sentiment = next(
        a["sentiment"] for a in pos_aspects if "taste" in a["category"]
    )
    neg_sentiment = next(
        a["sentiment"] for a in neg_aspects if "taste" in a["category"]
    )
    assert pos_sentiment > 0
    assert neg_sentiment < 0


def test_context_awareness():
    """Test context-aware aspect extraction."""
    extractor = AspectExtractor()
    text = """
    The appetizers were delicious but overpriced.
    For the main course, the portions were generous but the taste was bland.
    The dessert was amazing and worth every penny.
    """

    aspects = extractor.extract_aspects(text)

    # Check that aspects are correctly associated with their context
    categories = [a["category"] for a in aspects]
    assert "taste" in categories
    assert "value" in categories
    assert "portion_size" in categories

    # Check for course-specific aspects
    courses = [a.get("context", "") for a in aspects]
    assert any("appetizer" in c.lower() for c in courses)
    assert any("main" in c.lower() for c in courses)
    assert any("dessert" in c.lower() for c in courses)
