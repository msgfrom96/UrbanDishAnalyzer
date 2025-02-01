"""Module for sentiment analysis and aspect extraction from review text."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from langdetect import LangDetectException, detect
from nltk.tokenize import sent_tokenize

from ..constants.aspects import (
    ASPECT_MAPPING,
    ASPECT_RELATIONSHIPS,
    COMPOUND_ASPECTS,
    QUALITY_INDICATORS,
)
from ..constants.languages import INTENSITY_MODIFIERS, NEGATION_WORDS
from .extractor import get_extractor

# Configure logging
logger = logging.getLogger(__name__)


def extract_aspects_and_sentiments(text: str) -> List[Tuple[str, float, float, str]]:
    """Extract aspects and their sentiments from text.

    Args:
        text: Text to analyze

    Returns:
        List of tuples (aspect, sentiment, confidence, category)
    """
    extractor = get_extractor()
    aspects_sentiments = []

    # Split into sentences for better context handling
    sentences = sent_tokenize(text)

    # First pass: Detect dining style and price level
    dining_style = None
    dining_confidence = 0.0
    price_signals = []
    value_signals = []

    for sentence in sentences:
        # Extract dining style
        style, confidence = extractor.detect_dining_style(sentence)
        if confidence > dining_confidence:
            dining_style = style
            dining_confidence = confidence
            aspects_sentiments.append((dining_style, 1.0, confidence, "dining_style"))

        # Extract price and value signals
        price_level, value_ratio, price_category = extractor.detect_price_level(
            sentence, dining_style=dining_style
        )
        if price_level > 0 or value_ratio > 0:
            price_signals.append((price_level, value_ratio))

    # Calculate overall price level and value ratio
    if price_signals:
        avg_price = sum(p for p, _ in price_signals) / len(price_signals)
        avg_value = sum(v for _, v in price_signals) / len(price_signals)
        aspects_sentiments.append(("price_level", avg_price, 0.8, "price"))
        aspects_sentiments.append(("value_ratio", avg_value, 0.7, "price"))

    # Second pass: Process regular aspects
    for sentence in sentences:
        sentence_lower = sentence.lower()
        found_aspects = set()

        # Check for compound aspects first
        for compound_aspect, (components, weight) in COMPOUND_ASPECTS.items():
            if all(comp in sentence_lower for comp in components):
                sentiment, confidence = extractor.get_sentiment(sentence)
                sentiment *= weight
                confidence *= weight
                aspects_sentiments.append(
                    (compound_aspect, sentiment, confidence, "compound")
                )
                found_aspects.update(components)

        # Process regular aspects
        for category, aspects in ASPECT_MAPPING.items():
            for aspect, keywords in aspects.items():
                if aspect in found_aspects:
                    continue

                for keyword in keywords:
                    if keyword in sentence_lower:
                        sentiment, confidence = extractor.get_sentiment(sentence)

                        # Check for negations
                        negations = NEGATION_WORDS.get(
                            "en", set()
                        )  # Default to English
                        if any(neg in sentence_lower.split() for neg in negations):
                            sentiment = -sentiment

                        # Check for intensity modifiers
                        for modifier, weight in INTENSITY_MODIFIERS.items():
                            if modifier in sentence_lower and sentence_lower.index(
                                modifier
                            ) < sentence_lower.index(keyword):
                                sentiment *= weight
                                confidence *= weight
                                break

                        # Apply dining style context
                        if dining_style and dining_style in ASPECT_RELATIONSHIPS:
                            relationships = ASPECT_RELATIONSHIPS[dining_style]
                            if aspect in relationships:
                                sentiment *= relationships[aspect]
                                confidence *= relationships[aspect]

                        aspects_sentiments.append(
                            (aspect, sentiment, confidence, category)
                        )
                        break

    return aspects_sentiments


def map_aspects_to_profile(
    aspects_sentiments: List[Tuple[str, float, float, str]], profile: Any
) -> None:
    """Map extracted aspects to profile attributes.

    This function updates the taste profile with the extracted aspects and
    their sentiments, handling compound aspects and relationships between
    different aspects.

    Args:
        aspects_sentiments: List of (aspect, sentiment, confidence, category) tuples
        profile: TasteProfile instance to update
    """
    # Group aspects by category
    by_category = defaultdict(list)
    for aspect, sentiment, confidence, category in aspects_sentiments:
        by_category[category].append((aspect, sentiment, confidence))

    # Process compound aspects first
    if "compound" in by_category:
        for aspect, sentiment, confidence in by_category["compound"]:
            # Update the compound aspect
            if hasattr(profile, aspect):
                profile.update_aspect(aspect, sentiment, confidence)

            # Update related aspects with reduced confidence
            if aspect in COMPOUND_ASPECTS:
                components, _ = COMPOUND_ASPECTS[aspect]
                for component in components:
                    if hasattr(profile, component):
                        profile.update_aspect(component, sentiment, confidence * 0.8)

    # Process regular aspects
    for category, aspects in by_category.items():
        if category == "compound":
            continue

        for aspect, sentiment, confidence in aspects:
            # Update the main aspect
            if hasattr(profile, aspect):
                profile.update_aspect(aspect, sentiment, confidence)

            # Apply aspect relationships
            if category == "dining_style" and aspect in ASPECT_RELATIONSHIPS:
                for related_aspect, weight in ASPECT_RELATIONSHIPS[aspect].items():
                    if hasattr(profile, related_aspect):
                        profile.update_aspect(
                            related_aspect,
                            sentiment * weight,
                            confidence * 0.8,  # Reduce confidence for inferred aspects
                        )

    # Handle certifications separately
    if "certification" in by_category:
        for aspect, sentiment, confidence in by_category["certification"]:
            if aspect == "kosher_certified":
                profile.kosher_certified = True
                profile.kosher = 1.0
            elif aspect == "halal_certified":
                profile.halal_certified = True
                profile.halal = 1.0


def detect_price_level(text: str, dining_style: str = None) -> Tuple[float, float, str]:
    """Detect price level and value ratio from text.

    Args:
        text: Review text
        dining_style: Optional dining style for context

    Returns:
        Tuple of (price_level, value_ratio, price_category)
    """
    # Price signal words and their weights
    price_signals = {
        "expensive": 0.9,
        "pricey": 0.8,
        "high-end": 0.9,
        "luxury": 1.0,
        "upscale": 0.85,
        "fancy": 0.8,
        "premium": 0.85,
        "reasonable": 0.5,
        "affordable": 0.4,
        "cheap": 0.2,
        "budget": 0.3,
        "bargain": 0.3,
        "overpriced": -0.2,  # Negative signal
        "value": 0.5,
        "worth": 0.5,
        "fair": 0.5,
    }

    # Value signal words and their weights
    value_signals = {
        "worth every penny": 1.0,
        "great value": 0.9,
        "good value": 0.8,
        "fair price": 0.7,
        "reasonable price": 0.7,
        "not worth": -0.8,
        "overpriced": -0.7,
        "too expensive": -0.6,
    }

    # Dining style price modifiers
    style_modifiers = {
        "fine_dining": 1.3,
        "upscale": 1.2,
        "casual": 0.8,
        "bistro": 0.9,
        "fast_food": 0.6,
    }

    text_lower = text.lower()
    price_scores = []
    value_scores = []

    # Check for explicit price signals
    for word, weight in price_signals.items():
        if word in text_lower:
            # Check for negations
            if any(neg in text_lower.split() for neg in ["not", "no", "never"]):
                weight = -weight
            price_scores.append(weight)

    # Check for value signals
    for phrase, weight in value_signals.items():
        if phrase in text_lower:
            value_scores.append(weight)

    # Calculate base price level
    if price_scores:
        price_level = sum(price_scores) / len(price_scores)
    else:
        # Default to middle range if no explicit signals
        price_level = 0.5

    # Apply dining style modifier
    if dining_style and dining_style in style_modifiers:
        price_level *= style_modifiers[dining_style]

    # Normalize to 0-1 range
    price_level = max(0.0, min(1.0, (price_level + 1) / 2))

    # Calculate value ratio
    if value_scores:
        value_ratio = sum(value_scores) / len(value_scores)
        value_ratio = max(0.0, min(1.0, (value_ratio + 1) / 2))
    else:
        # Default to neutral if no explicit signals
        value_ratio = 0.5

    # Determine price category
    if price_level < 0.25:
        category = "$"
    elif price_level < 0.5:
        category = "$$"
    elif price_level < 0.75:
        category = "$$$"
    else:
        category = "$$$$"

    return price_level, value_ratio, category


def detect_dining_style(text: str) -> Tuple[str, float]:
    """Detect dining style from text.

    Args:
        text: Review text

    Returns:
        Tuple of (dining_style, confidence)
    """
    # Style indicators and their weights
    style_indicators = {
        "fine_dining": {
            "tasting menu": 1.0,
            "sommelier": 1.0,
            "wine pairing": 0.9,
            "michelin": 1.0,
            "fine dining": 1.0,
            "upscale": 0.9,
            "elegant": 0.8,
            "sophisticated": 0.8,
            "gourmet": 0.8,
            "haute cuisine": 1.0,
            "culinary experience": 0.9,
            "formal": 0.8,
        },
        "casual": {
            "casual": 1.0,
            "relaxed": 0.8,
            "laid-back": 0.8,
            "cozy": 0.7,
            "neighborhood spot": 0.9,
            "family-friendly": 0.8,
            "comfortable": 0.7,
        },
        "bistro": {
            "bistro": 1.0,
            "cafe": 0.8,
            "brasserie": 0.9,
            "wine bar": 0.8,
            "gastropub": 0.8,
        },
        "fast_food": {
            "fast food": 1.0,
            "quick service": 0.9,
            "drive-thru": 1.0,
            "takeout": 0.8,
            "food court": 0.9,
            "counter service": 0.9,
        },
    }

    text_lower = text.lower()
    style_scores = defaultdict(list)

    # Check for style indicators
    for style, indicators in style_indicators.items():
        for phrase, weight in indicators.items():
            if phrase in text_lower:
                # Check for negations
                if any(neg in text_lower.split() for neg in ["not", "no", "never"]):
                    weight = -weight
                style_scores[style].append(weight)

    # Calculate confidence and score for each style
    style_results = []
    for style, scores in style_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            confidence = min(
                1.0, 0.5 + (len(scores) * 0.2)
            )  # Higher confidence with more indicators
            style_results.append((style, avg_score, confidence))

    if style_results:
        # Sort by score * confidence
        style_results.sort(key=lambda x: x[1] * x[2], reverse=True)
        best_style, score, confidence = style_results[0]
        return best_style, confidence
    else:
        # Default to casual with low confidence if no clear indicators
        return "casual", 0.3
