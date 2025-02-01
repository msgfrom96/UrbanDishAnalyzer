"""Module for extracting taste profiles from reviews using ABSA."""

import logging
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.tokenize import sent_tokenize
from transformers import pipeline

from .constants import (
    ASPECT_MAPPING,
    ASPECT_RELATIONSHIPS,
    CERTIFICATION_KEYWORDS,
    COMPOUND_ASPECTS,
    DINING_STYLE_KEYWORDS,
    INTENSITY_MODIFIERS,
    NEGATION_WORDS,
    PRICE_KEYWORDS,
    PRICE_MODIFIERS,
    QUALITY_INDICATORS,
)

# Configure logging
logger = logging.getLogger(__name__)

# Single multilingual model for all languages
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
BATCH_SIZE = 32
CACHE_SIZE = 1024


def initialize_absa_pipeline() -> None:
    """Initialize the ABSA pipeline and required models."""
    try:
        # Initialize NLTK
        try:
            import nltk

            nltk.download("punkt")
            nltk.download("punkt_tab")
            # Test tokenization
            test_text = "This is a test sentence. This is another sentence."
            sentences = sent_tokenize(test_text)
            if not sentences or len(sentences) != 2:
                raise ValueError("NLTK sentence tokenization failed")
            logger.info("NLTK initialization successful")
        except Exception as e:
            logger.error(f"Failed to initialize NLTK: {str(e)}")
            raise

        # Initialize the extractor
        logger.info("ABSA pipeline initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize ABSA pipeline: {str(e)}")
        raise


class AspectExtractor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = {}

    def initialize(self):
        """Initialize the multilingual model once."""
        if self.model is None:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=MODEL_NAME,
                    tokenizer=MODEL_NAME,
                    device=self.device if torch.cuda.is_available() else -1,
                )
                logger.info(f"Initialized multilingual model on {self.device}")
            except Exception as e:
                logger.error(f"Failed to initialize model: {str(e)}")
                raise

    @lru_cache(maxsize=CACHE_SIZE)
    def get_sentiment(self, text: str) -> Tuple[float, float]:
        """Get sentiment with caching for frequently seen text."""
        result = self.sentiment_pipeline(text)[0]
        # Convert 1-5 star rating to -1 to 1 scale
        sentiment = (int(result["label"][0]) - 3) / 2
        return sentiment, result["score"]

    def process_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Process multiple texts in a single batch."""
        results = self.sentiment_pipeline(texts, batch_size=BATCH_SIZE)
        sentiments_confidences = []

        for result in results:
            # Convert 1-5 star rating to -1 to 1 scale
            sentiment = (int(result["label"][0]) - 3) / 2
            sentiments_confidences.append((sentiment, result["score"]))

        return sentiments_confidences


# Global extractor instance
_extractor: Optional[AspectExtractor] = None


def get_extractor() -> AspectExtractor:
    """Get or create the global extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = AspectExtractor()
        _extractor.initialize()
    return _extractor


def detect_price_level(
    text: str,
    dining_style: Optional[str] = None,
    cuisine_type: Optional[str] = None,
    location: Optional[str] = None,
) -> Tuple[float, float, str]:
    """Detect price level from text with context awareness.

    Args:
        text: Review text
        dining_style: Optional dining style for context
        cuisine_type: Optional cuisine type for context
        location: Optional location type for context

    Returns:
        Tuple[float, float, str]: (price_level, value_ratio, price_category)
    """
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "en"

    text_lower = text.lower()
    price_signals = []
    quality_signals = []

    # Get language-specific keywords
    keywords = PRICE_KEYWORDS.get(lang, PRICE_KEYWORDS["en"])

    # Extract price signals
    for word, weight in keywords.items():
        if word in text_lower:
            # Check for negations
            negated = any(
                neg in text_lower.split()
                for neg in NEGATION_WORDS.get(lang, NEGATION_WORDS["en"])
            )
            if negated:
                weight = -weight

            # Check for intensity modifiers
            for modifier, mod_weight in INTENSITY_MODIFIERS.items():
                if modifier in text_lower and text_lower.index(
                    modifier
                ) < text_lower.index(word):
                    weight *= mod_weight
                    break

            price_signals.append(weight)

    # Extract quality signals
    for level, indicators in QUALITY_INDICATORS.items():
        for indicator in indicators:
            if indicator in text_lower:
                if level == "high":
                    quality_signals.append(1.0)
                elif level == "good":
                    quality_signals.append(0.7)
                elif level == "average":
                    quality_signals.append(0.5)
                else:  # low
                    quality_signals.append(0.2)

    # Calculate base price level
    if price_signals:
        price_level = sum(price_signals) / len(price_signals)
    else:
        # If no explicit price signals, use quality as proxy
        price_level = (
            sum(quality_signals) / len(quality_signals) if quality_signals else 0.5
        )

    # Apply context modifiers
    if dining_style:
        price_level *= PRICE_MODIFIERS["dining_style"].get(dining_style, 1.0)
    if cuisine_type:
        price_level *= PRICE_MODIFIERS["cuisine_type"].get(cuisine_type, 1.0)
    if location:
        price_level *= PRICE_MODIFIERS["location_type"].get(location, 1.0)

    # Normalize price level
    price_level = max(0.0, min(1.0, price_level))

    # Calculate value ratio based on quality vs price
    avg_quality = (
        sum(quality_signals) / len(quality_signals) if quality_signals else price_level
    )
    value_ratio = avg_quality / price_level if price_level > 0 else avg_quality

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


def detect_dining_style(text: str) -> str:
    """Detect dining style from text.

    Args:
        text: Review text

    Returns:
        str: Dining style category
    """
    try:
        # Detect language
        lang = detect(text)
    except LangDetectException:
        lang = "en"

    text_lower = text.lower()
    style_matches = defaultdict(int)

    # Get language-specific keywords
    keywords = DINING_STYLE_KEYWORDS.get(lang, DINING_STYLE_KEYWORDS["en"])

    # Count matches for each style
    for style, words in keywords.items():
        for word in words:
            if word in text_lower:
                style_matches[style] += 1

    # Return most frequent style or default
    if style_matches:
        return max(style_matches.items(), key=lambda x: x[1])[0]
    return "casual"  # Default to casual


def detect_certifications(text: str) -> Dict[str, bool]:
    """Detect kosher and halal certifications from text.

    Args:
        text: Review text

    Returns:
        Dict[str, bool]: Certification status
    """
    try:
        # Detect language
        lang = detect(text)
    except LangDetectException:
        lang = "en"

    text_lower = text.lower()
    certifications = {"kosher_certified": False, "halal_certified": False}

    # Get language-specific keywords
    keywords = CERTIFICATION_KEYWORDS.get(lang, CERTIFICATION_KEYWORDS["en"])

    # Check for certifications
    for cert, words in keywords.items():
        for word in words:
            if word in text_lower:
                certifications[cert] = True
                break

    return certifications


def extract_aspects_and_sentiments(text: str) -> List[Tuple[str, float, float, str]]:
    """Extract aspects and their sentiments from text.

    Args:
        text: The text to analyze

    Returns:
        List of tuples (aspect, sentiment, confidence, category)
    """
    extractor = get_extractor()
    aspects_sentiments = []

    try:
        lang = detect(text)
    except LangDetectException:
        lang = "en"

    # Split into sentences for better context handling
    sentences = sent_tokenize(text)

    # First pass: Detect dining style and price level
    dining_style = None
    price_signals = []
    quality_signals = []

    for sentence in sentences:
        sentence_lower = sentence.lower()

        # Check dining style keywords
        style_keywords = DINING_STYLE_KEYWORDS.get(lang, DINING_STYLE_KEYWORDS["en"])
        for style, keywords in style_keywords.items():
            if any(kw in sentence_lower for kw in keywords):
                dining_style = style
                sentiment, confidence = extractor.get_sentiment(sentence)
                aspects_sentiments.append(
                    (style, sentiment, confidence, "dining_style")
                )
                break

        # Check price keywords
        price_kw = PRICE_KEYWORDS.get(lang, PRICE_KEYWORDS["en"])
        for word, weight in price_kw.items():
            if word in sentence_lower:
                sentiment, confidence = extractor.get_sentiment(sentence)

                # Check for negations
                negated = any(
                    neg in sentence_lower.split()
                    for neg in NEGATION_WORDS.get(lang, NEGATION_WORDS["en"])
                )
                if negated:
                    sentiment = -sentiment

                # Check for intensity modifiers
                for modifier, mod_weight in INTENSITY_MODIFIERS.items():
                    if modifier in sentence_lower and sentence_lower.index(
                        modifier
                    ) < sentence_lower.index(word):
                        sentiment *= mod_weight
                        break

                price_signals.append((sentiment * weight, confidence))

        # Check quality indicators
        for level, indicators in QUALITY_INDICATORS.items():
            for indicator in indicators:
                if indicator in sentence_lower:
                    sentiment, confidence = extractor.get_sentiment(sentence)
                    if level == "high":
                        quality_signals.append((1.0, confidence))
                    elif level == "good":
                        quality_signals.append((0.7, confidence))
                    elif level == "average":
                        quality_signals.append((0.5, confidence))
                    else:  # low
                        quality_signals.append((0.2, confidence))

    # Calculate price level and value ratio
    if price_signals:
        weighted_price = sum(p * c for p, c in price_signals) / sum(
            c for _, c in price_signals
        )
        price_confidence = sum(c for _, c in price_signals) / len(price_signals)
        aspects_sentiments.append(
            ("price_level", weighted_price, price_confidence, "price")
        )
    elif quality_signals:  # Use quality as proxy for price
        weighted_quality = sum(q * c for q, c in quality_signals) / sum(
            c for _, c in quality_signals
        )
        quality_confidence = sum(c for _, c in quality_signals) / len(quality_signals)
        aspects_sentiments.append(
            ("price_level", weighted_quality, quality_confidence * 0.8, "price")
        )

    # Apply dining style modifiers to price
    if dining_style and dining_style in PRICE_MODIFIERS["dining_style"]:
        modifier = PRICE_MODIFIERS["dining_style"][dining_style]
        for i, (aspect, sentiment, confidence, category) in enumerate(
            aspects_sentiments
        ):
            if aspect == "price_level":
                aspects_sentiments[i] = (
                    aspect,
                    sentiment * modifier,
                    confidence,
                    category,
                )

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
                        negated = any(
                            neg in sentence_lower.split()
                            for neg in NEGATION_WORDS.get(lang, NEGATION_WORDS["en"])
                        )
                        if negated:
                            sentiment = -sentiment

                        # Check for intensity modifiers
                        for modifier, weight in INTENSITY_MODIFIERS.items():
                            if modifier in sentence_lower and sentence_lower.index(
                                modifier
                            ) < sentence_lower.index(keyword):
                                sentiment *= weight
                                confidence *= weight
                                break

                        # Apply aspect relationships
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

    # Update mention counts
    for aspect, _, _, _ in aspects_sentiments:
        profile.mention_counts[aspect] = profile.mention_counts.get(aspect, 0) + 1

        # Detailed logging of the update
        logger.info(
            f"Updated {aspect}:\n" f"  Total mentions: {profile.mention_counts[aspect]}"
        )
