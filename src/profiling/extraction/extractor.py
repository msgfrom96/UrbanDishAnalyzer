"""Module for extracting aspects and sentiments from review text."""

import logging
from collections import defaultdict
from functools import lru_cache
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, TypedDict, Union

import torch
from langdetect import LangDetectException, detect
from nltk.tokenize import sent_tokenize
from transformers import Pipeline, pipeline

from ..constants.languages import (
    CERTIFICATION_KEYWORDS,
    DINING_STYLE_KEYWORDS,
    INTENSITY_MODIFIERS,
    NEGATION_WORDS,
    PRICE_KEYWORDS,
    SUPPORTED_LANGUAGES,
)
from ..constants.thresholds import (
    BATCH_SIZE,
    CACHE_SIZE,
    MIN_CONFIDENCE_THRESHOLD,
    MODEL_NAME,
)

# Configure logging
logger = logging.getLogger(__name__)


class AspectResult(TypedDict):
    """Type definition for aspect extraction results."""

    dining_style: str
    style_confidence: float
    price_level: float
    value_ratio: float
    price_category: str
    taste: Dict[str, Tuple[float, float]]
    food: Dict[str, Tuple[float, float]]
    ambiance: Dict[str, Tuple[float, float]]


class AspectExtractor:
    """Class for extracting aspects and sentiments from review text."""

    def __init__(self) -> None:
        """Initialize the AspectExtractor."""
        self.sentiment_pipeline: Optional[Pipeline] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache: Dict[str, Tuple[float, float]] = {}
        self.style_scores: DefaultDict[str, List[float]] = defaultdict(list)

    def initialize(self) -> None:
        """Initialize the multilingual model and required components."""
        if self.sentiment_pipeline is None:
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
        """Get sentiment score for text with caching.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (sentiment_score, confidence)
            sentiment_score is in range [-1, 1]
            confidence is in range [0, 1]
        """
        if self.sentiment_pipeline is None:
            self.initialize()
        result = self.sentiment_pipeline(text)[0]
        # Convert 1-5 star rating to -1 to 1 scale
        sentiment = (int(result["label"][0]) - 3) / 2
        return sentiment, result["score"]

    def process_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Process multiple texts in a single batch.

        Args:
            texts: List of texts to analyze

        Returns:
            List of (sentiment_score, confidence) tuples
        """
        if self.sentiment_pipeline is None:
            self.initialize()
        results = self.sentiment_pipeline(texts, batch_size=BATCH_SIZE)
        return [
            ((int(result["label"][0]) - 3) / 2, result["score"]) for result in results
        ]

    def extract_aspects_and_sentiments(self, text: str) -> AspectResult:
        """Extract aspects and their sentiments from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping aspect categories to their aspects and scores
        """
        # Initialize model if needed
        if self.sentiment_pipeline is None:
            self.initialize()

        # Extract dining style
        dining_style, style_confidence = self.detect_dining_style(text)

        # Extract price information
        price_level, value_ratio, price_category = self.detect_price_level(
            text, dining_style
        )

        # Extract taste aspects
        taste_aspects = self.extract_taste_aspects(text)

        # Extract food aspects
        food_aspects = self.extract_food_aspects(text)

        # Extract ambiance aspects
        ambiance_aspects = self.extract_ambiance_aspects(text)

        return {
            "dining_style": dining_style,
            "style_confidence": style_confidence,
            "price_level": price_level,
            "value_ratio": value_ratio,
            "price_category": price_category,
            "taste": taste_aspects,
            "food": food_aspects,
            "ambiance": ambiance_aspects,
        }

    def detect_dining_style(self, text: str) -> Tuple[str, float]:
        """Detect dining style from text.

        Args:
            text: Review text

        Returns:
            Tuple of (dining_style, confidence)
        """
        # Style indicators and their weights
        style_indicators: Dict[str, Dict[str, float]] = {
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
        self.style_scores.clear()

        # Check for style indicators
        for style, indicators in style_indicators.items():
            for phrase, weight in indicators.items():
                if phrase in text_lower:
                    # Check for negations
                    if any(neg in text_lower.split() for neg in NEGATION_WORDS["en"]):
                        weight = -weight
                    self.style_scores[style].append(weight)

        # Calculate confidence and score for each style
        style_results: List[Tuple[str, float, float]] = []
        for style, scores in self.style_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                confidence = min(
                    1.0, 0.5 + (len(scores) * 0.2)
                )  # Higher confidence with more indicators
                style_results.append((style, avg_score, confidence))

        if style_results:
            # Sort by score * confidence
            style_results.sort(key=lambda x: x[1] * x[2], reverse=True)
            best_style, _, confidence = style_results[0]
            return best_style, confidence
        else:
            # Default to casual with low confidence if no clear indicators
            return "casual", 0.3

    def detect_price_level(
        self, text: str, dining_style: Optional[str] = None
    ) -> Tuple[float, float, str]:
        """Detect price level and value ratio from text.

        Args:
            text: Review text
            dining_style: Optional dining style for context

        Returns:
            Tuple of (price_level, value_ratio, price_category)
        """
        # Price signal words and their weights
        price_signals: Dict[str, float] = {
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

        # Dining style price modifiers
        style_modifiers: Dict[str, float] = {
            "fine_dining": 1.3,
            "upscale": 1.2,
            "casual": 0.8,
            "bistro": 0.9,
            "fast_food": 0.6,
        }

        text_lower = text.lower()
        price_scores: List[float] = []

        # Check for explicit price signals
        for word, weight in price_signals.items():
            if word in text_lower:
                # Check for negations
                if any(neg in text_lower.split() for neg in NEGATION_WORDS["en"]):
                    weight = -weight
                price_scores.append(weight)

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

        # Calculate value ratio (default to neutral if no explicit signals)
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

    def detect_certifications(self, text: str) -> Dict[str, bool]:
        """Detect dietary certifications from text.

        Args:
            text: Review text

        Returns:
            Dict mapping certification types to boolean values
        """
        try:
            lang = detect(text)
        except LangDetectException:
            lang = "en"

        text_lower = text.lower()
        certifications = {"kosher_certified": False, "halal_certified": False}

        # Get language-specific keywords
        keywords = CERTIFICATION_KEYWORDS.get(lang, CERTIFICATION_KEYWORDS["en"])

        # Check for certifications
        for cert, words in keywords.items():
            if any(word in text_lower for word in words):
                certifications[cert] = True

        return certifications

    def extract_taste_aspects(self, text: str) -> Dict[str, Tuple[float, float]]:
        """Extract taste-related aspects from text.

        Args:
            text: Review text

        Returns:
            Dictionary mapping aspects to (score, confidence) tuples
        """
        # TODO: Implement taste aspect extraction
        return {}

    def extract_food_aspects(self, text: str) -> Dict[str, Tuple[float, float]]:
        """Extract food-related aspects from text.

        Args:
            text: Review text

        Returns:
            Dictionary mapping aspects to (score, confidence) tuples
        """
        # TODO: Implement food aspect extraction
        return {}

    def extract_ambiance_aspects(self, text: str) -> Dict[str, Tuple[float, float]]:
        """Extract ambiance-related aspects from text.

        Args:
            text: Review text

        Returns:
            Dictionary mapping aspects to (score, confidence) tuples
        """
        # TODO: Implement ambiance aspect extraction
        return {}


# Global extractor instance
_extractor: Optional[AspectExtractor] = None


def get_extractor() -> AspectExtractor:
    """Get or create the global extractor instance.

    Returns:
        Initialized AspectExtractor instance
    """
    global _extractor
    if _extractor is None:
        _extractor = AspectExtractor()
        _extractor.initialize()
    return _extractor
