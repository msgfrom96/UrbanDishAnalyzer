"""Aspect extraction module for analyzing restaurant reviews."""

import logging
from typing import Dict, List, Optional, Set, Tuple

from ..constants.aspects import ASPECT_KEYWORDS, QUALITY_INDICATORS
from ..constants.languages import (
    DINING_STYLE_KEYWORDS,
    INTENSITY_MODIFIERS,
    PRICE_KEYWORDS,
    SUPPORTED_LANGUAGES,
)
from ..constants.thresholds import MIN_CONFIDENCE_THRESHOLD
from ..utils import clean_text, detect_language

logger = logging.getLogger(__name__)


class AspectExtractor:
    """Extract aspects and their sentiments from review text."""

    def __init__(self) -> None:
        """Initialize the aspect extractor."""
        self.reset()

    def reset(self) -> None:
        """Reset the extractor state."""
        self.aspects: Dict[str, float] = {}
        self.mentions: Dict[str, int] = {}
        self.confidence_scores: Dict[str, float] = {}
        self.dining_styles: Set[str] = set()
        self.price_indicators: List[float] = []

    def extract_aspects(self, text: str) -> Dict[str, float]:
        """Extract aspects and their sentiment scores from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary mapping aspects to their sentiment scores
        """
        clean = clean_text(text)
        lang = detect_language(clean)

        if lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language: {lang}")
            return {}

        # Extract dining style
        style_scores: Dict[str, float] = {}
        style_matches: Dict[str, int] = {}

        for style, keywords in DINING_STYLE_KEYWORDS.items():
            matches = sum(1 for keyword in keywords if keyword in clean)
            if matches > 0:
                style_matches[style] = matches
                style_scores[style] = matches / len(keywords)

        if style_matches:
            max_style = max(style_scores.items(), key=lambda x: x[1])[0]
            self.dining_styles.add(max_style)

        # Extract price indicators
        for keyword, value in PRICE_KEYWORDS.items():
            if keyword in clean:
                self.price_indicators.append(value)

        # Extract aspects
        for aspect, keywords in ASPECT_KEYWORDS.items():
            matches = sum(1 for keyword in keywords if keyword in clean)
            if matches > 0:
                self.mentions[aspect] = self.mentions.get(aspect, 0) + matches
                confidence = matches / len(keywords)
                if confidence >= MIN_CONFIDENCE_THRESHOLD:
                    self.confidence_scores[aspect] = confidence
                    # Apply quality indicators
                    quality_score = self._calculate_quality_score(clean, aspect)
                    self.aspects[aspect] = quality_score

        return self.aspects

    def _calculate_quality_score(self, text: str, aspect: str) -> float:
        """Calculate quality score for an aspect based on indicators.

        Args:
            text: Cleaned text
            aspect: Aspect to calculate score for

        Returns:
            Quality score between 0 and 1
        """
        base_score = 0.5  # Neutral starting point
        indicators = QUALITY_INDICATORS.get(aspect, [])
        matches = sum(1 for indicator in indicators if indicator in text)

        if matches > 0:
            # Adjust score based on matches
            intensity = min(1.0, matches / len(indicators))
            base_score += intensity * 0.5  # Max adjustment of 0.5

            # Check for intensity modifiers
            for modifier in INTENSITY_MODIFIERS:
                if modifier in text:
                    base_score = min(1.0, base_score * 1.2)  # Boost by 20%

        return base_score

    def get_dining_style(self, min_confidence: Optional[float] = None) -> Optional[str]:
        """Get the dominant dining style.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            Dominant dining style or None if no style meets the threshold
        """
        if not self.dining_styles:
            return None

        style_counts: Dict[str, int] = {}
        for style in self.dining_styles:
            style_counts[style] = style_counts.get(style, 0) + 1

        if not style_counts:
            return None

        max_style = max(style_counts.items(), key=lambda x: x[1])[0]
        max_count = style_counts[max_style]
        total = sum(style_counts.values())

        confidence = max_count / total
        if min_confidence is not None and confidence < min_confidence:
            return None

        return max_style

    def get_price_level(self) -> Optional[float]:
        """Get the average price level.

        Returns:
            Average price level or None if no price indicators found
        """
        if not self.price_indicators:
            return None

        return sum(self.price_indicators) / len(self.price_indicators)

    def get_aspect_scores(
        self, min_mentions: Optional[int] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Get aspect scores with their confidence values.

        Args:
            min_mentions: Minimum number of mentions required

        Returns:
            Dictionary mapping aspects to (score, confidence) tuples
        """
        scores: Dict[str, Tuple[float, float]] = {}

        for aspect, mentions in self.mentions.items():
            if min_mentions is not None and mentions < min_mentions:
                continue

            confidence = self.confidence_scores.get(aspect, 0.0)
            if confidence >= MIN_CONFIDENCE_THRESHOLD:
                score = self.aspects.get(aspect, 0.0)
                scores[aspect] = (score, confidence)

        return scores
