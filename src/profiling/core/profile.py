"""Core module for taste profile representation."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..constants.aspects import ASPECT_MAPPING, ASPECT_WEIGHTS
from ..constants.thresholds import (
    CONFIDENCE_BOOST_PER_MENTION,
    MAX_CONFIDENCE_SCORE,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_MENTIONS_FOR_RELIABLE_SCORE,
    MIXED_SENTIMENT_VARIANCE_THRESHOLD,
    PRICE_CATEGORY_THRESHOLDS,
    VALUE_RATIO_THRESHOLDS,
)

# Configure logging
logger = logging.getLogger(__name__)


class TasteProfile:
    """Class representing a business's taste profile.

    This class maintains the taste profile of a business, including various
    aspects like taste, texture, dietary restrictions, and ambiance. It also
    tracks confidence scores and sentiment history for each aspect.

    Attributes:
        business_id: Unique identifier for the business
        confidence_scores: Confidence scores for each aspect
        mention_counts: Number of mentions for each aspect
        sentiment_history: History of sentiment scores for each aspect
        price_level: Price level from 0 (very cheap) to 1 (very expensive)
        price_value_ratio: Value for money ratio
        price_category: Price category ($, $$, $$$, $$$$)
        dining_style: Dining style (casual, fine_dining, fast_food, bistro)
        [Various aspect scores]: Scores for different aspects (-1.0 to 1.0)
    """

    def __init__(self, business_id: str) -> None:
        """Initialize the profile.

        Args:
            business_id: Unique identifier for the business
        """
        self.business_id = business_id
        self.confidence_scores: Dict[str, float] = {}
        self.mention_counts: Dict[str, int] = {}
        self.sentiment_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        # Price dimensions
        self.price_level: float = 0.0
        self.price_value_ratio: float = 0.0
        self.price_category: str = "$"

        # Dining style
        self.dining_style: str = "casual"

        # Dietary certifications
        self.kosher_certified: bool = False
        self.halal_certified: bool = False

        # Initialize all aspects from mapping
        for category, aspects in ASPECT_MAPPING.items():
            for aspect in aspects:
                setattr(self, aspect, 0.0)
                self.confidence_scores[aspect] = 0.0
                self.mention_counts[aspect] = 0
                self.sentiment_history[aspect] = []

        # Initialize price and dining style attributes
        price_attrs = ["price_level", "price_value_ratio"]
        for attr in price_attrs:
            if attr not in self.sentiment_history:
                self.sentiment_history[attr] = []
                self.confidence_scores[attr] = 0.0
                self.mention_counts[attr] = 0

    def update_aspect(
        self, aspect: str, sentiment: float, confidence: float = 0.5
    ) -> None:
        """Update an aspect's score with a new sentiment value.

        Args:
            aspect: The aspect to update
            sentiment: The sentiment score (-1.0 to 1.0)
            confidence: Confidence in the sentiment (0.0 to 1.0)
        """
        if not hasattr(self, aspect):
            logger.warning(f"Unknown aspect '{aspect}' - skipping update")
            return

        # Add to sentiment history
        self.sentiment_history[aspect].append((sentiment, confidence))
        self.mention_counts[aspect] = self.mention_counts.get(aspect, 0) + 1

        # Calculate weighted average based on confidence
        total_weighted_sentiment = 0.0
        total_confidence = 0.0

        for sent, conf in self.sentiment_history[aspect]:
            total_weighted_sentiment += sent * conf
            total_confidence += conf

        if total_confidence > 0:
            # Update aspect score
            new_score = total_weighted_sentiment / total_confidence

            # Special handling for price level and dining style
            if aspect == "price_level":
                self._update_price_level(new_score)
            elif aspect == "dining_style":
                self._update_dining_style(new_score, confidence)
            else:
                setattr(self, aspect, new_score)

            # Update confidence score with boost for multiple mentions
            mention_boost = min(
                CONFIDENCE_BOOST_PER_MENTION * self.mention_counts[aspect],
                MAX_CONFIDENCE_SCORE - total_confidence,
            )
            self.confidence_scores[aspect] = min(
                total_confidence
                / max(
                    MIN_MENTIONS_FOR_RELIABLE_SCORE, len(self.sentiment_history[aspect])
                )
                + mention_boost,
                MAX_CONFIDENCE_SCORE,
            )

    def _update_price_level(self, score: float) -> None:
        """Update price level and related attributes.

        Args:
            score: New price level score (0.0 to 1.0)
        """
        # Update price level
        self.price_level = max(0.0, min(1.0, score))

        # Update price category based on thresholds
        if self.price_level < 0.25:
            self.price_category = "$"
        elif self.price_level < 0.5:
            self.price_category = "$$"
        elif self.price_level < 0.75:
            self.price_category = "$$$"
        else:
            self.price_category = "$$$$"

        # Update value ratio based on quality indicators
        self._update_value_ratio()

    def _update_dining_style(self, score: float, confidence: float) -> None:
        """Update dining style based on score and confidence.

        Args:
            score: Style score (0.0 to 1.0)
            confidence: Confidence in the style
        """
        # Only update if new confidence is higher
        current_confidence = self.confidence_scores.get("dining_style", 0.0)
        if confidence > current_confidence:
            # Map score ranges to styles
            if score > 0.8:
                self.dining_style = "fine_dining"
            elif score > 0.6:
                self.dining_style = "upscale"
            elif score > 0.4:
                self.dining_style = "bistro"
            elif score > 0.2:
                self.dining_style = "casual"
            else:
                self.dining_style = "fast_food"

            self.confidence_scores["dining_style"] = confidence

    def _update_value_ratio(self) -> None:
        """Update the value ratio based on quality indicators."""
        quality_aspects = [
            "taste_quality",
            "service_quality",
            "freshness",
            "authenticity",
            "plating_aesthetics",
            "cleanliness",
        ]

        quality_scores = []
        total_confidence = 0.0

        for aspect in quality_aspects:
            confidence = self.confidence_scores.get(aspect, 0.0)
            if confidence > MIN_CONFIDENCE_THRESHOLD:
                score = getattr(self, aspect)
                quality_scores.append(score * confidence)
                total_confidence += confidence

        if quality_scores and total_confidence > 0:
            # Calculate weighted average quality
            avg_quality = sum(quality_scores) / total_confidence

            # Calculate value ratio (quality relative to price)
            if self.price_level > 0:
                self.price_value_ratio = avg_quality / self.price_level
            else:
                self.price_value_ratio = avg_quality
        else:
            # Default to neutral if no quality indicators
            self.price_value_ratio = 0.5

    def get_significant_aspects(
        self, min_confidence: float = MIN_CONFIDENCE_THRESHOLD
    ) -> Dict[str, Dict[str, Any]]:
        """Get aspects with significant confidence scores.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            Dict mapping categories to their significant aspects
        """
        significant = {}

        for category, aspects in ASPECT_MAPPING.items():
            category_aspects = {}
            for aspect in aspects:
                confidence = self.confidence_scores.get(aspect, 0.0)
                if confidence >= min_confidence:
                    score = getattr(self, aspect)
                    mentions = self.mention_counts.get(aspect, 0)
                    history = self.sentiment_history.get(aspect, [])

                    variance = self._calculate_sentiment_variance(history, score)

                    category_aspects[aspect] = {
                        "score": score,
                        "confidence": confidence,
                        "mentions": mentions,
                        "variance": variance,
                        "mixed_sentiment": variance
                        > MIXED_SENTIMENT_VARIANCE_THRESHOLD,
                    }

            if category_aspects:
                significant[category] = category_aspects

        return significant

    def _calculate_sentiment_variance(
        self, history: List[Tuple[float, float]], mean_score: float
    ) -> float:
        """Calculate variance of sentiment scores.

        Args:
            history: List of (sentiment, confidence) tuples
            mean_score: Mean sentiment score

        Returns:
            float: Variance of sentiment scores
        """
        if len(history) <= 1:
            return 0.0

        squared_diffs = [(s - mean_score) ** 2 for s, _ in history]
        return sum(squared_diffs) / len(history)

    def get_aspect_summary(self, aspect: str) -> Optional[Dict[str, Any]]:
        """Get detailed summary for an aspect.

        Args:
            aspect: Name of the aspect

        Returns:
            Dict with aspect statistics or None if aspect not found
        """
        if not hasattr(self, aspect):
            return None

        score = getattr(self, aspect)
        confidence = self.confidence_scores.get(aspect, 0.0)
        history = self.sentiment_history.get(aspect, [])

        if not history:
            return None

        # Calculate statistics
        sentiments = [s for s, _ in history]
        mean = sum(sentiments) / len(sentiments)
        variance = self._calculate_sentiment_variance(history, mean)

        # Count sentiment distributions
        positive = sum(1 for s in sentiments if s > 0.2)
        negative = sum(1 for s in sentiments if s < -0.2)
        neutral = len(sentiments) - positive - negative

        return {
            "score": score,
            "confidence": confidence,
            "mentions": len(history),
            "variance": variance,
            "mixed_sentiment": variance > MIXED_SENTIMENT_VARIANCE_THRESHOLD,
            "positive_mentions": positive,
            "negative_mentions": negative,
            "neutral_mentions": neutral,
            "sentiment_distribution": {
                "positive": positive / len(history),
                "negative": negative / len(history),
                "neutral": neutral / len(history),
            },
        }

    def get_aspect_weight(self, aspect: str) -> float:
        """Get the weight for an aspect from constants.

        Args:
            aspect: Aspect name

        Returns:
            float: Weight for the aspect
        """
        return ASPECT_WEIGHTS.get(aspect, 1.0)

    def get_weighted_distance(self, other: "TasteProfile") -> float:
        """Calculate weighted Euclidean distance between profiles.

        Args:
            other: Another TasteProfile to compare with

        Returns:
            float: Weighted distance score (lower means more similar)
        """
        squared_diff_sum = 0.0
        total_weight = 0.0

        for aspect, weight in ASPECT_WEIGHTS.items():
            # Skip aspects with low confidence in either profile
            if (
                self.confidence_scores.get(aspect, 0) < MIN_CONFIDENCE_THRESHOLD
                or other.confidence_scores.get(aspect, 0) < MIN_CONFIDENCE_THRESHOLD
            ):
                continue

            value1 = getattr(self, aspect)
            value2 = getattr(other, aspect)

            squared_diff_sum += weight * ((value1 - value2) ** 2)
            total_weight += weight

        if total_weight == 0:
            return float("inf")

        return (squared_diff_sum / total_weight) ** 0.5

    def to_dict(self) -> Dict[str, Union[float, str, bool]]:
        """Convert profile to dictionary.

        Returns:
            Dict containing all profile attributes
        """
        return {
            k: v
            for k, v in vars(self).items()
            if isinstance(v, (float, str, bool)) and k not in {"confidence_scores"}
        }
