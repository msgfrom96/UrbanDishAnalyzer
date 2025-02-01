"""Module for managing restaurant taste profiles."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nltk.tokenize import sent_tokenize

from ...constants.thresholds import (
    MIN_CONFIDENCE_THRESHOLD,
)
from .constants import ASPECT_MAPPING, ASPECT_WEIGHTS
from .extraction import extract_aspects_and_sentiments, map_aspects_to_profile

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TasteProfile:
    """Represents a restaurant's taste profile.

    This class encapsulates all aspects of a restaurant's profile, including:
    - Taste qualities (e.g., spiciness, sweetness)
    - Service aspects
    - Ambiance characteristics
    - Price and value metrics
    - Dining style
    """

    business_id: str
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    mention_counts: Dict[str, int] = field(default_factory=dict)
    sentiment_history: Dict[str, list] = field(default_factory=lambda: {})

    # Price dimensions
    price_level: float = 0.0  # Scale from 0 (very cheap) to 1 (very expensive)
    price_value_ratio: float = 0.0  # Value for money
    price_category: str = "$$"  # $, $$, $$$, $$$$

    # Dining style
    dining_style: str = "casual"  # casual, fine_dining, fast_food, bistro

    # Dietary restrictions (boolean with strong separation)
    kosher_certified: bool = False  # Strict kosher certification
    halal_certified: bool = False  # Strict halal certification

    # Taste dimensions
    sweet: float = 0.0
    salty: float = 0.0
    spicy: float = 0.0
    savory: float = 0.0
    bitter: float = 0.0
    sour: float = 0.0

    # Texture dimensions
    crunchiness: float = 0.0
    smoothness: float = 0.0
    chewiness: float = 0.0
    creaminess: float = 0.0
    firmness: float = 0.0
    juiciness: float = 0.0
    softness: float = 0.0

    # Other dietary attributes (float for degree of accommodation)
    gluten_free: float = 0.0
    dairy_free: float = 0.0
    vegan: float = 0.0
    vegetarian: float = 0.0
    nut_free: float = 0.0
    shellfish_free: float = 0.0
    kosher: float = 0.0
    halal: float = 0.0

    # Health dimensions
    health_consciousness: float = 0.0
    organic: float = 0.0

    # Ambiance dimensions
    lighting_quality: float = 0.0
    noise_level: float = 0.0
    seating_comfort: float = 0.0
    plating_aesthetics: float = 0.0
    portion_size: float = 0.0
    service_speed: float = 0.0
    cleanliness: float = 0.0
    temperature: float = 0.0
    accessibility: float = 0.0
    friendly_staff: float = 0.0
    family_friendly: float = 0.0
    romantic_ambiance: float = 0.0

    # New attributes
    overall_score: float = 0.0
    review_count: int = 0
    taste_quality: float = 0.0
    spiciness: float = 0.0
    umami: float = 0.0
    complexity: float = 0.0
    service_quality: float = 0.0
    speed: float = 0.0
    friendliness: float = 0.0
    professionalism: float = 0.0
    ambiance: float = 0.0
    decor: float = 0.0
    cuisine_type: str = "unknown"
    boolean_attributes: Dict[str, bool] = field(default_factory=lambda: {})

    def __post_init__(self):
        """Initialize all aspect scores to 0.0."""
        # Initialize all aspects from mapping
        for category, aspects in ASPECT_MAPPING.items():
            for aspect in aspects:
                setattr(self, aspect, 0.0)
                self.confidence_scores[aspect] = 0.0
                self.mention_counts[aspect] = 0
                self.sentiment_history[aspect] = []

        # Initialize price and dining style attributes
        price_attrs = [
            "price_level",
            "price_value_ratio",
            "price_category",
            "dining_style",
        ]
        for attr in price_attrs:
            if attr not in self.sentiment_history:
                self.sentiment_history[attr] = []
                self.confidence_scores[attr] = 0.0
                self.mention_counts[attr] = 0

        # Initialize certification attributes
        cert_attrs = ["kosher_certified", "halal_certified"]
        for attr in cert_attrs:
            if attr not in self.sentiment_history:
                self.sentiment_history[attr] = []
                self.confidence_scores[attr] = 0.0
                self.mention_counts[attr] = 0

        # Initialize new attributes
        for attr in [
            "taste_quality",
            "spiciness",
            "sweetness",
            "saltiness",
            "umami",
            "complexity",
            "service_quality",
            "speed",
            "friendliness",
            "professionalism",
            "ambiance",
            "decor",
        ]:
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

            # Special handling for price level
            if aspect == "price_level":
                # Normalize to 0-1 range
                new_score = (new_score + 1) / 2
                # Update price category
                if new_score < 0.25:
                    self.price_category = "$"
                elif new_score < 0.5:
                    self.price_category = "$$"
                elif new_score < 0.75:
                    self.price_category = "$$$"
                else:
                    self.price_category = "$$$$"

                # Calculate value ratio based on quality indicators
                quality_aspects = [
                    "freshness",
                    "authenticity",
                    "plating_aesthetics",
                    "cleanliness",
                ]
                quality_scores = []
                for qa in quality_aspects:
                    if hasattr(self, qa) and self.confidence_scores.get(qa, 0) > 0.3:
                        quality_scores.append(getattr(self, qa))

                if quality_scores:
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    self.price_value_ratio = avg_quality / new_score

            setattr(self, aspect, total_weighted_sentiment / total_confidence)
            # Update confidence score (normalized by number of mentions)
            self.confidence_scores[aspect] = min(
                total_confidence / max(3.0, len(self.sentiment_history[aspect])), 1.0
            )

    def get_significant_aspects(
        self, min_confidence: float = MIN_CONFIDENCE_THRESHOLD
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Get aspects with significant scores.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary mapping categories to lists of (aspect, score) tuples
        """
        significant = {
            "taste": [],
            "service": [],
            "ambiance": [],
            "value": [],
        }

        # Check taste aspects
        taste_aspects = [
            "taste_quality",
            "spiciness",
            "sweetness",
            "saltiness",
            "umami",
            "complexity",
        ]
        for aspect in taste_aspects:
            if (
                hasattr(self, aspect)
                and self.confidence_scores.get(aspect, 0.0) >= min_confidence
            ):
                score = getattr(self, aspect)
                if abs(score - 0.5) >= 0.2:  # Significant deviation from neutral
                    significant["taste"].append((aspect, score))

        # Check service aspects
        service_aspects = [
            "service_quality",
            "speed",
            "friendliness",
            "professionalism",
        ]
        for aspect in service_aspects:
            if (
                hasattr(self, aspect)
                and self.confidence_scores.get(aspect, 0.0) >= min_confidence
            ):
                score = getattr(self, aspect)
                if abs(score - 0.5) >= 0.2:
                    significant["service"].append((aspect, score))

        # Check ambiance aspects
        ambiance_aspects = [
            "ambiance",
            "noise_level",
            "lighting_quality",
            "decor",
            "cleanliness",
        ]
        for aspect in ambiance_aspects:
            if (
                hasattr(self, aspect)
                and self.confidence_scores.get(aspect, 0.0) >= min_confidence
            ):
                score = getattr(self, aspect)
                if abs(score - 0.5) >= 0.2:
                    significant["ambiance"].append((aspect, score))

        # Check value aspects
        value_aspects = ["price_level", "price_value_ratio"]
        for aspect in value_aspects:
            if (
                hasattr(self, aspect)
                and self.confidence_scores.get(aspect, 0.0) >= min_confidence
            ):
                score = getattr(self, aspect)
                if abs(score - 0.5) >= 0.2:
                    significant["value"].append((aspect, score))

        return {k: v for k, v in significant.items() if v}

    def get_mixed_sentiment_aspects(self, threshold: float = 0.3) -> Dict[str, float]:
        """Get aspects that show mixed sentiments.

        Args:
            threshold: Variance threshold for considering sentiments mixed

        Returns:
            Dict mapping aspect names to their sentiment variances
        """
        mixed_aspects = {}

        for aspect, history in self.sentiment_history.items():
            if len(history) > 1:
                sentiments = [s for s, _ in history]
                mean = sum(sentiments) / len(sentiments)
                variance = sum((s - mean) ** 2 for s in sentiments) / len(sentiments)

                if variance > threshold:
                    mixed_aspects[aspect] = variance

        return mixed_aspects

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
        variance = (
            sum((s - mean) ** 2 for s in sentiments) / len(sentiments)
            if len(sentiments) > 1
            else 0.0
        )

        # Count positive and negative mentions
        positive = sum(1 for s in sentiments if s > 0.2)
        negative = sum(1 for s in sentiments if s < -0.2)
        neutral = len(sentiments) - positive - negative

        return {
            "score": score,
            "confidence": confidence,
            "mentions": len(history),
            "variance": variance,
            "mixed_sentiment": variance > 0.3,
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
        """Get the weight for an aspect from constants."""
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

        for aspect in ASPECT_WEIGHTS:
            # Skip aspects with low confidence in either profile
            if (
                self.confidence_scores.get(aspect, 0) < 0.3
                or other.confidence_scores.get(aspect, 0) < 0.3
            ):
                continue

            weight = self.get_aspect_weight(aspect)
            value1 = getattr(self, aspect)
            value2 = getattr(other, aspect)

            squared_diff_sum += weight * ((value1 - value2) ** 2)
            total_weight += weight

        if total_weight == 0:
            return float("inf")

        return (squared_diff_sum / total_weight) ** 0.5

    def update_scores(self, review_text: str, absa_pipeline: Any):
        """
        Update taste profile scores based on review text analysis.

        Args:
            review_text: The review text to analyze
            absa_pipeline: The aspect-based sentiment analysis pipeline
        """
        logger.info(f"Updating scores based on review: {review_text}")

        # Extract aspects and sentiments using ABSA
        aspects_sentiments = extract_aspects_and_sentiments(review_text, absa_pipeline)
        logger.debug(f"Extracted aspects and sentiments: {aspects_sentiments}")

        if not aspects_sentiments:
            logger.info("No aspects extracted from the review")
            return

        # Map to profile dimensions
        profile_updates = map_aspects_to_profile(aspects_sentiments)
        logger.debug(f"Mapped profile updates: {profile_updates}")

        # Split review into sentences for intensity adjustment
        sentences = sent_tokenize(review_text)

        # Update scores with exponential moving average and intensity adjustment
        for aspect, weighted_score in profile_updates.items():
            if not hasattr(self, aspect):
                logger.warning(
                    f"Aspect '{aspect}' not found in TasteProfile attributes"
                )
                continue

            current = getattr(self, aspect)
            confidence = self.confidence_scores[aspect]

            # Find the sentence containing the aspect for intensity adjustment
            relevant_sentences = [s for s in sentences if aspect.lower() in s.lower()]
            if relevant_sentences:
                is_boolean = aspect in self.boolean_attributes
                adjusted_score = adjust_intensity(
                    relevant_sentences[0], weighted_score, is_boolean
                )
            else:
                adjusted_score = weighted_score

            # Handle boolean attributes differently
            if aspect in self.boolean_attributes:
                if adjusted_score > 0.5:  # Clear indication of presence
                    new_value = 1.0
                    new_confidence = (
                        1.0  # Boolean attributes have full confidence when detected
                    )
                else:
                    new_value = current
                    new_confidence = confidence
            else:
                # Calculate new confidence score for non-boolean attributes
                mention_boost = 0.3  # Larger confidence boost per mention
                new_confidence = min(confidence + mention_boost, 1.0)

                # Use confidence-weighted update
                alpha = (1 / (self.review_count + 1)) * new_confidence
                new_value = (alpha * adjusted_score) + ((1 - alpha) * current)
                new_value = min(max(new_value, 0.0), 1.0)

            setattr(self, aspect, new_value)
            self.confidence_scores[aspect] = new_confidence
            logger.info(
                f"Updated {aspect}: {current:.2f} -> {new_value:.2f} (confidence: {new_confidence:.2f})"
            )

        self.review_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary.

        Returns:
            Dictionary representation of the profile
        """
        base_dict = {
            "overall_score": self.overall_score,
            "review_count": self.review_count,
            "taste_quality": self.taste_quality,
            "service_quality": self.service_quality,
            "ambiance": self.ambiance,
            "value_ratio": self.price_value_ratio,
            "price_level": self.price_level,
            "price_category": self.price_category,
            "dining_style": self.dining_style,
            "cuisine_type": self.cuisine_type,
        }

        # Add all other attributes
        for key, value in vars(self).items():
            if key not in base_dict:
                base_dict[key] = value

        return base_dict
