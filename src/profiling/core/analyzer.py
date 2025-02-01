"""Core module for taste profile analysis."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..analyzer.aspect_extractor import AspectExtractor
from ..constants.aspects import ASPECT_MAPPING
from ..constants.thresholds import (
    MAX_CLUSTERS,
    MIN_CLUSTERS,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_MENTIONS_FOR_RELIABLE_SCORE,
    SILHOUETTE_THRESHOLD,
)
from .profile import TasteProfile

# Configure logging
logger = logging.getLogger(__name__)


class TasteProfileAnalyzer:
    """Class for analyzing and clustering taste profiles.

    This class handles the analysis of reviews to build taste profiles for
    businesses, and provides methods for clustering similar businesses based
    on their profiles.

    Attributes:
        profiles: Dictionary mapping business IDs to their taste profiles
    """

    def __init__(self) -> None:
        """Initialize the analyzer."""
        self.profiles: Dict[str, TasteProfile] = {}
        self.extractor = AspectExtractor()
        logger.info("Initialized TasteProfileAnalyzer")

    def analyze_review(self, business_id: str, review_text: str) -> None:
        """Analyze a review and update the business's taste profile.

        Args:
            business_id: Unique identifier for the business
            review_text: Text of the review to analyze
        """
        try:
            # Get or create profile
            if business_id not in self.profiles:
                self.profiles[business_id] = TasteProfile(business_id=business_id)

            # Extract aspects using our improved extractor
            aspects = self.extractor.extract_aspects_and_sentiments(review_text)

            # Update profile with extracted aspects
            profile = self.profiles[business_id]

            # Update dining style
            if "dining_style" in aspects:
                profile.dining_style = aspects["dining_style"]
                profile.confidence_scores["dining_style"] = aspects["style_confidence"]

            # Update price information
            if "price_level" in aspects:
                profile.price_level = aspects["price_level"]
                profile.price_value_ratio = aspects["value_ratio"]
                profile.price_category = aspects["price_category"]

            # Update other aspects
            for category in ["Taste", "Food", "Ambiance", "Texture"]:
                for aspect, (score, confidence) in aspects.get(
                    category.lower(), {}
                ).items():
                    setattr(profile, aspect, score)
                    profile.confidence_scores[aspect] = confidence
                    profile.mention_counts[aspect] = (
                        profile.mention_counts.get(aspect, 0) + 1
                    )

                    # Track sentiment history
                    if aspect not in profile.sentiment_history:
                        profile.sentiment_history[aspect] = []
                    profile.sentiment_history[aspect].append((score, confidence))

        except Exception as e:
            logger.error(f"Error analyzing review: {str(e)}")
            raise

    def get_business_profile(self, business_id: str) -> Optional[TasteProfile]:
        """Get the taste profile for a specific business.

        Args:
            business_id: Unique identifier for the business

        Returns:
            TasteProfile if found, None otherwise
        """
        return self.profiles.get(business_id)

    def get_aggregate_profile(self, business_ids: List[str]) -> Optional[TasteProfile]:
        """Get an aggregate taste profile for multiple businesses.

        Args:
            business_ids: List of business IDs to aggregate

        Returns:
            Aggregated TasteProfile if businesses found, None otherwise
        """
        if not business_ids or not all(bid in self.profiles for bid in business_ids):
            return None

        # Create aggregate profile
        aggregate = TasteProfile(business_id="aggregate")

        # Track aspect statistics
        aspect_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_score": 0.0,
                "total_confidence": 0.0,
                "mentions": 0,
                "businesses": 0,
                "variances": [],
            }
        )

        # Collect statistics
        for bid in business_ids:
            profile = self.profiles[bid]
            for aspect in ASPECT_MAPPING.keys():
                if profile.confidence_scores.get(aspect, 0) > MIN_CONFIDENCE_THRESHOLD:
                    stats = aspect_stats[aspect]
                    score = getattr(profile, aspect)
                    confidence = profile.confidence_scores[aspect]

                    stats["total_score"] += float(score * confidence)
                    stats["total_confidence"] += float(confidence)
                    stats["mentions"] += profile.mention_counts.get(aspect, 0)
                    stats["businesses"] += 1

                    # Track sentiment variance
                    if aspect in profile.sentiment_history:
                        history = profile.sentiment_history[aspect]
                        if len(history) > 1:
                            sentiments = [s for s, _ in history]
                            variance = float(np.var(sentiments))
                            stats["variances"].append(variance)

        # Calculate aggregate scores
        for aspect, stats in aspect_stats.items():
            if stats["total_confidence"] > 0:
                # Set aggregate score
                score = stats["total_score"] / stats["total_confidence"]
                setattr(aggregate, aspect, score)

                # Set confidence based on number of businesses and mentions
                confidence = min(
                    stats["total_confidence"] / len(business_ids),
                    stats["businesses"] / len(business_ids),
                )
                aggregate.confidence_scores[aspect] = confidence

                # Track mention count
                aggregate.mention_counts[aspect] = stats["mentions"]

                # Log mixed sentiments
                if stats["variances"]:
                    avg_variance = float(np.mean(stats["variances"]))
                    if avg_variance > 0.3:
                        logger.info(
                            f"Aspect '{aspect}' shows mixed sentiments "
                            f"(variance: {avg_variance:.2f})"
                        )

        return aggregate

    def _extract_features(
        self, min_confidence: float = MIN_CONFIDENCE_THRESHOLD
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Extract feature matrix for clustering.

        Args:
            min_confidence: Minimum confidence score to include an aspect

        Returns:
            Tuple of:
            - Feature matrix (n_samples, n_features)
            - List of business IDs
            - List of feature names
        """
        # Get significant aspects
        feature_aspects: Set[str] = set()
        for profile in self.profiles.values():
            significant = profile.get_significant_aspects(min_confidence)
            for category_aspects in significant.values():
                feature_aspects.update(category_aspects.keys())

        feature_aspects_list = sorted(feature_aspects)
        business_ids = []
        feature_arrays = []

        # Build feature matrix
        for bid, profile in self.profiles.items():
            features = []
            include_business = False

            for aspect in feature_aspects_list:
                confidence = profile.confidence_scores.get(aspect, 0)
                if confidence >= min_confidence:
                    score = getattr(profile, aspect)
                    features.append(score)
                    include_business = True
                else:
                    features.append(0.0)

            if include_business and features:
                feature_arrays.append(features)
                business_ids.append(bid)

        if not feature_arrays:
            return np.array([]), [], []

        return np.array(feature_arrays), business_ids, feature_aspects_list

    def cluster_businesses(
        self,
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
        min_clusters: int = MIN_CLUSTERS,
        max_clusters: int = MAX_CLUSTERS,
    ) -> Dict[str, Dict[str, Any]]:
        """Cluster businesses based on their taste profiles.

        Args:
            min_confidence: Minimum confidence score to include an aspect
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try

        Returns:
            Dictionary mapping business IDs to cluster information
        """
        if len(self.profiles) < min_clusters:
            logger.warning("Not enough businesses for clustering")
            return {}

        # Extract features
        X, business_ids, feature_names = self._extract_features(min_confidence)

        if len(X) == 0:
            logger.warning("No features with sufficient confidence for clustering")
            return {}

        # Scale features
        X_scaled = StandardScaler().fit_transform(X)

        # Find optimal number of clusters
        best_score = -1
        best_kmeans = None

        for n_clusters in range(min_clusters, min(max_clusters + 1, len(business_ids))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X_scaled)

            if len(np.unique(kmeans.labels_)) > 1:
                score = silhouette_score(X_scaled, kmeans.labels_)
                logger.info(
                    f"Tried {n_clusters} clusters, silhouette score: {score:.3f}"
                )

                if score > best_score:
                    best_score = score
                    best_kmeans = kmeans

        if best_kmeans is None or best_score < SILHOUETTE_THRESHOLD:
            logger.warning(
                f"Clustering quality too low (score: {best_score:.3f}, "
                f"threshold: {SILHOUETTE_THRESHOLD})"
            )
            return {}

        # Create cluster assignments
        clusters = {}
        for bid, label in zip(business_ids, best_kmeans.labels_):
            profile = self.profiles[bid]

            # Determine cluster characteristics
            cluster_center = best_kmeans.cluster_centers_[label]
            significant_features = []

            # Find distinctive features
            for idx, value in enumerate(cluster_center):
                feature_name = feature_names[idx]
                feature_std = np.std(X_scaled[:, idx])
                if abs(value) > 0.5 * feature_std:
                    significant_features.append(
                        (feature_name, value, abs(value / feature_std))
                    )

            # Sort by significance
            significant_features.sort(key=lambda x: abs(x[2]), reverse=True)

            clusters[bid] = {
                "cluster_label": f"cluster_{label}",
                "price_category": profile.price_category,
                "dining_style": profile.dining_style,
                "distinctive_features": [
                    (name, score) for name, score, _ in significant_features[:5]
                ],
            }

            # Add dietary information if present
            if profile.kosher_certified:
                clusters[bid]["dietary"] = "kosher"
            elif profile.halal_certified:
                clusters[bid]["dietary"] = "halal"
            else:
                clusters[bid]["dietary"] = "none"

        return clusters

    def get_language(self, text: str) -> str:
        """Get the language of a text.

        Args:
            text: Text to analyze

        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        return detect(text)
