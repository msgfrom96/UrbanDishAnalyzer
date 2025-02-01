"""Module for analyzing taste profiles from reviews."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .constants import (
    ASPECT_MAPPING,
    MAX_CLUSTERS,
    MIN_CLUSTERS,
    MIN_CONFIDENCE_THRESHOLD,
    SILHOUETTE_THRESHOLD,
)
from .extraction import (
    extract_aspects_and_sentiments,
    get_extractor,
    initialize_absa_pipeline,
    map_aspects_to_profile,
)
from .profile import TasteProfile

# Configure logging
logger = logging.getLogger(__name__)


class TasteProfileAnalyzer:
    """Class for analyzing taste profiles from reviews."""

    def __init__(self):
        """Initialize the analyzer."""
        logger.info("Initializing TasteProfileAnalyzer")
        self.profiles = {}
        self.review_counts = {}
        initialize_absa_pipeline()
        self.extractor = get_extractor()
        logger.info("Initialized TasteProfileAnalyzer")

    def analyze_review(self, business_id: str, review_text: str) -> None:
        """Analyze a review and update the business profile.

        Args:
            business_id: ID of the business
            review_text: Text of the review
        """
        try:
            # Get or create profile
            profile = self.get_business_profile(business_id)
            if not profile:
                profile = TasteProfile(business_id=business_id)
                self.profiles[business_id] = profile

            # Extract aspects and sentiments
            aspects_sentiments = extract_aspects_and_sentiments(review_text)

            # Map to profile
            map_aspects_to_profile(aspects_sentiments, profile)

            # Update review count
            self.review_counts[business_id] = self.review_counts.get(business_id, 0) + 1

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

        # Track aspect statistics across businesses
        aspect_stats = defaultdict(
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

                    stats["total_score"] += score * confidence
                    stats["total_confidence"] += confidence
                    stats["mentions"] += profile.mention_counts.get(aspect, 0)
                    stats["businesses"] += 1

                    # Track sentiment variance
                    if aspect in profile.sentiment_history:
                        history = profile.sentiment_history[aspect]
                        if len(history) > 1:
                            sentiments = [s for s, _ in history]
                            variance = np.var(sentiments)
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

                # Calculate average variance if available
                if stats["variances"]:
                    avg_variance = np.mean(stats["variances"])
                    if avg_variance > 0.3:  # Threshold for mixed sentiments
                        logger.info(
                            f"Aspect '{aspect}' shows mixed sentiments across businesses"
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
        # Get all aspects with sufficient confidence in any profile
        feature_aspects = set()
        for profile in self.profiles.values():
            significant = profile.get_significant_aspects(min_confidence)
            for category_aspects in significant.values():
                feature_aspects.update(category_aspects.keys())

        feature_aspects = sorted(feature_aspects)  # Sort for consistency
        business_ids = []
        feature_arrays = []

        # Build feature matrix
        for bid, profile in self.profiles.items():
            features = []
            include_business = False

            for aspect in feature_aspects:
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

        return np.array(feature_arrays), business_ids, list(feature_aspects)

    def cluster_businesses(
        self,
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
        min_clusters: int = MIN_CLUSTERS,
        max_clusters: int = MAX_CLUSTERS,
    ) -> Dict[str, Dict[str, Any]]:
        """Cluster businesses based on their taste profiles with considerations.

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

        # First, separate businesses by dietary restrictions
        kosher_businesses = []
        halal_businesses = []
        regular_businesses = []

        for bid, profile in self.profiles.items():
            if profile.kosher_certified or getattr(profile, "kosher", False):
                kosher_businesses.append(bid)
            elif profile.halal_certified or getattr(profile, "halal", False):
                halal_businesses.append(bid)
            else:
                regular_businesses.append(bid)

        # Log dietary separation
        logger.info("\nDietary Separation:")
        logger.info(f"Kosher certified businesses: {len(kosher_businesses)}")
        logger.info(f"Halal certified businesses: {len(halal_businesses)}")
        logger.info(f"Regular businesses: {len(regular_businesses)}")

        # Extract features for clustering
        X, business_ids, feature_names = self._extract_features(min_confidence)

        if len(X) == 0:
            logger.warning("No features with sufficient confidence for clustering")
            return {}

        # Add price level and dining style as features
        price_features = np.array(
            [
                [self.profiles[bid].price_level, self.profiles[bid].price_value_ratio]
                for bid in business_ids
            ]
        )

        # Convert dining style to one-hot encoding
        dining_styles = ["casual", "fine_dining", "fast_food", "bistro"]
        dining_features = np.zeros((len(business_ids), len(dining_styles)))
        for i, bid in enumerate(business_ids):
            style = self.profiles[bid].dining_style
            if style in dining_styles:
                dining_features[i, dining_styles.index(style)] = 1.0

        # Add dietary features with higher weight
        dietary_features = np.zeros((len(business_ids), 2))  # [kosher, halal]
        for i, bid in enumerate(business_ids):
            profile = self.profiles[bid]
            if bid in kosher_businesses:
                dietary_features[i, 0] = 2.0  # Higher weight for dietary restrictions
            elif bid in halal_businesses:
                dietary_features[i, 1] = 2.0

        # Combine all features
        X_combined = np.hstack(
            [
                X,  # Base features
                price_features * 1.5,  # Price features with higher weight
                dining_features * 1.2,  # Dining style features with medium weight
                dietary_features,  # Already weighted
            ]
        )

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)

        # Try different numbers of clusters
        best_score = -1
        best_labels: List[int] = []  # Change to a list to ensure it's iterable

        logger.info(f"\nTried clustering with {len(feature_names)} features:")
        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            logger.info(f"Tried {n_clusters} clusters, silhouette score: {score:.3f}")

            if score > best_score:
                best_score = score
                best_labels = labels.tolist()  # Convert to list to ensure it's iterable

        if best_score < SILHOUETTE_THRESHOLD:
            logger.warning(
                f"Best silhouette score ({best_score:.3f}) below threshold "
                f"({SILHOUETTE_THRESHOLD})"
            )

        # Create cluster assignments
        cluster_info = {}
        if best_labels:  # Ensure best_labels is not empty
            for bid, label in zip(business_ids, best_labels):
                profile = self.profiles[bid]
                dietary = (
                    "kosher"
                    if bid in kosher_businesses
                    else ("halal" if bid in halal_businesses else "none")
                )

                # Determine cluster label based on dietary restriction and cluster number
                if dietary == "kosher":
                    cluster_label = f"kosher_{label}"
                elif dietary == "halal":
                    cluster_label = f"halal_{label}"
                else:
                    cluster_label = f"regular_{label}"

                cluster_info[bid] = {
                    "cluster_label": cluster_label,
                    "dietary_restriction": dietary,
                    "price_category": profile.price_category,
                    "dining_style": profile.dining_style,
                }

        return cluster_info
