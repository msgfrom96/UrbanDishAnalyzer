"""Module for detecting restaurant hotspots based on geographical and taste profile data."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import folium
import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ..core.analyzer import TasteProfileAnalyzer
from ..core.profile import TasteProfile
from ..utils import calculate_distance

logger = logging.getLogger(__name__)


class Hotspot:
    """Represents a potential restaurant hotspot."""

    def __init__(
        self,
        location: Tuple[float, float],
        score: float,
        nearby_businesses: List[str],
        profile: Optional[TasteProfile] = None,
    ) -> None:
        """Initialize a hotspot.

        Args:
            location: (latitude, longitude) of the hotspot
            score: Hotspot score (0-1)
            nearby_businesses: List of nearby business IDs
            profile: Optional taste profile for the area
        """
        self.location = location
        self.score = score
        self.nearby_businesses = nearby_businesses
        self.profile = profile


class HotspotDetector:
    """Detector for potential restaurant hotspots."""

    def __init__(self, analyzer: Optional[TasteProfileAnalyzer] = None) -> None:
        """Initialize the hotspot detector.

        Args:
            analyzer: Optional taste profile analyzer
        """
        self.analyzer = analyzer
        self.locations: Dict[str, Tuple[float, float]] = {}
        self.profiles: Dict[str, TasteProfile] = {}

    def add_business(
        self,
        business_id: str,
        location: Tuple[float, float],
        profile: Optional[TasteProfile] = None,
    ) -> None:
        """Add a business to the detector.

        Args:
            business_id: Business ID
            location: (latitude, longitude) of the business
            profile: Optional taste profile
        """
        self.locations[business_id] = location
        if profile is not None:
            self.profiles[business_id] = profile

    def detect_hotspots(
        self, radius_km: float = 1.0, min_businesses: int = 3
    ) -> List[Hotspot]:
        """Detect potential hotspots.

        Args:
            radius_km: Radius in kilometers to consider
            min_businesses: Minimum number of businesses for a hotspot

        Returns:
            List of detected hotspots
        """
        hotspots: List[Hotspot] = []
        business_clusters: Dict[str, List[str]] = defaultdict(list)

        # Group businesses by proximity
        for business_id, location in self.locations.items():
            cluster_key = f"{location[0]:.3f},{location[1]:.3f}"
            business_clusters[cluster_key].append(business_id)

            # Find nearby businesses
            for other_id, other_loc in self.locations.items():
                if other_id != business_id:
                    distance = calculate_distance(location, other_loc)
                    if distance <= radius_km:
                        business_clusters[cluster_key].append(other_id)

        # Analyze clusters
        for center_str, nearby in business_clusters.items():
            if len(nearby) >= min_businesses:
                lat, lon = map(float, center_str.split(","))
                center = (lat, lon)

                # Calculate hotspot score based on:
                # 1. Number of nearby businesses
                # 2. Average ratings (if profiles available)
                # 3. Diversity of cuisine types
                base_score = min(1.0, len(nearby) / (min_businesses * 2))
                profile_score = self._calculate_profile_score(nearby)
                diversity_score = self._calculate_diversity_score(nearby)

                # Combine scores with weights
                score = base_score * 0.4 + profile_score * 0.4 + diversity_score * 0.2

                # Create aggregate profile for the area
                area_profile = self._create_area_profile(nearby)

                hotspots.append(Hotspot(center, score, nearby, area_profile))

        return sorted(hotspots, key=lambda x: x.score, reverse=True)

    def _calculate_profile_score(self, business_ids: List[str]) -> float:
        """Calculate score based on business profiles.

        Args:
            business_ids: List of business IDs

        Returns:
            Profile-based score (0-1)
        """
        if not self.profiles:
            return 0.5

        scores = []
        for business_id in business_ids:
            if business_id in self.profiles:
                profile = self.profiles[business_id]
                # Use average of key metrics
                if hasattr(profile, "overall_score"):
                    scores.append(profile.overall_score)

        return sum(scores) / len(scores) if scores else 0.5

    def _calculate_diversity_score(self, business_ids: List[str]) -> float:
        """Calculate score based on cuisine diversity.

        Args:
            business_ids: List of business IDs

        Returns:
            Diversity score (0-1)
        """
        if not self.profiles:
            return 0.5

        cuisine_types: Dict[str, int] = defaultdict(int)
        for business_id in business_ids:
            if business_id in self.profiles:
                profile = self.profiles[business_id]
                if hasattr(profile, "cuisine_type"):
                    cuisine_types[profile.cuisine_type] += 1

        # Calculate diversity using number of unique cuisines
        unique_cuisines = len(cuisine_types)
        return min(1.0, unique_cuisines / 5)  # Cap at 5 unique cuisines

    def _create_area_profile(self, business_ids: List[str]) -> Optional[TasteProfile]:
        """Create aggregate profile for an area.

        Args:
            business_ids: List of business IDs

        Returns:
            Aggregate taste profile or None if no profiles available
        """
        if not self.profiles or not self.analyzer:
            return None

        # Collect all profiles
        area_profiles = []
        for business_id in business_ids:
            if business_id in self.profiles:
                area_profiles.append(self.profiles[business_id])

        if not area_profiles:
            return None

        # Use analyzer to create aggregate profile
        return self.analyzer.create_aggregate_profile(area_profiles)

    def generate_map(
        self, hotspots: List[Hotspot], center: Optional[Tuple[float, float]] = None
    ) -> folium.Map:
        """Generate an interactive map of detected hotspots.

        Args:
            hotspots: List of hotspot objects from detect_hotspots()
            center: Optional map center coordinates

        Returns:
            Folium map object
        """
        # Determine map center
        if center is None and hotspots:
            center = hotspots[0].location
        elif center is None:
            center = (0, 0)

        # Create base map
        m = folium.Map(location=center, zoom_start=13)

        # Add hotspots
        for i, hotspot in enumerate(hotspots, 1):
            # Add circle for hotspot area
            folium.Circle(
                location=hotspot.location,
                radius=1000 * hotspot.score,  # Convert score to meters
                color="red",
                fill=True,
                popup=f"Hotspot {i}",
            ).add_to(m)

            # Add markers for restaurants
            for bid in hotspot.nearby_businesses:
                lat, lon = self.locations[bid]
                profile = self.profiles[bid]

                # Create popup content
                popup_html = f"""
                <b>{bid}</b><br>
                Price: {profile.price_category}<br>
                Style: {profile.dining_style}<br>
                """

                folium.Marker(
                    location=(lat, lon),
                    popup=popup_html,
                    icon=folium.Icon(color="blue", icon="info-sign"),
                ).add_to(m)

        return m
