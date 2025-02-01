"""Module for location-based clustering of restaurants."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import numpy as np
from folium import plugins
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ..core.analyzer import TasteProfileAnalyzer
from ..core.profile import TasteProfile
from ..logging import get_logger

logger = get_logger(__name__)


class LocationClusterer:
    """Class for clustering restaurants based on location and taste profiles."""

    def __init__(self, analyzer: Optional[TasteProfileAnalyzer] = None):
        """Initialize the clusterer.

        Args:
            analyzer: Optional analyzer instance (creates new one if None)
        """
        self.analyzer = analyzer or TasteProfileAnalyzer()
        self.locations: Dict[str, Tuple[float, float]] = {}
        self.profiles: Dict[str, TasteProfile] = {}

    def add_restaurant(
        self, business_id: str, latitude: float, longitude: float, reviews: List[str]
    ) -> None:
        """Add a restaurant with its location and reviews.

        Args:
            business_id: Business ID
            latitude: Restaurant's latitude
            longitude: Restaurant's longitude
            reviews: List of review texts
        """
        # Store location
        self.locations[business_id] = (latitude, longitude)

        # Process reviews
        for review in reviews:
            self.analyzer.analyze_review(business_id, review)

        # Store profile
        self.profiles[business_id] = self.analyzer.get_business_profile(business_id)

    def cluster_by_location(
        self, max_distance: float = 1.0, min_samples: int = 3
    ) -> Dict[str, List[str]]:
        """Cluster restaurants based on geographical proximity.

        Args:
            max_distance: Maximum distance in kilometers
            min_samples: Minimum samples per cluster

        Returns:
            Dictionary mapping cluster labels to business IDs
        """
        if len(self.locations) < min_samples:
            return {"cluster_0": list(self.locations.keys())}

        # Convert locations to array
        coords = np.array([[lat, lon] for lat, lon in self.locations.values()])

        # Scale coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)

        # Perform clustering
        clustering = DBSCAN(
            eps=max_distance / 111.0,  # Convert km to degrees
            min_samples=min_samples,
            metric="haversine",
        ).fit(coords_scaled)

        # Group businesses by cluster
        clusters: Dict[str, List[str]] = {}
        for i, label in enumerate(clustering.labels_):
            cluster_name = f"cluster_{label}"
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(list(self.locations.keys())[i])

        return clusters

    def cluster_by_profile(self, min_confidence: float = 0.3) -> Dict[str, List[str]]:
        """Cluster restaurants based on taste profiles.

        Args:
            min_confidence: Minimum confidence score

        Returns:
            Dictionary mapping cluster labels to business IDs
        """
        # Get clusters from analyzer
        cluster_info = self.analyzer.cluster_businesses(min_confidence=min_confidence)

        # Group businesses by cluster
        clusters: Dict[str, List[str]] = {}
        for bid, info in cluster_info.items():
            label = info["cluster_label"]
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(bid)

        return clusters

    def cluster_combined(
        self,
        max_distance: float = 1.0,
        min_samples: int = 3,
        location_weight: float = 0.5,
    ) -> Dict[str, List[str]]:
        """Cluster restaurants using both location and taste profiles.

        Args:
            max_distance: Maximum distance in kilometers
            min_samples: Minimum samples per cluster
            location_weight: Weight for location vs taste (0.0-1.0)

        Returns:
            Dictionary mapping cluster labels to business IDs
        """
        if len(self.locations) < min_samples:
            return {"cluster_0": list(self.locations.keys())}

        # Get location features
        coords = np.array([[lat, lon] for lat, lon in self.locations.values()])

        # Get profile features
        profile_features = []
        for bid in self.locations.keys():
            if bid in self.profiles:
                profile = self.profiles[bid]
                features = [
                    getattr(profile, attr, 0)
                    for attr in [
                        "taste_quality",
                        "service_quality",
                        "ambiance",
                        "value_ratio",
                    ]
                ]
                profile_features.append(features)
            else:
                profile_features.append([0] * 4)

        profile_features = np.array(profile_features)

        # Scale features
        scaler_loc = StandardScaler()
        scaler_prof = StandardScaler()

        coords_scaled = scaler_loc.fit_transform(coords)
        profile_scaled = scaler_prof.fit_transform(profile_features)

        # Combine features with weights
        combined_features = np.hstack(
            [coords_scaled * location_weight, profile_scaled * (1 - location_weight)]
        )

        # Perform clustering
        clustering = DBSCAN(
            eps=max_distance / 111.0, min_samples=min_samples, metric="euclidean"
        ).fit(combined_features)

        # Group businesses by cluster
        clusters: Dict[str, List[str]] = {}
        for i, label in enumerate(clustering.labels_):
            cluster_name = f"cluster_{label}"
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(list(self.locations.keys())[i])

        return clusters

    def analyze_cluster(self, cluster_businesses: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of a cluster.

        Args:
            cluster_businesses: List of business IDs in cluster

        Returns:
            Dictionary containing cluster analysis:
            - center: (latitude, longitude)
            - radius: radius in kilometers
            - dominant_aspects: list of (aspect, score) tuples
            - price_distribution: price category distribution
            - dining_styles: dining style distribution
        """
        if not cluster_businesses:
            return {}

        # Get cluster center and radius
        cluster_coords = np.array(
            [self.locations[bid] for bid in cluster_businesses if bid in self.locations]
        )

        if len(cluster_coords) == 0:
            return {}

        center = cluster_coords.mean(axis=0)
        radius = max(geodesic(center, coord).kilometers for coord in cluster_coords)

        # Get aggregate profile
        aggregate = self.analyzer.get_aggregate_profile(cluster_businesses)
        if not aggregate:
            return {"center": tuple(center), "radius": radius}

        # Get dominant aspects
        significant = aggregate.get_significant_aspects()
        dominant_aspects = []
        for category, aspects in significant.items():
            for aspect, data in aspects.items():
                if data["confidence"] > 0.5:
                    dominant_aspects.append((aspect, data["score"]))

        # Analyze price distribution
        price_dist = {}
        style_dist = {}

        for bid in cluster_businesses:
            if bid in self.profiles:
                profile = self.profiles[bid]
                price = profile.price_category
                style = profile.dining_style

                if price not in price_dist:
                    price_dist[price] = 0
                price_dist[price] += 1

                if style not in style_dist:
                    style_dist[style] = 0
                style_dist[style] += 1

        return {
            "center": tuple(center),
            "radius": radius,
            "dominant_aspects": dominant_aspects,
            "price_distribution": price_dist,
            "dining_styles": style_dist,
        }

    def generate_map(self, clusters: Dict[str, List[str]]) -> folium.Map:
        """Generate an interactive map of clusters.

        Args:
            clusters: Dictionary mapping cluster labels to business IDs

        Returns:
            Folium map object
        """
        # Determine map center
        all_coords = np.array([[lat, lon] for lat, lon in self.locations.values()])
        center = all_coords.mean(axis=0)

        # Create base map
        m = folium.Map(location=center, zoom_start=13)

        # Create color map for clusters
        colors = [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "darkred",
            "darkblue",
            "darkgreen",
            "cadetblue",
            "darkpurple",
            "pink",
            "lightblue",
            "lightgreen",
        ]

        # Add clusters
        for i, (label, businesses) in enumerate(clusters.items()):
            color = colors[i % len(colors)]

            # Analyze cluster
            analysis = self.analyze_cluster(businesses)
            if not analysis:
                continue

            # Add circle for cluster area
            folium.Circle(
                location=analysis["center"],
                radius=analysis["radius"] * 1000,  # Convert to meters
                color=color,
                fill=True,
                popup=f"Cluster {label}",
            ).add_to(m)

            # Add markers for restaurants
            for bid in businesses:
                if bid not in self.locations or bid not in self.profiles:
                    continue

                lat, lon = self.locations[bid]
                profile = self.profiles[bid]

                # Create popup content
                popup_html = f"""
                <div style="width:200px">
                    <b>{bid}</b><br>
                    Price: {profile.price_category}<br>
                    Style: {profile.dining_style}<br>
                    <hr>
                    <b>Significant Aspects:</b><br>
                """

                significant = profile.get_significant_aspects()
                for category, aspects in significant.items():
                    popup_html += f"<b>{category.title()}:</b><br>"
                    for aspect, data in aspects.items():
                        if data["confidence"] > 0.3:
                            popup_html += (
                                f"{aspect}: {data['score']:.2f} "
                                f"({data['confidence']:.2f})<br>"
                            )

                popup_html += "</div>"

                folium.Marker(
                    location=(lat, lon),
                    popup=popup_html,
                    icon=folium.Icon(color=color, icon="info-sign"),
                ).add_to(m)

        # Add heatmap layer
        heat_data = [[lat, lon] for lat, lon in self.locations.values()]
        plugins.HeatMap(heat_data).add_to(m)

        return m
