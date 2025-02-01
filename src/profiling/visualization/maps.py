"""Module for generating interactive maps of restaurant data."""

from typing import Dict, List, Optional, Tuple

import folium
import numpy as np
from branca.colormap import LinearColormap
from folium import plugins

from ..core.profile import TasteProfile
from ..logging import get_logger

logger = get_logger(__name__)


class MapVisualizer:
    """Class for creating interactive maps of restaurant data.

    This class provides methods for visualizing restaurant locations,
    clusters, and hotspots on interactive maps.

    Attributes:
        default_location: Default map center coordinates
        default_zoom: Default zoom level
    """

    def __init__(
        self, default_location: Tuple[float, float] = (0, 0), default_zoom: int = 13
    ):
        """Initialize the map visualizer.

        Args:
            default_location: Default map center (lat, lon)
            default_zoom: Default zoom level
        """
        self.default_location = default_location
        self.default_zoom = default_zoom

    def create_base_map(
        self, center: Optional[Tuple[float, float]] = None, zoom: Optional[int] = None
    ) -> folium.Map:
        """Create a base map with optional center and zoom.

        Args:
            center: Map center coordinates (lat, lon)
            zoom: Zoom level

        Returns:
            Folium map object
        """
        return folium.Map(
            location=center or self.default_location,
            zoom_start=zoom or self.default_zoom,
            tiles="cartodbpositron",
        )

    def add_marker(
        self,
        map_obj: folium.Map,
        location: Tuple[float, float],
        popup: str = "",
        color: str = "red",
        icon: str = "info-sign",
    ) -> None:
        """Add a marker to the map.

        Args:
            map_obj: Map object to add marker to
            location: (latitude, longitude) for marker
            popup: Popup text for marker
            color: Marker color
            icon: Icon name from Bootstrap
        """
        folium.Marker(
            location=location,
            popup=popup,
            icon=folium.Icon(color=color, icon=icon),
        ).add_to(map_obj)

    def add_restaurants(
        self,
        m: folium.Map,
        locations: Dict[str, Tuple[float, float]],
        profiles: Dict[str, TasteProfile],
        color: str = "blue",
    ) -> folium.Map:
        """Add restaurant markers to a map.

        Args:
            m: Folium map object
            locations: Dictionary mapping business IDs to coordinates
            profiles: Dictionary mapping business IDs to profiles
            color: Marker color

        Returns:
            Updated map object
        """
        for bid, (lat, lon) in locations.items():
            profile = profiles.get(bid)
            if not profile:
                continue

            # Create popup content
            popup_html = f"""
            <div style="width:200px">
                <b>{bid}</b><br>
                Price: {profile.price_category}<br>
                Style: {profile.dining_style}<br>
                <hr>
                <b>Significant Aspects:</b><br>
            """

            # Add significant aspects
            significant = profile.get_significant_aspects()
            for category, aspects in significant.items():
                for aspect, data in aspects.items():
                    if data["confidence"] > 0.5:
                        popup_html += (
                            f"{aspect}: {data['score']:.2f} "
                            f"({data['confidence']:.2f})<br>"
                        )

            popup_html += "</div>"

            # Add marker
            folium.Marker(
                location=(lat, lon),
                popup=popup_html,
                icon=folium.Icon(color=color, icon="info-sign"),
            ).add_to(m)

        return m

    def add_clusters(
        self,
        m: folium.Map,
        clusters: Dict[str, List[str]],
        locations: Dict[str, Tuple[float, float]],
        profiles: Dict[str, TasteProfile],
    ) -> folium.Map:
        """Add cluster visualization to a map.

        Args:
            m: Folium map object
            clusters: Dictionary mapping cluster labels to business IDs
            locations: Dictionary mapping business IDs to coordinates
            profiles: Dictionary mapping business IDs to profiles

        Returns:
            Updated map object
        """
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

        for i, (label, businesses) in enumerate(clusters.items()):
            color = colors[i % len(colors)]

            # Calculate cluster center and radius
            cluster_coords = np.array(
                [locations[bid] for bid in businesses if bid in locations]
            )

            if len(cluster_coords) == 0:
                continue

            center = cluster_coords.mean(axis=0)

            # Add circle for cluster area
            folium.Circle(
                location=center,
                radius=1000,  # 1km radius
                color=color,
                fill=True,
                popup=f"Cluster {label}",
            ).add_to(m)

            # Add markers for restaurants
            for bid in businesses:
                if bid not in locations or bid not in profiles:
                    continue

                lat, lon = locations[bid]
                profile = profiles[bid]

                # Create popup content
                popup_html = f"""
                <div style="width:200px">
                    <b>{bid}</b><br>
                    Price: {profile.price_category}<br>
                    Style: {profile.dining_style}<br>
                    Cluster: {label}
                </div>
                """

                folium.Marker(
                    location=(lat, lon),
                    popup=popup_html,
                    icon=folium.Icon(color=color, icon="info-sign"),
                ).add_to(m)

        return m

    def add_heatmap(
        self,
        m: folium.Map,
        locations: Dict[str, Tuple[float, float]],
        weights: Optional[Dict[str, float]] = None,
    ) -> folium.Map:
        """Add a heatmap layer to the map.

        Args:
            m: Folium map object
            locations: Dictionary mapping business IDs to coordinates
            weights: Optional dictionary mapping business IDs to weights

        Returns:
            Updated map object
        """
        # Prepare heatmap data
        heat_data = []
        for bid, (lat, lon) in locations.items():
            if weights and bid in weights:
                heat_data.append([lat, lon, weights[bid]])
            else:
                heat_data.append([lat, lon])

        # Add heatmap layer
        plugins.HeatMap(heat_data).add_to(m)
        return m

    def add_choropleth(
        self,
        m: folium.Map,
        geojson_data: Dict,
        values: Dict[str, float],
        name: str,
        fill_color: str = "YlOrRd",
        legend_name: str = "",
    ) -> folium.Map:
        """Add a choropleth layer to the map.

        Args:
            m: Folium map object
            geojson_data: GeoJSON data for areas
            values: Dictionary mapping area IDs to values
            name: Layer name
            fill_color: Color scheme
            legend_name: Legend title

        Returns:
            Updated map object
        """
        folium.Choropleth(
            geo_data=geojson_data,
            name=name,
            data=values,
            columns=["id", "value"],
            key_on="feature.id",
            fill_color=fill_color,
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=legend_name,
        ).add_to(m)

        return m

    def add_value_markers(
        self,
        m: folium.Map,
        locations: Dict[str, Tuple[float, float]],
        values: Dict[str, float],
        colormap: Optional[str] = "RdYlBu",
        radius: int = 15,
    ) -> folium.Map:
        """Add circular markers colored by value.

        Args:
            m: Folium map object
            locations: Dictionary mapping IDs to coordinates
            values: Dictionary mapping IDs to values
            colormap: Color scheme name
            radius: Marker radius in pixels

        Returns:
            Updated map object
        """
        # Create color map
        vmin = min(values.values())
        vmax = max(values.values())
        cmap = LinearColormap(colors=colormap, vmin=vmin, vmax=vmax)

        # Add markers
        for id_, (lat, lon) in locations.items():
            if id_ not in values:
                continue

            value = values[id_]
            color = cmap(value)

            folium.CircleMarker(
                location=(lat, lon),
                radius=radius,
                popup=f"{id_}: {value:.2f}",
                color=None,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
            ).add_to(m)

        # Add colorbar
        cmap.add_to(m)
        return m

    def save_map(self, m: folium.Map, filename: str) -> None:
        """Save map to HTML file.

        Args:
            m: Folium map object
            filename: Output filename
        """
        logger.info(f"Saving map to {filename}")
        m.save(filename)
