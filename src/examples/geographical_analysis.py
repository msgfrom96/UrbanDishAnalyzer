"""Example demonstrating geographical analysis capabilities of the UDA package.

This script shows how to:
1. Cluster restaurants based on location and taste profiles
2. Detect potential hotspots for new restaurants
3. Generate interactive maps and visualizations
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from ..profiling.core.analyzer import TasteProfileAnalyzer
from ..profiling.data.loader import DataLoader
from ..profiling.geo.clustering import LocationClusterer
from ..profiling.geo.hotspots import HotspotDetector
from ..profiling.visualization.maps import MapVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_geographical_analysis(
    data_path: str, output_path: str, metro_area: str = "St. Petersburg"
) -> None:
    """Run a geographical analysis demonstration.

    Args:
        data_path: Path to data directory
        output_path: Path for output files
        metro_area: Metropolitan area to analyze
    """
    # Initialize components
    data_loader = DataLoader()
    analyzer = TasteProfileAnalyzer()
    clusterer = LocationClusterer(analyzer)
    hotspot_detector = HotspotDetector()
    visualizer = MapVisualizer((37.7749, -122.4194))  # San Francisco coordinates

    # Load and process data
    logger.info("Loading data...")
    data_loader.load_businesses(
        str(Path(data_path) / "yelp_academic_dataset_business.json")
    )
    data_loader.load_reviews(str(Path(data_path) / "yelp_academic_dataset_review.json"))

    # Filter for metro area
    businesses = data_loader.filter_by_city(metro_area)
    logger.info(f"Found {len(businesses)} businesses in {metro_area}")

    # Process reviews and generate profiles
    profiles: Dict[str, Any] = {}
    locations: Dict[str, Tuple[float, float]] = {}
    for _, business in businesses.iterrows():
        business_id = business["business_id"]
        business_reviews = data_loader.get_business_reviews(business_id, min_reviews=3)

        if business_reviews:
            logger.info(
                f"Processing {len(business_reviews)} reviews for {business['name']}"
            )
            for review in business_reviews:
                analyzer.analyze_review(business_id, review["text"])
            profiles[business_id] = analyzer.get_business_profile(business_id)
            locations[business_id] = (
                float(business["latitude"]),
                float(business["longitude"]),
            )

    if not profiles:
        logger.warning("No profiles generated. Check if data was loaded correctly.")
        return

    # Perform geographical analysis
    logger.info("Performing geographical analysis...")

    # Cluster restaurants
    clusters = clusterer.cluster_by_location(1.0)  # 1.0 km radius
    logger.info(f"Found {len(clusters)} restaurant clusters")

    # Detect hotspots
    hotspots = hotspot_detector.detect_hotspots()
    logger.info(f"Detected {len(hotspots)} potential hotspots")

    # Generate visualizations
    logger.info("Generating maps...")
    visualizer.create_base_map(None, 12)  # zoom level 12
    for cluster in clusters:
        # Mapvisualizer does not have a plot_cluster method
        # so we need to create a new map and add the cluster to it
        m = visualizer.create_base_map(None, 12)
        visualizer.add_clusters(m, clusters, locations, profiles)
        visualizer.save_map(m, "clusters.html")

    logger.info("Analysis complete! Check the output directory for results.")


if __name__ == "__main__":
    # Example usage
    run_geographical_analysis(
        data_path="data/",
        output_path="output/geo_analysis/",
        metro_area="San Francisco",
    )
