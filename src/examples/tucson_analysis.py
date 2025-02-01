"""Example script for analyzing restaurant hotspots in Tucson, Arizona.

This script performs geographical analysis of restaurants in the Tucson area,
identifying clusters and potential hotspots for new restaurants.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

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


def run_tucson_analysis(data_path: str, output_path: str) -> None:
    """Run a geographical analysis for Tucson restaurants.

    Args:
        data_path: Path to data directory
        output_path: Path for output files
    """
    # Initialize components
    data_loader = DataLoader()
    analyzer = TasteProfileAnalyzer()
    clusterer = LocationClusterer(analyzer)
    hotspot_detector = HotspotDetector()

    # Tucson coordinates (32.2226° N, 110.9747° W)
    visualizer = MapVisualizer((32.2226, -110.9747))

    # Load and process data
    logger.info("Loading data...")
    data_loader.load_businesses(
        str(Path(data_path) / "yelp_academic_dataset_business.json")
    )
    data_loader.load_reviews(str(Path(data_path) / "yelp_academic_dataset_review.json"))

    # Filter for Tucson area and get all reviews at once
    businesses = data_loader.filter_by_city("Tucson")
    logger.info(f"Found {len(businesses)} businesses in Tucson")

    # Get all reviews for Tucson businesses
    all_reviews = data_loader.reviews

    # Filter reviews for Tucson businesses and group by business_id
    tucson_reviews = pd.merge(
        businesses[["business_id", "name", "latitude", "longitude"]],
        all_reviews,
        on="business_id",
        how="inner",
    )

    # Group reviews by business and count them
    review_counts = (
        tucson_reviews.groupby("business_id").size().reset_index(name="review_count")
    )

    # Filter businesses with minimum number of reviews
    qualified_businesses = review_counts[review_counts["review_count"] >= 10]

    # Process reviews and generate profiles
    profiles: Dict[str, Any] = {}
    locations: Dict[str, Tuple[float, float]] = {}

    # Group reviews by business for efficient processing
    business_reviews = tucson_reviews[
        tucson_reviews["business_id"].isin(qualified_businesses["business_id"])
    ].groupby("business_id")

    for business_id, group in business_reviews:
        logger.info(f"Processing {len(group)} reviews for {group['name'].iloc[0]}")
        for _, review in group.iterrows():
            analyzer.analyze_review(business_id, review["text"])

        profile = analyzer.get_business_profile(business_id)
        if profile is not None:
            profiles[business_id] = profile
            locations[business_id] = (
                float(group["latitude"].iloc[0]),
                float(group["longitude"].iloc[0]),
            )

    if not profiles:
        logger.warning("No profiles generated. Check if data was loaded correctly.")
        return

    # Perform geographical analysis
    logger.info("Performing geographical analysis...")

    # Cluster restaurants (using 1.5 km radius for Tucson's spread-out nature)
    clusters = clusterer.cluster_by_location(1.5)
    logger.info(f"Found {len(clusters)} restaurant clusters")

    # Detect hotspots
    hotspots = hotspot_detector.detect_hotspots()
    logger.info(f"Detected {len(hotspots)} potential hotspots")

    # Generate visualizations
    logger.info("Generating maps...")

    # Create base map centered on Tucson
    base_map = visualizer.create_base_map(None, 13)  # zoom level 13 for city view

    # Add clusters to map
    visualizer.add_clusters(base_map, clusters, locations, profiles)

    # Add hotspots to map with markers
    for hotspot in hotspots:
        visualizer.add_marker(
            base_map,
            location=hotspot.location,
            popup=f"Potential Hotspot\nScore: {hotspot.score:.2f}",
            color="red",
        )

    # Save the map
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer.save_map(base_map, str(output_dir / "tucson_analysis.html"))

    logger.info("Analysis complete! Check the output directory for results.")


if __name__ == "__main__":
    # Example usage
    run_tucson_analysis(data_path="data/", output_path="output/tucson_analysis/")
