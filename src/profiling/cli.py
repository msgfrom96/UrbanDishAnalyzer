"""Command-line interface for the profiling package."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .analysis.taste_profile import TasteProfileAnalyzer
from .config import Config
from .data.loader import DataLoader
from .data.storage import DataStorage
from .logging import setup_logging
from .visualization.reports import ReportGenerator

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def run_analysis(config_path: str) -> Optional[Dict[str, Any]]:
    """Run the analysis pipeline.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing analysis results or None if analysis fails
    """
    # Load configuration
    config = Config()
    config.load_from_file(config_path)
    if not config.validate():
        logger.error("Failed to load configuration")
        return None

    # Create output directory
    output_dir = config.output_dir if hasattr(config, "output_dir") else "output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize components
    data_loader = DataLoader()
    analyzer = TasteProfileAnalyzer()
    storage = DataStorage(output_dir)
    report_gen = ReportGenerator(output_dir)

    try:
        # Load data
        logger.info("Loading data...")
        data_dir = config.data_dir if hasattr(config, "data_dir") else "data"
        data_loader.load_businesses(str(Path(data_dir) / "businesses.json"))
        data_loader.load_reviews(str(Path(data_dir) / "reviews.json"))
        businesses = data_loader.businesses
        _ = data_loader.reviews  # Keep reference but don't use directly

        if businesses is None:
            logger.error("No businesses loaded")
            return None

        # Process data
        logger.info("Processing data...")
        profiles: Dict[str, Any] = {}
        locations: Dict[str, Tuple[float, float]] = {}

        for _, business in businesses.iterrows():
            business_id = str(business["business_id"])
            business_reviews = data_loader.get_business_reviews(
                business_id, min_reviews=3
            )

            if business_reviews:
                logger.info(
                    f"Processing {len(business_reviews)} reviews for {business['name']}"
                )
                for review in business_reviews:
                    analyzer.analyze_review(business_id, review["text"])
                profile = analyzer.get_business_profile(business_id)
                if profile is not None:
                    profiles[business_id] = profile
                    locations[business_id] = (
                        float(business["latitude"]),
                        float(business["longitude"]),
                    )

        if not profiles:
            logger.error("No profiles generated")
            return None

        # Save results
        logger.info("Saving results...")
        storage.save_profiles(profiles, "profiles.json")
        cluster_data: Dict[str, Any] = {"locations": list(locations.keys())}
        storage.save_clusters(cluster_data, "clusters.json")

        # Generate visualizations
        logger.info("Generating visualizations...")
        report_gen.plot_aspect_distributions(
            profiles,
            ["taste_quality", "service_quality", "ambiance", "value_ratio"],
            "aspect_distributions.png",
        )
        report_gen.plot_price_analysis(profiles, "price_analysis.png")
        report_gen.export_summary_stats(profiles, "summary_stats.csv")

        logger.info("Analysis complete! Check the output directory for results.")
        return {"profiles": profiles, "locations": locations}

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return None


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Restaurant profiling tool")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()

    run_analysis(args.config)


if __name__ == "__main__":
    main()
