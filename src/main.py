"""Main module for the restaurant profiling application."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .analysis.taste_profile import TasteProfileAnalyzer
from .config import Config
from .data.loader import DataLoader
from .data.storage import DataStorage
from .logging import setup_logging
from .visualization.maps import MapVisualizer
from .visualization.reports import ReportGenerator

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def load_data(config: Config) -> Optional[Dict[str, Any]]:
    """Load restaurant data from configured sources.

    Args:
        config: Application configuration

    Returns:
        Dictionary containing loaded data or None if loading fails
    """
    try:
        data_loader = DataLoader()
        data_loader.load_businesses(str(Path(config.data_path) / "businesses.json"))
        data_loader.load_reviews(str(Path(config.data_path) / "reviews.json"))
        return data_loader.get_data()
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return None


def analyze_profiles(data: Dict[str, Any], config: Config) -> Optional[Dict[str, Any]]:
    """Analyze restaurant profiles.

    Args:
        data: Input data dictionary
        config: Application configuration

    Returns:
        Dictionary containing analysis results or None if analysis fails
    """
    try:
        analyzer = TasteProfileAnalyzer()
        return analyzer.analyze_data(data)
    except Exception as e:
        logger.error(f"Failed to analyze profiles: {str(e)}")
        return None


def generate_visualizations(
    data: Dict[str, Any], results: Dict[str, Any], config: Config
) -> bool:
    """Generate visualization reports.

    Args:
        data: Input data dictionary
        results: Analysis results dictionary
        config: Application configuration

    Returns:
        True if visualization generation succeeds, False otherwise
    """
    try:
        report_gen = ReportGenerator(config.output_path)
        map_viz = MapVisualizer((37.7749, -122.4194))  # San Francisco coordinates

        # Generate reports
        report_gen.plot_aspect_distributions(
            results["profiles"],
            ["taste_quality", "service_quality", "ambiance", "value_ratio"],
            "aspect_distributions.png",
        )
        report_gen.plot_price_analysis(results["profiles"], "price_analysis.png")
        report_gen.export_summary_stats(results["profiles"], "summary_stats.csv")

        # Generate maps
        map_viz.create_base_map(None, 12)  # zoom level 12
        map_viz.add_value_markers(data["locations"], "restaurant_locations.html")

        return True
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {str(e)}")
        return False


def save_results(results: Dict[str, Any], config: Config) -> bool:
    """Save analysis results.

    Args:
        results: Analysis results dictionary
        config: Application configuration

    Returns:
        True if saving succeeds, False otherwise
    """
    try:
        storage = DataStorage(config.output_path)
        storage.save_profiles(results["profiles"], "profiles.json")
        storage.save_clusters(results["clusters"], "clusters.json")
        return True
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        return False


def main() -> None:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Restaurant profiling application")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = Config.load_from_file(args.config)
    if not config:
        logger.error("Failed to load configuration")
        return

    # Create output directory
    Path(config.output_path).mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_data(config)
    if not data:
        return

    # Analyze profiles
    results = analyze_profiles(data, config)
    if not results:
        return

    # Generate visualizations
    if not generate_visualizations(data, results, config):
        return

    # Save results
    if not save_results(results, config):
        return

    logger.info("Analysis complete! Check the output directory for results.")


if __name__ == "__main__":
    main()
