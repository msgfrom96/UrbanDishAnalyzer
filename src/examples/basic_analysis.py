import logging
from pathlib import Path
from typing import Any, Dict

from ..profiling.core.analyzer import TasteProfileAnalyzer
from ..profiling.data.loader import DataLoader
from ..profiling.visualization.reports import ReportGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_basic_analysis(
    data_path: str, output_path: str, metro_area: str = "St. Petersburg"
) -> None:
    """Run a basic analysis demonstration.

    Args:
        data_path: Path to data directory
        output_path: Path for output files
        metro_area: Metropolitan area to analyze
    """
    # Initialize components
    data_loader = DataLoader()
    analyzer = TasteProfileAnalyzer()
    report_gen = ReportGenerator(output_path)

    # Load data
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

    if not profiles:
        logger.warning("No profiles generated. Check if data was loaded correctly.")
        return

    # Generate visualizations
    logger.info("Generating visualizations...")

    # Aspect distributions
    report_gen.plot_aspect_distributions(
        profiles,
        ["taste_quality", "service_quality", "ambiance", "value_ratio"],
        "aspect_distributions.png",
    )

    # Price analysis
    report_gen.plot_price_analysis(profiles, "price_analysis.png")

    # Export statistics
    report_gen.export_summary_stats(profiles, "summary_stats.csv")

    logger.info("Analysis complete! Check the output directory for results.")


if __name__ == "__main__":
    # Example usage
    run_basic_analysis(
        data_path="data/",
        output_path="output/basic_analysis/",
        metro_area="St. Petersburg",
    )
