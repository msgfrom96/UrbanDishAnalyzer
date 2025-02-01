"""Example demonstrating multilingual analysis capabilities of the UDA package.

This script shows how to:
1. Process reviews in multiple languages
2. Generate language-specific taste profiles
3. Compare dining preferences across languages
"""

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


def run_multilingual_analysis(
    data_path: str, output_path: str, metro_area: str = "St. Petersburg"
) -> None:
    """Run a multilingual analysis demonstration.

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

    # Process reviews by language
    profiles_by_lang: Dict[str, Dict[str, Any]] = {}
    for _, business in businesses.iterrows():
        business_id = business["business_id"]
        business_reviews = data_loader.get_business_reviews(business_id, min_reviews=3)

        if business_reviews:
            logger.info(
                f"Processing {len(business_reviews)} reviews for {business['name']}"
            )
            for review in business_reviews:
                lang = analyzer.get_language(review["text"])
                if lang not in profiles_by_lang:
                    profiles_by_lang[lang] = {}

                analyzer.analyze_review(business_id, review["text"])
                profiles_by_lang[lang][business_id] = analyzer.get_business_profile(
                    business_id
                )

    if not profiles_by_lang:
        logger.warning("No profiles generated. Check if data was loaded correctly.")
        return

    # Generate visualizations
    logger.info("Generating visualizations...")

    for lang, profiles in profiles_by_lang.items():
        if not profiles:
            continue

        logger.info(f"Generating reports for {lang} reviews...")

        # Aspect distributions
        report_gen.plot_aspect_distributions(
            profiles,
            ["taste_quality", "service_quality", "ambiance", "value_ratio"],
            f"aspect_distributions_{lang}.png",
        )

        # Price analysis
        report_gen.plot_price_analysis(profiles, f"price_analysis_{lang}.png")

        # Export statistics
        report_gen.export_summary_stats(profiles, f"summary_stats_{lang}.csv")

    logger.info("Analysis complete! Check the output directory for results.")


if __name__ == "__main__":
    # Example usage
    run_multilingual_analysis("data", "output/multilingual_analysis/")
