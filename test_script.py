"""Test script for the Urban Dining Analyzer."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from profiling.analysis.taste_profile.analyzer import TasteProfileAnalyzer
from profiling.core.profile import TasteProfile
from profiling.data.loader import DataLoader
from profiling.visualization.reports import ReportGenerator


def setup_test_environment() -> None:
    """Set up the test environment."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)


def run_test_analysis() -> None:
    """Run test analysis on sample data."""
    # Load test data
    loader = DataLoader()
    loader.load_businesses("data/sample_businesses.json")
    loader.load_reviews("data/sample_reviews.json")

    # Filter for test city
    test_businesses = loader.filter_by_city("Test City")

    # Create analyzer
    analyzer = TasteProfileAnalyzer()

    # Process each business
    profiles: Dict[str, TasteProfile] = {}
    for _, business in test_businesses.iterrows():
        # Create profile
        profile = TasteProfile(
            restaurant_id=business["business_id"], name=business["name"]
        )

        # Get reviews
        reviews = loader.get_business_reviews(business["business_id"])

        # Process reviews
        for review in reviews:
            profile.update_from_review(review)

        profiles[business["business_id"]] = profile

    # Generate reports
    report_gen = ReportGenerator()
    report_gen.plot_aspect_distributions(profiles)
    report_gen.plot_price_analysis(profiles)
    report_gen.export_summary_stats(profiles)


def main() -> None:
    """Main entry point."""
    # Set up environment
    setup_test_environment()

    # Run analysis
    run_test_analysis()


if __name__ == "__main__":
    main()
