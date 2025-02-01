"""Script for extracting restaurant data from Yelp dataset."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_restaurants(
    input_file: str, output_file: str, city: Optional[str] = None, min_reviews: int = 10
) -> None:
    """Extract restaurant data from Yelp dataset.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        city: Optional city to filter for
        min_reviews: Minimum number of reviews required
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Create output directory if needed
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)

    # Process businesses
    restaurants: List[Dict[str, Any]] = []
    with open(input_file) as f:
        for line in f:
            business = json.loads(line)

            # Check if restaurant
            if "Restaurants" not in business.get("categories", []):
                continue

            # Check city if specified
            if city and business["city"] != city:
                continue

            # Check review count
            if business["review_count"] < min_reviews:
                continue

            # Extract relevant fields
            restaurant = {
                "business_id": business["business_id"],
                "name": business["name"],
                "city": business["city"],
                "state": business["state"],
                "stars": business["stars"],
                "review_count": business["review_count"],
                "categories": business["categories"],
                "attributes": business.get("attributes", {}),
                "hours": business.get("hours", {}),
            }
            restaurants.append(restaurant)

    # Write output
    logger.info(f"Found {len(restaurants)} restaurants")
    with open(output_file, "w") as f:
        for restaurant in restaurants:
            f.write(json.dumps(restaurant) + "\n")
    logger.info(f"Wrote output to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract restaurant data from Yelp dataset"
    )
    parser.add_argument("input_file", help="Input JSON file")
    parser.add_argument("output_file", help="Output JSON file")
    parser.add_argument("--city", help="Filter for specific city")
    parser.add_argument(
        "--min-reviews", type=int, default=10, help="Minimum number of reviews required"
    )

    args = parser.parse_args()
    extract_restaurants(args.input_file, args.output_file, args.city, args.min_reviews)
