"""Module for generating analysis reports and visualizations."""

import csv
import itertools
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..constants.aspects import ASPECT_MAPPING
from ..core.profile import TasteProfile
from ..logging import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Class for generating analysis reports and visualizations.

    This class provides methods for creating various types of reports
    and visualizations from taste profile analysis results.

    Attributes:
        output_dir: Directory for saving reports
        style: Visual style for plots
    """

    def __init__(self, output_dir: str = "reports", style: str = "whitegrid"):
        """Initialize the report generator.

        Args:
            output_dir: Directory for saving reports
            style: Seaborn style name
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style(style)

    def generate_cluster_report(
        self,
        clusters: Dict[str, List[str]],
        profiles: Dict[str, TasteProfile],
        filename: str = "cluster_report.html",
    ) -> None:
        """Generate a report analyzing cluster characteristics.

        Args:
            clusters: Dictionary mapping cluster labels to business IDs
            profiles: Dictionary mapping business IDs to profiles
            filename: Output filename
        """
        logger.info("Generating cluster report")

        # Collect cluster statistics
        stats = []
        for label, businesses in clusters.items():
            cluster_profiles = [profiles[bid] for bid in businesses if bid in profiles]

            if not cluster_profiles:
                continue

            # Calculate price distribution
            price_dist = {"$": 0, "$$": 0, "$$$": 0, "$$$$": 0}
            for profile in cluster_profiles:
                price_dist[profile.price_category] += 1

            # Calculate dining style distribution
            style_dist = defaultdict(int)
            for profile in cluster_profiles:
                style_dist[profile.dining_style] += 1

            # Get dominant aspects
            aspects = defaultdict(list)
            for profile in cluster_profiles:
                significant = profile.get_significant_aspects()
                for category, category_aspects in significant.items():
                    for aspect, data in category_aspects.items():
                        if data["confidence"] > 0.5:
                            aspects[category].append((aspect, data["score"]))

            # Calculate average scores for aspects
            avg_aspects = {}
            for category, aspect_scores in aspects.items():
                if not aspect_scores:
                    continue

                category_avg = {}
                for aspect, scores in itertools.groupby(
                    aspect_scores, key=lambda x: x[0]
                ):
                    scores = list(scores)
                    avg_score = sum(s for _, s in scores) / len(scores)
                    category_avg[aspect] = avg_score

                avg_aspects[category] = category_avg

            stats.append(
                {
                    "cluster": label,
                    "size": len(cluster_profiles),
                    "price_distribution": price_dist,
                    "style_distribution": dict(style_dist),
                    "aspects": avg_aspects,
                }
            )

        # Create report HTML
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .cluster { margin-bottom: 30px; }
                .chart { margin: 10px 0; }
                table { border-collapse: collapse; }
                th, td { padding: 8px; border: 1px solid #ddd; }
                th { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
            <h1>Cluster Analysis Report</h1>
        """

        for cluster_stats in stats:
            html += f"""
            <div class="cluster">
                <h2>Cluster {cluster_stats['cluster']} ({cluster_stats['size']} businesses)</h2>

                <h3>Price Distribution</h3>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
            """

            for category, count in cluster_stats["price_distribution"].items():
                pct = count / cluster_stats["size"] * 100
                html += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{count}</td>
                        <td>{pct:.1f}%</td>
                    </tr>
                """

            html += """
                </table>

                <h3>Dining Styles</h3>
                <table>
                    <tr>
                        <th>Style</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
            """

            for style, count in cluster_stats["style_distribution"].items():
                pct = count / cluster_stats["size"] * 100
                html += f"""
                    <tr>
                        <td>{style}</td>
                        <td>{count}</td>
                        <td>{pct:.1f}%</td>
                    </tr>
                """

            html += """
                </table>

                <h3>Dominant Aspects</h3>
            """

            for category, aspects in cluster_stats["aspects"].items():
                html += f"""
                <h4>{category.title()}</h4>
                <table>
                    <tr>
                        <th>Aspect</th>
                        <th>Average Score</th>
                    </tr>
                """

                for aspect, score in aspects.items():
                    html += f"""
                    <tr>
                        <td>{aspect}</td>
                        <td>{score:.2f}</td>
                    </tr>
                    """

                html += "</table>"

            html += "</div>"

        html += """
        </body>
        </html>
        """

        # Save report
        path = self.output_dir / filename
        with open(path, "w") as f:
            f.write(html)

        logger.info(f"Saved cluster report to {path}")

    def plot_aspect_distributions(
        self,
        profiles: Dict[str, Any],
        aspects: List[str],
        output_file: str,
        figsize: Tuple[int, int] = (12, 6),
    ) -> None:
        """Plot distributions of aspect scores.

        Args:
            profiles: Dictionary of business profiles
            aspects: List of aspects to plot
            output_file: Output file path
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)

        # Prepare data
        data = []
        for aspect in aspects:
            scores = []
            for profile in profiles.values():
                if aspect in profile:
                    scores.append(profile[aspect])
            if scores:
                data.append(scores)

        # Create violin plot
        if data:
            sns.violinplot(data=data)
            plt.xticks(range(len(aspects)), aspects, rotation=45)
            plt.ylabel("Score")
            plt.title("Distribution of Aspect Scores")

            # Save plot
            output_path = self.output_dir / output_file
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved aspect distributions plot to {output_path}")
        else:
            logger.warning("No data available for aspect distributions plot")

    def plot_price_analysis(
        self,
        profiles: Dict[str, Any],
        output_file: str,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """Plot price analysis visualization.

        Args:
            profiles: Dictionary of business profiles
            output_file: Output file path
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)

        # Extract price and value data
        prices = []
        values = []
        for profile in profiles.values():
            if "price_level" in profile and "value_ratio" in profile:
                prices.append(profile["price_level"])
                values.append(profile["value_ratio"])

        if prices and values:
            # Create scatter plot
            plt.scatter(prices, values, alpha=0.5)
            plt.xlabel("Price Level")
            plt.ylabel("Value Ratio")
            plt.title("Price vs. Value Analysis")

            # Add trend line
            z = np.polyfit(prices, values, 1)
            p = np.poly1d(z)
            plt.plot(prices, p(prices), "r--", alpha=0.8)

            # Save plot
            output_path = self.output_dir / output_file
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved price analysis plot to {output_path}")
        else:
            logger.warning("No data available for price analysis plot")

    def plot_correlation_matrix(
        self,
        profiles: Dict[str, Any],
        aspects: List[str],
        output_file: str,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """Plot correlation matrix between aspects.

        Args:
            profiles: Dictionary of business profiles
            aspects: List of aspects to analyze
            output_file: Output file path
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)

        # Prepare correlation matrix
        data = np.zeros((len(aspects), len(aspects)))
        for i, j in itertools.product(range(len(aspects)), range(len(aspects))):
            aspect1, aspect2 = aspects[i], aspects[j]
            values1, values2 = [], []

            for profile in profiles.values():
                if aspect1 in profile and aspect2 in profile:
                    values1.append(profile[aspect1])
                    values2.append(profile[aspect2])

            if values1 and values2:
                correlation = np.corrcoef(values1, values2)[0, 1]
                data[i, j] = correlation

        # Create heatmap
        sns.heatmap(
            data,
            xticklabels=aspects,
            yticklabels=aspects,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
        )
        plt.title("Aspect Correlation Matrix")

        # Save plot
        output_path = self.output_dir / output_file
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved correlation matrix to {output_path}")

    def export_summary_stats(self, profiles: Dict[str, Any], output_file: str) -> None:
        """Export summary statistics to CSV.

        Args:
            profiles: Dictionary of business profiles
            output_file: Output file path
        """
        # Prepare statistics
        stats = {
            "total_businesses": len(profiles),
            "avg_price_level": 0.0,
            "avg_value_ratio": 0.0,
            "price_categories": {},
            "dining_styles": {},
        }

        # Calculate averages
        price_levels = []
        value_ratios = []
        price_cats = {}
        styles = {}

        for profile in profiles.values():
            if "price_level" in profile:
                price_levels.append(profile["price_level"])
            if "value_ratio" in profile:
                value_ratios.append(profile["value_ratio"])
            if "price_category" in profile:
                cat = profile["price_category"]
                price_cats[cat] = price_cats.get(cat, 0) + 1
            if "dining_style" in profile:
                style = profile["dining_style"]
                styles[style] = styles.get(style, 0) + 1

        if price_levels:
            stats["avg_price_level"] = sum(price_levels) / len(price_levels)
        if value_ratios:
            stats["avg_value_ratio"] = sum(value_ratios) / len(value_ratios)

        stats["price_categories"] = price_cats
        stats["dining_styles"] = styles

        # Write to CSV
        output_path = self.output_dir / output_file
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Businesses", stats["total_businesses"]])
            writer.writerow(["Average Price Level", f"{stats['avg_price_level']:.2f}"])
            writer.writerow(["Average Value Ratio", f"{stats['avg_value_ratio']:.2f}"])

            writer.writerow([])
            writer.writerow(["Price Categories", "Count"])
            for cat, count in stats["price_categories"].items():
                writer.writerow([cat, count])

            writer.writerow([])
            writer.writerow(["Dining Styles", "Count"])
            for style, count in stats["dining_styles"].items():
                writer.writerow([style, count])

        logger.info(f"Saved summary statistics to {output_path}")
