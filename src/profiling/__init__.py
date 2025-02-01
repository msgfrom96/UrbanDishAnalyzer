"""Urban Dining Analysis (UDA) package.

A Python package for analyzing restaurant reviews and generating taste profiles
through multilingual aspect-based sentiment analysis. It clusters reviews based
on location and specific business profiles, while also identifying potential
hotspots for new restaurant openings by analyzing the area's taste profile.

Example:
    >>> from profiling import TasteProfileAnalyzer, DataLoader
    >>> loader = DataLoader()
    >>> loader.load_businesses("data/business.json")
    >>> loader.load_reviews("data/review.json")
    >>> analyzer = TasteProfileAnalyzer()
    >>> analyzer.analyze_review("business_1", "Great food and service!")
    >>> profile = analyzer.get_business_profile("business_1")
    >>> print(profile.taste_quality)
    0.85
"""

from .analyzer.aspect_extractor import AspectExtractor
from .core.analyzer import TasteProfileAnalyzer
from .core.profile import TasteProfile

__all__ = ["TasteProfileAnalyzer", "TasteProfile", "AspectExtractor"]

__version__ = "0.1.0"

# Configuration and logging
from .config import config

# Data handling
from .data.loader import DataLoader
from .data.storage import DataStorage

# Geographical analysis
from .geo.clustering import LocationClusterer
from .geo.hotspots import HotspotDetector
from .logging import get_logger

# Utility functions
from .utils import (
    calculate_distance,
    clean_text,
    detect_language,
    is_within_radius,
    validate_coordinates,
    validate_date,
    validate_review,
)

# Visualization
from .visualization.maps import MapVisualizer
from .visualization.reports import ReportGenerator

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core classes
    "TasteProfileAnalyzer",
    "TasteProfile",
    "AspectExtractor",
    # Data handling
    "DataLoader",
    "DataStorage",
    # Geographical analysis
    "LocationClusterer",
    "HotspotDetector",
    # Visualization
    "MapVisualizer",
    "ReportGenerator",
    # Configuration and logging
    "config",
    "get_logger",
    # Utility functions
    "clean_text",
    "detect_language",
    "calculate_distance",
    "is_within_radius",
    "validate_coordinates",
    "validate_date",
    "validate_review",
]
