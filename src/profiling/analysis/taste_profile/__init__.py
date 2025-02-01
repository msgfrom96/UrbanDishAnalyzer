"""Taste profile analysis package for understanding business characteristics.

This package provides tools for:
1. Extracting taste profiles from reviews using aspect-based sentiment analysis
2. Analyzing and aggregating profiles across businesses
3. Clustering similar businesses based on their taste profiles
4. Generating area-specific taste profiles

Main components:
- TasteProfileAnalyzer: Main class for analyzing reviews and managing profiles
- TasteProfile: Class representing a business's taste characteristics
- Constants: Aspect mappings and weights for analysis
"""

from .analyzer import TasteProfileAnalyzer
from .constants import ASPECT_MAPPING, MIN_CONFIDENCE_THRESHOLD
from .profile import TasteProfile

__all__ = [
    "TasteProfileAnalyzer",
    "TasteProfile",
    "ASPECT_MAPPING",
    "MIN_CONFIDENCE_THRESHOLD",
]
