"""Extraction functionality for aspect-based sentiment analysis."""

from .extractor import AspectExtractor, get_extractor
from .sentiment import extract_aspects_and_sentiments, map_aspects_to_profile

__all__ = [
    "AspectExtractor",
    "get_extractor",
    "extract_aspects_and_sentiments",
    "map_aspects_to_profile",
]
