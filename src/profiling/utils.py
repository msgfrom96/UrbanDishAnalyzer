"""Utility functions for the UDA package.

This module provides common utility functions used across the package:
- Text processing and cleaning
- Geographical calculations
- Data validation and conversion
- File operations
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon

from .config import config
from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# Text processing functions
def clean_text(text: str) -> str:
    """Clean and normalize text.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def detect_language(text: str) -> str:
    """Detect text language.

    Args:
        text: Input text

    Returns:
        Language code (e.g., 'en', 'es', 'fr')
    """
    # Import langdetect only when needed
    from langdetect import detect

    try:
        # return a string of the detected language
        return str(detect(text))
    except Exception:
        logger.warning("Language detection failed, defaulting to English")
        return "en"


def normalize_aspect(aspect: str) -> str:
    """Normalize aspect name.

    Args:
        aspect: Aspect name

    Returns:
        Normalized aspect name
    """
    # Remove special characters
    aspect = re.sub(r"[^\w\s]", "", aspect)

    # Convert to lowercase and replace spaces with underscores
    aspect = aspect.lower().replace(" ", "_")

    return aspect


# Geographical functions
def calculate_distance(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """Calculate distance between two points in kilometers.

    Args:
        point1: (latitude, longitude) of first point
        point2: (latitude, longitude) of second point

    Returns:
        Distance in kilometers
    """
    distance = geodesic(point1, point2).kilometers
    return float(distance)


def is_within_radius(
    center: Tuple[float, float], point: Tuple[float, float], radius_km: float
) -> bool:
    """Check if point is within radius of center.

    Args:
        center: (latitude, longitude) of center point
        point: (latitude, longitude) of point to check
        radius_km: Radius in kilometers

    Returns:
        True if point is within radius
    """
    return calculate_distance(center, point) <= radius_km


def create_grid(
    bounds: Tuple[float, float, float, float], cell_size_km: float
) -> List[Polygon]:
    """Create grid of cells within bounds.

    Args:
        bounds: (min_lat, min_lon, max_lat, max_lon)
        cell_size_km: Cell size in kilometers

    Returns:
        List of cell polygons
    """
    min_lat, min_lon, max_lat, max_lon = bounds

    # Convert cell size to degrees (approximate)
    cell_size_lat = cell_size_km / 111.0  # 1 degree ≈ 111km
    cell_size_lon = cell_size_km / (111.0 * np.cos(np.radians(min_lat)))

    # Create grid
    cells = []
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            # Create cell polygon
            cell = Polygon(
                [
                    (lon, lat),
                    (lon + cell_size_lon, lat),
                    (lon + cell_size_lon, lat + cell_size_lat),
                    (lon, lat + cell_size_lat),
                ]
            )
            cells.append(cell)
            lon += cell_size_lon
        lat += cell_size_lat

    return cells


# Data validation functions
def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate latitude and longitude.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        True if coordinates are valid
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def validate_date(date_str: str, max_age_days: Optional[int] = None) -> bool:
    """Validate date string.

    Args:
        date_str: Date string (YYYY-MM-DD)
        max_age_days: Maximum age in days

    Returns:
        True if date is valid
    """
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        if max_age_days is not None:
            max_age = timedelta(days=max_age_days)
            return datetime.now() - date <= max_age
        return True
    except ValueError:
        return False


def validate_review(review: Dict[str, Any]) -> List[str]:
    """Validate review data.

    Args:
        review: Review dictionary

    Returns:
        List of validation error messages
    """
    errors = []

    # Required fields
    required = ["business_id", "text", "date"]
    for field in required:
        if field not in review:
            errors.append(f"Missing required field: {field}")

    # Text length
    if "text" in review:
        min_length = config.get("validation.min_review_length", 10)
        if len(review["text"]) < min_length:
            errors.append(f"Review text too short (min: {min_length})")

    # Date
    if "date" in review:
        max_age = config.get("validation.max_review_age_days", 365)
        if not validate_date(review["date"], max_age):
            errors.append(f"Invalid or too old date (max age: {max_age} days)")

    return errors


# File operations
def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Loaded JSON data (parsed from JSON format)

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {str(e)}")
        raise


def save_json(data: Any, path: Union[str, Path]) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        path: Path to save to

    Raises:
        OSError: If file cannot be written
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except OSError as e:
        logger.error(f"Failed to save {path}: {str(e)}")
        raise


def get_file_age(path: Union[str, Path]) -> float:
    """Get file age in hours.

    Args:
        path: File path

    Returns:
        Age in hours
    """
    mtime = Path(path).stat().st_mtime
    age = datetime.now().timestamp() - mtime
    return age / 3600  # Convert to hours


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to range [min_val, max_val].

    Args:
        score: Score to normalize
        min_val: Minimum value of range
        max_val: Maximum value of range

    Returns:
        Normalized score
    """
    return float(min_val + (max_val - min_val) * (score + 1) / 2)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers.

    Args:
        a: Numerator
        b: Denominator
        default: Default value if denominator is zero

    Returns:
        Result of division or default value
    """
    try:
        if b == 0:
            return default
        return float(a / b)
    except (TypeError, ValueError) as e:
        logger.warning(f"Division error: {str(e)}")
        return default


def ensure_list(value: Union[T, List[T]]) -> List[T]:
    """Ensure value is a list.

    Args:
        value: Value to convert

    Returns:
        List containing value if not already a list
    """
    if isinstance(value, list):
        return value
    return [value]


def create_point(lat: float, lon: float) -> Point:
    """Create Point geometry from coordinates.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Point geometry
    """
    return Point(lon, lat)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Haversine distance between two points.

    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point

    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in kilometers
    r = 6371

    return float(c * r)


def weighted_average(
    values: List[float], weights: Optional[List[float]] = None
) -> float:
    """Calculate weighted average.

    Args:
        values: List of values
        weights: Optional list of weights (defaults to equal weights)

    Returns:
        Weighted average

    Raises:
        ValueError: If lengths of values and weights don't match
    """
    if not values:
        return 0.0

    if weights is None:
        weights = [1.0] * len(values)
    elif len(weights) != len(values):
        raise ValueError("Length of values and weights must match")

    return float(np.average(values, weights=weights))


def moving_average(values: List[float], window: int = 3) -> List[float]:
    """Calculate moving average.

    Args:
        values: List of values
        window: Window size

    Returns:
        List of moving averages
    """
    if not values:
        return []

    result = np.convolve(values, np.ones(window) / window, mode="valid")
    return cast(List[float], result.tolist())
