"""Configuration module for the UDA package.

This module provides a centralized configuration system for managing:
- Analysis parameters and thresholds
- Language support settings
- Aspect definitions and mappings
- Logging configuration
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import yaml

logger = logging.getLogger(__name__)


class ThresholdConfig(TypedDict):
    """Type definition for threshold configuration."""

    min_confidence: float
    min_reviews: int
    min_mentions: int
    max_distance: float


class AspectConfig(TypedDict):
    """Type definition for aspect configuration."""

    weights: Dict[str, float]
    thresholds: Dict[str, float]


class Config:
    """Configuration manager for the UDA package."""

    _instance = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration with default values."""
        if not hasattr(self, "initialized"):
            self.initialized = True
            self._load_defaults()

    def _load_defaults(self):
        """Load default configuration values."""
        # Analysis thresholds
        self.thresholds: ThresholdConfig = {
            "min_confidence": 0.3,
            "min_reviews": 3,
            "min_mentions": 2,
            "max_distance": 1.0,
        }

        # Aspect categories and weights
        self.aspects: AspectConfig = {
            "weights": {"taste": 1.0, "service": 0.8, "ambiance": 0.6, "value": 0.7},
            "thresholds": {"taste": 0.5, "service": 0.5, "ambiance": 0.5, "value": 0.5},
        }

        # Language support
        self.languages = {
            "english": {
                "enabled": True,
                "weight": 1.0,
                "sentiment_model": "en_core_web_lg",
            },
            "spanish": {
                "enabled": True,
                "weight": 1.0,
                "sentiment_model": "es_core_news_lg",
            },
            "french": {
                "enabled": True,
                "weight": 1.0,
                "sentiment_model": "fr_core_news_lg",
            },
        }

        # Output settings
        self.output = {
            "formats": {
                "reports": ["html", "pdf"],
                "data": ["json", "csv"],
                "maps": ["html", "png"],
            },
            "style": {
                "theme": "light",
                "colors": {
                    "primary": "#2C3E50",
                    "secondary": "#E74C3C",
                    "accent": "#3498DB",
                },
                "fonts": {"main": "Arial", "headers": "Helvetica"},
            },
        }

        # Cache settings
        self.cache = {
            "enabled": True,
            "directory": ".cache",
            "max_age_hours": 24,
            "max_size_mb": 1000,
        }

        # Logging settings
        self.logging = {
            "directory": "./logs",
            "level": "INFO",
            "console": True,
            "file": True,
            "max_size_mb": 10,
            "backup_count": 5,
        }

    def load_from_file(self, config_path: str) -> None:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to configuration file
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return

        try:
            with open(path) as f:
                config = yaml.safe_load(f)

            # Update configuration
            self._update_config(config)
            logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")

    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            config: Dictionary of configuration values
        """
        for section, values in config.items():
            if hasattr(self, section):
                current = getattr(self, section)
                if isinstance(current, dict):
                    current.update(values)
                else:
                    setattr(self, section, values)

    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation path.

        Args:
            path: Configuration path (e.g., 'thresholds.min_confidence')
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        try:
            value = self
            for key in path.split("."):
                if hasattr(value, key):
                    value = getattr(value, key)
                elif isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, AttributeError):
            return default

    def set(self, path: str, value: Any) -> None:
        """Set configuration value by dot-notation path.

        Args:
            path: Configuration path (e.g., 'thresholds.min_confidence')
            value: Value to set
        """
        keys = path.split(".")
        target = self

        # Navigate to the parent object
        for key in keys[:-1]:
            if hasattr(target, key):
                target = getattr(target, key)
            elif isinstance(target, dict):
                if key not in target:
                    target[key] = {}
                target = target[key]
            else:
                raise ValueError(f"Invalid config path: {path}")

        # Set the value
        if hasattr(target, keys[-1]):
            setattr(target, keys[-1], value)
        elif isinstance(target, dict):
            target[keys[-1]] = value
        else:
            raise ValueError(f"Invalid config path: {path}")

    def validate(self) -> List[str]:
        """Validate configuration values.

        Returns:
            List of validation error messages
        """
        errors = []

        # Validate thresholds
        if (
            self.thresholds["min_confidence"] < 0
            or self.thresholds["min_confidence"] > 1
        ):
            errors.append("min_confidence must be between 0 and 1")

        if self.thresholds["min_reviews"] < 1:
            errors.append("min_reviews must be at least 1")

        if self.thresholds["min_mentions"] < 1:
            errors.append("min_mentions must be at least 1")

        if self.thresholds["max_distance"] <= 0:
            errors.append("max_distance must be positive")

        # Validate languages
        enabled_languages = [
            lang for lang, config in self.languages.items() if config["enabled"]
        ]
        if not enabled_languages:
            errors.append("At least one language must be enabled")

        # Validate cache
        if self.cache["enabled"] and self.cache["max_age_hours"] < 0:
            errors.append("cache.max_age_hours must be non-negative")

        # Validate aspect weights
        for aspect, weight in self.aspects["weights"].items():
            if weight < 0 or weight > 1:
                errors.append(f"Weight for {aspect} must be between 0 and 1")

        # Validate aspect thresholds
        for aspect, threshold in self.aspects["thresholds"].items():
            if threshold < 0 or threshold > 1:
                errors.append(f"Threshold for {aspect} must be between 0 and 1")

        return errors


# Global configuration instance
config = Config()
