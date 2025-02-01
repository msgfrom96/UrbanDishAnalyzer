"""Script to install required language models.

This script downloads and installs the required spaCy language models for:
- English (en_core_web_lg)
- Spanish (es_core_news_lg)
- French (fr_core_news_lg)
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from ..config import config
from ..logging import get_logger

logger = get_logger(__name__)

MODELS = [
    ("en_core_web_lg", "English"),
    ("es_core_news_lg", "Spanish"),
    ("fr_core_news_lg", "French"),
]


def check_model(model: str) -> bool:
    """Check if model is installed.

    Args:
        model: Model name

    Returns:
        True if model is installed
    """
    try:
        import spacy

        spacy.load(model)
        return True
    except:
        return False


def install_model(model: str) -> bool:
    """Install spaCy model.

    Args:
        model: Model name

    Returns:
        True if installation successful
    """
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> None:
    """Main entry point."""
    logger.info("Checking language models...")

    # Get enabled languages
    enabled_models = [
        (model, lang)
        for model, lang in MODELS
        if config.languages.get(lang.lower(), {}).get("enabled", False)
    ]

    if not enabled_models:
        logger.warning("No languages enabled in configuration")
        return

    # Check and install models
    for model, language in enabled_models:
        if check_model(model):
            logger.info(f"{language} model ({model}) already installed")
            continue

        logger.info(f"Installing {language} model ({model})...")
        if install_model(model):
            logger.info(f"{language} model installed successfully")
        else:
            logger.error(f"Failed to install {language} model")


if __name__ == "__main__":
    main()
