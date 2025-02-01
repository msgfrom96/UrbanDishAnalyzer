"""Constants related to thresholds and configuration values."""

from typing import Dict, Tuple

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.3
MIN_REVIEWS_THRESHOLD = 3

# Clustering parameters
MIN_CLUSTERS = 2
MAX_CLUSTERS = 5
SILHOUETTE_THRESHOLD = 0.15

# Model configuration
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
BATCH_SIZE = 32
CACHE_SIZE = 1024

# Sentiment ranges for classification
SENTIMENT_RANGES: Dict[str, Tuple[float, float]] = {
    "very_negative": (-1.0, -0.6),
    "negative": (-0.6, -0.2),
    "neutral": (-0.2, 0.2),
    "positive": (0.2, 0.6),
    "very_positive": (0.6, 1.0),
}

# Price modifiers for different contexts
PRICE_MODIFIERS: Dict[str, Dict[str, float]] = {
    "dining_style": {
        "fine_dining": 1.5,
        "casual": 0.7,
        "fast_food": 0.5,
        "bistro": 0.9,
    },
    "cuisine_type": {"kosher": 1.3, "organic": 1.2, "vegan": 1.1},
    "location_type": {"downtown": 1.2, "mall": 0.8, "food_court": 0.6},
}

# Price category thresholds
PRICE_CATEGORY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "$": (0.0, 0.25),
    "$$": (0.25, 0.5),
    "$$$": (0.5, 0.75),
    "$$$$": (0.75, 1.0),
}

# Value ratio thresholds
VALUE_RATIO_THRESHOLDS = {
    "excellent": 1.2,  # Quality significantly exceeds price
    "good": 1.0,  # Quality matches price
    "fair": 0.8,  # Quality slightly below price
    "poor": 0.6,  # Quality significantly below price
}

# Mixed sentiment threshold
MIXED_SENTIMENT_VARIANCE_THRESHOLD = 0.3

# Confidence boost per mention
CONFIDENCE_BOOST_PER_MENTION = 0.3

# Maximum confidence score
MAX_CONFIDENCE_SCORE = 1.0

# Minimum mentions for reliable scoring
MIN_MENTIONS_FOR_RELIABLE_SCORE = 3

# Review weight decay factor
REVIEW_WEIGHT_DECAY = 0.9  # For temporal weighting of reviews
