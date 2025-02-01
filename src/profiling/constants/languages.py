"""Constants related to language processing and translations."""

from typing import Dict, List, Set

# Supported languages for analysis
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese",
}

# Negation words by language
NEGATION_WORDS: Dict[str, Set[str]] = {
    "en": {"not", "no", "never", "none", "neither", "nor", "n't", "without"},
    "es": {"no", "nunca", "jamás", "ni", "sin", "ninguno"},
    "fr": {"ne", "pas", "jamais", "aucun", "sans"},
    "it": {"non", "no", "mai", "nessuno", "senza"},
}

# Intensity modifiers by language
INTENSITY_MODIFIERS: Dict[str, float] = {
    # English
    "very": 1.2,
    "really": 1.2,
    "extremely": 1.3,
    "incredibly": 1.3,
    "super": 1.2,
    "quite": 1.1,
    "somewhat": 0.8,
    "slightly": 0.7,
    "a bit": 0.8,
    "barely": 0.6,
    # Spanish
    "muy": 1.2,
    "realmente": 1.2,
    "extremadamente": 1.3,
    "increíblemente": 1.3,
    "súper": 1.2,
    "bastante": 1.1,
    "algo": 0.8,
    "ligeramente": 0.7,
    "un poco": 0.8,
    "apenas": 0.6,
}

# Multilingual price-related keywords
PRICE_KEYWORDS: Dict[str, Dict[str, float]] = {
    "en": {
        "expensive": 1.0,
        "pricey": 0.8,
        "costly": 0.8,
        "overpriced": -0.8,
        "cheap": 0.2,
        "affordable": 0.4,
        "reasonable": 0.5,
        "value": 0.5,
        "worth": 0.5,
        "budget": 0.3,
        "high-end": 0.9,
        "luxury": 1.0,
        "$": 0.25,
        "$$": 0.5,
        "$$$": 0.75,
        "$$$$": 1.0,
    },
    "es": {
        "caro": 0.8,
        "costoso": 0.8,
        "económico": 0.2,
        "asequible": 0.4,
        "razonable": 0.5,
        "valor": 0.5,
        "precio": 0.5,
        "barato": 0.2,
        "lujoso": 1.0,
        "calidad-precio": 0.5,
    },
    "fr": {
        "cher": 0.8,
        "coûteux": 0.8,
        "abordable": 0.4,
        "raisonnable": 0.5,
        "économique": 0.2,
        "bon marché": 0.2,
        "luxueux": 1.0,
        "prix": 0.5,
        "rapport qualité-prix": 0.5,
    },
    "it": {
        "costoso": 0.8,
        "caro": 0.8,
        "economico": 0.2,
        "conveniente": 0.4,
        "ragionevole": 0.5,
        "lussuoso": 1.0,
        "prezzo": 0.5,
        "qualità-prezzo": 0.5,
    },
}

# Multilingual dining style keywords
DINING_STYLE_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "en": {
        "casual": ["casual", "relaxed", "informal", "laid-back"],
        "fine_dining": [
            "elegant",
            "upscale",
            "fine dining",
            "sophisticated",
            "gourmet",
        ],
        "fast_food": ["fast food", "quick service", "takeout", "drive-thru"],
        "bistro": ["bistro", "café", "brasserie"],
    },
    "es": {
        "casual": ["casual", "relajado", "informal"],
        "fine_dining": ["elegante", "refinado", "gourmet", "alta cocina"],
        "fast_food": ["comida rápida", "para llevar"],
        "bistro": ["bistró", "café", "cafetería"],
    },
    "fr": {
        "casual": ["décontracté", "relax", "informel"],
        "fine_dining": ["gastronomique", "raffiné", "haute cuisine"],
        "fast_food": ["restauration rapide", "à emporter"],
        "bistro": ["bistrot", "café", "brasserie"],
    },
    "it": {
        "casual": ["casual", "informale", "rilassato"],
        "fine_dining": ["elegante", "raffinato", "alta cucina"],
        "fast_food": ["fast food", "cibo veloce"],
        "bistro": ["bistrot", "caffè", "trattoria"],
    },
}

# Multilingual certification keywords
CERTIFICATION_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "en": {
        "kosher_certified": [
            "certified kosher",
            "kosher certified",
            "strictly kosher",
            "kashrut",
        ],
        "halal_certified": ["certified halal", "halal certified", "strictly halal"],
    },
    "es": {
        "kosher_certified": ["certificado kosher", "kosher certificado"],
        "halal_certified": ["certificado halal", "halal certificado"],
    },
    "fr": {
        "kosher_certified": ["certifié casher", "casher certifié"],
        "halal_certified": ["certifié halal", "halal certifié"],
    },
    "it": {
        "kosher_certified": ["certificato kosher", "kosher certificato"],
        "halal_certified": ["certificato halal", "halal certificato"],
    },
}
