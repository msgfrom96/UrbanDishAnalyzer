"""Constants for taste profile analysis."""

# Aspect mapping for categorizing features
ASPECT_MAPPING = {
    "taste": {
        "sweet": ["sweet", "sugary", "dulce", "azucarado", "suave", "sucroso"],
        "salty": ["salty", "salt", "salado", "sal", "salino"],
        "spicy": ["spicy", "hot", "heat", "picante", "chile", "picante"],
        "savory": ["savory", "umami", "rich", "sabroso", "delicioso", "sabroso"],
        "bitter": ["bitter", "amargo", "agrio"],
        "sour": ["sour", "acidic", "tart", "ácido", "agrio", "ácido"],
        "flavor_intensity": [
            "flavorful",
            "bland",
            "intense",
            "mild",
            "strong",
            "weak",
            "sabor",
            "saboroso",
        ],
        "complexity": ["complex", "simple", "layered", "subtle", "complejo", "sutil"],
        "balance": [
            "balanced",
            "harmonious",
            "overwhelming",
            "equilibrado",
            "armonioso",
        ],
    },
    "food": {
        "portion_size": [
            "portion",
            "serving",
            "amount",
            "porción",
            "ración",
            "cantidad",
        ],
        "freshness": ["fresh", "freshly", "fresco", "recién", "fresquito"],
        "temperature": [
            "temperature",
            "warm",
            "cold",
            "temperatura",
            "caliente",
            "frío",
            "templado",
        ],
        "plating_aesthetics": [
            "presentation",
            "plating",
            "beautiful",
            "presentación",
            "plato",
            "estética",
        ],
        "variety": [
            "variety",
            "selection",
            "options",
            "variedad",
            "selección",
            "opciones",
        ],
        "authenticity": [
            "authentic",
            "traditional",
            "auténtico",
            "tradicional",
            "genuino",
        ],
    },
    "texture": {
        "crunchiness": ["crunchy", "crispy", "crujiente", "crocante", "crujidor"],
        "smoothness": ["smooth", "silky", "suave", "sedoso", "liso"],
        "chewiness": ["chewy", "tender", "tierno", "masticable", "masticador"],
        "creaminess": ["creamy", "velvety", "cremoso", "aterciopelado", "cremosito"],
        "firmness": ["firm", "solid", "firme", "sólido", "duro"],
        "juiciness": ["juicy", "moist", "jugoso", "húmedo", "jugosito"],
        "softness": ["soft", "tender", "blando", "suave", "blandito"],
    },
    "dietary": {
        "gluten_free": [
            "gluten-free",
            "gluten free",
            "no gluten",
            "sin gluten",
            "sans gluten",
            "sin gluten",
        ],
        "dairy_free": [
            "dairy-free",
            "non-dairy",
            "no dairy",
            "sin lácteos",
            "sans lactose",
            "sin lácteos",
        ],
        "vegan": [
            "vegan",
            "plant-based",
            "no animal",
            "vegano",
            "végétalien",
            "vegetal",
        ],
        "vegetarian": [
            "vegetarian",
            "meatless",
            "no meat",
            "vegetariano",
            "végétarien",
            "vegetal",
        ],
        "nut_free": [
            "nut-free",
            "no nuts",
            "sin nueces",
            "sans noix",
            "sin frutos secos",
        ],
        "shellfish_free": [
            "no shellfish",
            "shellfish-free",
            "sin mariscos",
            "sans fruits de mer",
            "sin mariscos",
        ],
        "kosher": ["kosher", "casher", "kashrut", "pareve", "chalav yisrael", "kosher"],
        "halal": ["halal", "halaal", "حلال", "halal", "halal"],
    },
    "health": {
        "health_consciousness": [
            "healthy",
            "nutritious",
            "saludable",
            "nutritivo",
            "sano",
        ],
        "organic": ["organic", "natural", "orgánico", "natural", "biológico"],
    },
    "ambiance": {
        "lighting_quality": [
            "lighting",
            "bright",
            "dim",
            "iluminación",
            "luz",
            "brillante",
        ],
        "noise_level": ["quiet", "noisy", "loud", "silencioso", "ruidoso", "tranquilo"],
        "seating_comfort": [
            "comfortable",
            "seating",
            "chairs",
            "cómodo",
            "asientos",
            "confortable",
        ],
        "service_speed": ["quick", "fast", "slow", "rápido", "lento", "veloz"],
        "cleanliness": ["clean", "spotless", "dirty", "limpio", "sucio", "inmaculado"],
        "accessibility": [
            "accessible",
            "handicap",
            "accesible",
            "discapacitado",
            "accesible",
        ],
        "friendly_staff": [
            "friendly",
            "nice",
            "helpful",
            "amable",
            "atento",
            "cordial",
        ],
        "family_friendly": ["family", "kid", "child", "familiar", "niños", "familiar"],
        "romantic_ambiance": [
            "romantic",
            "intimate",
            "date",
            "romántico",
            "íntimo",
            "romántico",
        ],
    },
}

# Minimum confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.3
MIN_REVIEWS_THRESHOLD = 3

# Clustering parameters
MIN_CLUSTERS = 2
MAX_CLUSTERS = 5
SILHOUETTE_THRESHOLD = 0.15  # Increased for better cluster separation

# Language detection
SUPPORTED_LANGUAGES = {
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

# Sentiment analysis
SENTIMENT_RANGES = {
    "very_negative": (-1.0, -0.6),
    "negative": (-0.6, -0.2),
    "neutral": (-0.2, 0.2),
    "positive": (0.2, 0.6),
    "very_positive": (0.6, 1.0),
}

# Intensity modifiers
INTENSITY_MODIFIERS = {
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

# Negation words by language
NEGATION_WORDS = {
    "en": {"not", "no", "never", "none", "neither", "nor", "n't", "without"},
    "es": {"no", "nunca", "jamás", "ni", "sin", "ninguno"},
    "fr": {"ne", "pas", "jamais", "aucun", "sans"},
    "it": {"non", "no", "mai", "nessuno", "senza"},
}

# Aspect weights for similarity calculations
ASPECT_WEIGHTS = {
    "taste": 1.0,
    "food": 0.9,
    "dietary": 1.5,  # Increased weight for dietary restrictions
    "texture": 0.8,
    "ambiance": 0.6,
    "health": 0.7,
}

# Add compound aspects
COMPOUND_ASPECTS = {
    "sweet_spicy": (["sweet", "spicy"], 1.5),
    "rich_creamy": (["rich", "creamy"], 1.3),
    "light_crispy": (["light", "crispy"], 1.2),
    "hot_juicy": (["hot", "juicy"], 1.4),
    "sweet_sour": (["sweet", "sour"], 1.3),
    "spicy_savory": (["spicy", "savory"], 1.4),
}

# Add aspect relationships
ASPECT_RELATIONSHIPS = {
    "fine_dining": {
        "price_level": 1.5,
        "plating_aesthetics": 1.3,
        "service_quality": 1.4,
        "ambiance": 1.3,
        "cleanliness": 1.2,
    },
    "kosher": {
        "price_level": 1.2,
        "authenticity": 1.3,
        "cleanliness": 1.2,
        "quality": 1.1,
    },
    "family_friendly": {
        "portion_size": 1.2,
        "noise_level": 0.8,
        "price_level": 0.7,
        "service_speed": 1.1,
    },
    "casual": {"price_level": 0.6, "portion_size": 1.1, "service_speed": 1.2},
}

# Add quality indicators
QUALITY_INDICATORS = {
    "high": ["excellent", "outstanding", "exceptional", "premium", "exquisite"],
    "good": ["good", "nice", "decent", "solid"],
    "average": ["okay", "average", "fair", "mediocre"],
    "low": ["poor", "bad", "terrible", "awful"],
}

# Add price modifiers
PRICE_MODIFIERS = {
    "dining_style": {
        "fine_dining": 1.5,
        "casual": 0.7,
        "fast_food": 0.5,
        "bistro": 0.9,
    },
    "cuisine_type": {"kosher": 1.3, "organic": 1.2, "vegan": 1.1},
    "location_type": {"downtown": 1.2, "mall": 0.8, "food_court": 0.6},
}

# Multilingual price-related keywords
PRICE_KEYWORDS = {
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
DINING_STYLE_KEYWORDS = {
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
CERTIFICATION_KEYWORDS = {
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
