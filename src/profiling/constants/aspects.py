"""Constants related to aspects and their categorization."""

from typing import Dict, List, Tuple

# Aspect mapping for categorizing features
ASPECT_MAPPING: Dict[str, Dict[str, List[str]]] = {
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
        ],
        "dairy_free": [
            "dairy-free",
            "non-dairy",
            "no dairy",
            "sin lácteos",
            "sans lactose",
        ],
        "vegan": ["vegan", "plant-based", "no animal", "vegano", "végétalien"],
        "vegetarian": [
            "vegetarian",
            "meatless",
            "no meat",
            "vegetariano",
            "végétarien",
        ],
        "nut_free": ["nut-free", "no nuts", "sin nueces", "sans noix"],
        "shellfish_free": [
            "no shellfish",
            "shellfish-free",
            "sin mariscos",
            "sans fruits de mer",
        ],
        "kosher": ["kosher", "casher", "kashrut", "pareve", "chalav yisrael"],
        "halal": ["halal", "halaal", "حلال"],
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
        "seating_comfort": ["comfortable", "seating", "chairs", "cómodo", "asientos"],
        "service_speed": ["quick", "fast", "slow", "rápido", "lento", "veloz"],
        "cleanliness": ["clean", "spotless", "dirty", "limpio", "sucio", "inmaculado"],
        "accessibility": ["accessible", "handicap", "accesible", "discapacitado"],
        "friendly_staff": [
            "friendly",
            "nice",
            "helpful",
            "amable",
            "atento",
            "cordial",
        ],
        "family_friendly": ["family", "kid", "child", "familiar", "niños"],
        "romantic_ambiance": ["romantic", "intimate", "date", "romántico", "íntimo"],
    },
}

# Compound aspects with weights
COMPOUND_ASPECTS: Dict[str, Tuple[List[str], float]] = {
    "sweet_spicy": (["sweet", "spicy"], 1.5),
    "rich_creamy": (["rich", "creamy"], 1.3),
    "light_crispy": (["light", "crispy"], 1.2),
    "hot_juicy": (["hot", "juicy"], 1.4),
    "sweet_sour": (["sweet", "sour"], 1.3),
    "spicy_savory": (["spicy", "savory"], 1.4),
}

# Aspect weights for similarity calculations
ASPECT_WEIGHTS: Dict[str, float] = {
    "taste": 1.0,
    "food": 0.9,
    "dietary": 1.5,  # Increased weight for dietary restrictions
    "texture": 0.8,
    "ambiance": 0.6,
    "health": 0.7,
}

# Quality indicators for value calculation
QUALITY_INDICATORS: Dict[str, List[str]] = {
    "high": ["excellent", "outstanding", "exceptional", "premium", "exquisite"],
    "good": ["good", "nice", "decent", "solid"],
    "average": ["okay", "average", "fair", "mediocre"],
    "low": ["poor", "bad", "terrible", "awful"],
}

# Aspect relationships for context-aware analysis
ASPECT_RELATIONSHIPS: Dict[str, Dict[str, float]] = {
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
