# UDA (User Dining Analysis)

A Python package designed for analyzing restaurant reviews and generating taste profiles through multilingual aspect-based sentiment analysis. It clusters reviews based on location and specific business profiles, while also identifying potential hotspots for new restaurant openings by analyzing the area's taste profile.

## Features

- Multilingual review analysis (English, Spanish)
- Aspect-based sentiment analysis for various dining aspects:
  - Taste (sweet, spicy, savory, etc.)
  - Texture (crunchy, smooth, etc.)
  - Ambiance (lighting, noise level, etc.)
  - Service (speed, friendliness, etc.)
  - Dietary considerations (kosher, halal, vegan, etc.)
- Price level and value ratio analysis
- Business clustering based on taste profiles
- Support for compound aspects (e.g., sweet-spicy, rich-creamy)
- Confidence scoring for all extracted aspects

## Installation

1. Clone the repository:
```bash
git clone https://github.com/msgfrom96/UDA.git
cd UDA
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
pip install -e .
```

## Project Structure

```
UDA/
├── src/
│   └── profiling/
│       ├── core/               # Core functionality
│       │   ├── analyzer.py     # Business logic
│       │   └── profile.py      # Profile representation
│       ├── extraction/         # Extraction functionality
│       │   ├── extractor.py    # Aspect extraction
│       │   └── sentiment.py    # Sentiment analysis
│       └── constants/          # Configuration
│           ├── aspects.py      # Aspect definitions
│           ├── languages.py    # Language support
│           └── thresholds.py   # Parameters
├── tests/                      # Test suite
├── examples/                   # Example scripts
└── docs/                      # Documentation
```

## Usage

Here's a simple example of how to use the package:

```python
from profiling import TasteProfileAnalyzer

# Initialize the analyzer
analyzer = TasteProfileAnalyzer()

# Analyze some reviews
reviews = [
    "The food was extremely spicy with authentic Thai flavors.",
    "Very fresh ingredients and quick service.",
    "The ambiance was romantic but the prices were too high."
]

# Process each review
for review in reviews:
    analyzer.analyze_review("restaurant_1", review)

# Get the taste profile
profile = analyzer.get_business_profile("restaurant_1")

# Print significant aspects
significant = profile.get_significant_aspects()
for category, aspects in significant.items():
    print(f"\n{category}:")
    for aspect, data in aspects.items():
        print(f"  {aspect}: {data['score']:.2f} (confidence: {data['confidence']:.2f})")
```

For more examples, check the `examples/` directory.

## Configuration

The package behavior can be customized through various constants in the `constants/` directory:

- `aspects.py`: Define aspect categories and keywords
- `languages.py`: Configure language support and translations
- `thresholds.py`: Adjust confidence thresholds and other parameters

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest tests/
```

4. Check code quality:
```bash
black src/ tests/
flake8 src/ tests/
mypy src/ tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

Please follow these guidelines:
- Use [Black](https://github.com/psf/black) for code formatting
- Add tests for new functionality
- Update documentation as needed
- Follow [Conventional Commits](https://www.conventionalcommits.org/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [NLTK](https://www.nltk.org/) for text processing
- [Hugging Face Transformers](https://huggingface.co/transformers/) for sentiment analysis
- [scikit-learn](https://scikit-learn.org/) for clustering
