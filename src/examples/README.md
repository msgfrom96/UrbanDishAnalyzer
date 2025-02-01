# UDA Package Examples

This directory contains example scripts demonstrating various features of the Urban Dining Analysis (UDA) package.

## Available Examples

### 1. Basic Analysis (`basic_analysis.py`)
Demonstrates core functionality including:
- Loading and preprocessing restaurant data
- Generating taste profiles from reviews
- Basic visualization and reporting

```bash
python basic_analysis.py \
    --data-path data/ \
    --output-path output/basic_analysis/ \
    --metro-area "San Francisco"
```

### 2. Geographical Analysis (`geographical_analysis.py`)
Shows geographical analysis capabilities including:
- Restaurant clustering based on location and taste profiles
- Hotspot detection for new restaurant opportunities
- Interactive map generation

```bash
python geographical_analysis.py \
    --data-path data/ \
    --output-path output/geo_analysis/ \
    --metro-area "San Francisco" \
    --radius-km 10.0
```

### 3. Multilingual Analysis (`multilingual_analysis.py`)
Demonstrates multilingual analysis features including:
- Processing reviews in multiple languages
- Comparing taste profiles across languages
- Cross-language visualization and reporting

```bash
python multilingual_analysis.py \
    --output-path output/multilingual_analysis/
```

### 4. Yelp Dataset Analysis (`yelp_analysis.py`)
Shows how to analyze the Yelp Open Dataset including:
- Loading and filtering Yelp dataset files
- Analyzing restaurant reviews and profiles
- Detecting patterns and hotspots
- Generating visualizations and reports

```bash
python yelp_analysis.py \
    --data-dir data/yelp_dataset/ \
    --output-dir output/yelp_analysis/ \
    --metro-area "San Francisco" \
    --radius-km 10.0 \
    --min-reviews 5 \
    --max-age-days 365
```

## Data Requirements

### Basic Examples
The basic examples expect the following data files:
- `business.json`: Restaurant information (location, attributes, etc.)
- `review.json`: Review texts and metadata
- `metro_areas.geojson`: Metropolitan area boundaries

### Yelp Dataset
The Yelp dataset example requires the Yelp Open Dataset files:
- `business.json`: Business data including location and attributes
- `review.json`: Full review texts with metadata
- `user.json`: User information and metadata
- `checkin.json`: Business checkin data
- `tip.json`: Short tips/suggestions
- `photo.json`: Photo metadata

You can download the Yelp dataset from: https://www.yelp.com/dataset

## Output Structure

Each example script creates its own output directory with:
- Visualization files (PNG, HTML)
- Analysis reports (CSV, JSON)
- Log files

## Running the Examples

1. Install the UDA package:
```bash
pip install -e .
```

2. Prepare your data files:
   - For basic examples, use the sample data generator:
     ```bash
     python -m profiling.scripts.generate_samples
     ```
   - For Yelp analysis, download and extract the Yelp dataset

3. Run any example script with appropriate arguments

4. Check the output directory for results

## Customization

Each example can be modified to:
- Use different parameters
- Analyze different aspects
- Generate custom visualizations
- Export additional data

See the docstrings and comments in each script for details.
