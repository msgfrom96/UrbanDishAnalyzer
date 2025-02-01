"""Setup script for the profiling package."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="profiling",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Yelp Dataset Analysis for identifying taste profile hotspots",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/profiling",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "torch>=1.9.0",
        "transformers>=4.9.1",
        "nltk>=3.6.2",
        "langdetect>=1.0.9",
        "black>=21.7b0",
        "flake8>=3.9.2",
        "mypy>=0.910",
        "pytest>=6.2.5",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.7b0",
            "flake8>=3.9.2",
            "mypy>=0.910",
            "isort>=5.9.3",
            "pre-commit>=2.15.0",
            "coverage>=6.0.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "profiling=profiling.main:main",
        ],
    },
)
