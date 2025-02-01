"""UDA package initialization."""

from profiling.analysis.taste_profile import TasteProfileAnalyzer
from profiling.data.loader import DataLoader
from profiling.data.storage import DataStorage
from profiling.visualization.maps import MapVisualizer
from profiling.visualization.reports import ReportGenerator

__version__ = "0.1.0"

__all__ = [
    "TasteProfileAnalyzer",
    "DataLoader",
    "DataStorage",
    "ReportGenerator",
    "MapVisualizer",
]
