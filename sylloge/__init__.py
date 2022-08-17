import logging
from importlib.metadata import version  # pragma: no cover

from .moviegraph_benchmark_loader import MovieGraphBenchmark
from .open_ea_loader import OpenEA

__all__ = [
    "OpenEA",
    "MovieGraphBenchmark",
]
__version__ = version(__package__)
logging.getLogger(__name__).setLevel(logging.INFO)
