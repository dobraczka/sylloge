import logging
from importlib.metadata import version  # pragma: no cover

from .moviegraph_benchmark_loader import MovieGraphBenchmark
from .open_ea_loader import OpenEA
from .oaei_loader import OAEI

__all__ = [
    "OpenEA",
    "MovieGraphBenchmark",
    "OAEI"
]
__version__ = version(__package__)
logging.getLogger(__name__).setLevel(logging.INFO)
