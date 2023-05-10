import logging
from importlib.metadata import version  # pragma: no cover

from .base import EADataset
from .id_mapped import IdMappedEADataset
from .moviegraph_benchmark_loader import MovieGraphBenchmark
from .open_ea_loader import OpenEA
from .oaei_loader import OAEI

__all__ = [
    "OpenEA",
    "MovieGraphBenchmark",
<<<<<<< HEAD
    "OAEI"
=======
    "IdMappedEADataset",
    "EADataset",
>>>>>>> main
]
__version__ = version(__package__)
logging.getLogger(__name__).setLevel(logging.INFO)
