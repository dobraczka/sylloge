import logging
from importlib.metadata import version  # pragma: no cover

from .base import (
    BinaryEADataset,
    BinaryParquetEADataset,
    CacheableEADataset,
    MultiSourceEADataset,
    ParquetEADataset,
    ZipEADataset,
    ZipEADatasetWithPreSplitFolds,
)
from .id_mapped import IdMappedEADataset
from .med_bbk_loader import MED_BBK
from .moviegraph_benchmark_loader import MovieGraphBenchmark
from .oaei_loader import OAEI
from .open_ea_loader import OpenEA

__all__ = [
    "OpenEA",
    "MovieGraphBenchmark",
    "OAEI",
    "MED_BBK",
    "IdMappedEADataset",
    "MultiSourceEADataset",
    "BinaryEADataset",
    "BinaryParquetEADataset",
    "ParquetEADataset",
    "BinaryCacheableEADataset",
    "CacheableEADataset",
    "BinaryZipEADataset",
    "ZipEADataset",
    "BinaryZipEADatasetWithPreSplitFolds",
    "ZipEADatasetWithPreSplitFolds",
]
__version__ = version(__package__)
logging.getLogger(__name__).setLevel(logging.INFO)
