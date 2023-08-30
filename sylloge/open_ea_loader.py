# largely adapted from pykeen.datasets.ea.openea
import pathlib
from typing import Dict, Literal, Tuple, Optional

from .base import BACKEND_LITERAL, BASE_DATASET_MODULE, ZipEADatasetWithPreSplitFolds

OPEN_EA_MODULE = BASE_DATASET_MODULE.module("open_ea")

# graph pairs
GraphPair = Literal["D_W", "D_Y", "EN_DE", "EN_FR"]
D_W: GraphPair = "D_W"
D_Y: GraphPair = "D_Y"
EN_DE: GraphPair = "EN_DE"
EN_FR: GraphPair = "EN_FR"
GRAPH_PAIRS: Tuple[GraphPair, ...] = (D_W, D_Y, EN_DE, EN_FR)

# graph sizes
GraphSize = Literal["15K", "100K"]
SIZE_15K: GraphSize = "15K"
SIZE_100K: GraphSize = "100K"
GRAPH_SIZES = (SIZE_15K, SIZE_100K)

# graph versions
GraphVersion = Literal["V1", "V2"]
V1: GraphVersion = "V1"
V2: GraphVersion = "V2"
GRAPH_VERSIONS = (V1, V2)


class OpenEA(ZipEADatasetWithPreSplitFolds):
    """Class containing the OpenEA dataset family.
    Published in `Sun, Z. et. al. (2020) A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs <http://www.vldb.org/pvldb/vol13/p2326-sun.pdf>`_,
    *Proceedings of the VLDB Endowment*"""

    #: The link to the zip file
    _FIGSHARE_LINK: str = "https://figshare.com/ndownloader/files/34234391"

    #: The hex digest for the zip file
    _SHA512: str = (
        "c1589f185f86e05c497de147b4d6c243c66775cb4b50c6b41ecc71b36cfafb4c"
        "9f86fbee94e1e78a7ee056dd69df1ce3fc210ae07dc64955ad2bfda7450545ef"
    )

    _GRAPH_PAIR_TO_DS_NAMES: Dict[GraphPair, Tuple[str, str]] = {
        "D_W": ("DBpedia", "Wikidata"),
        "D_Y": ("DBpedia", "YAGO"),
        "EN_DE": ("DBpedia_EN", "DBpedia_DE"),
        "EN_FR": ("DBpedia_EN", "DBpedia_FR"),
    }

    def __init__(
        self,
        graph_pair: GraphPair = "D_W",
        size: GraphSize = "15K",
        version: GraphVersion = "V1",
        backend: BACKEND_LITERAL = "pandas",
        npartitions: int = 1,
        use_cache: bool = True,
        cache_path: Optional[pathlib.Path] = None,
    ):
        """Initializes an OpenEA dataset.

        :param graph_pair: which pair to use of "D_W", "D_Y", "EN_DE" or "EN_FR"
        :param size: what size ("15K" or "100K")
        :param version: which version to use ("V1" or "V2")
        :param backend: Whether to use "pandas" or "dask"
        :param npartitions: how many partitions to use for each frame, when using dask
        :raises ValueError: if unknown graph_pair,size or version values are provided
        """
        # Input validation.
        if graph_pair not in GRAPH_PAIRS:
            raise ValueError(f"Invalid graph pair: Allowed are: {GRAPH_PAIRS}")
        if size not in GRAPH_SIZES:
            raise ValueError(f"size must be one of {GRAPH_SIZES}")
        if version not in GRAPH_VERSIONS:
            raise ValueError(f"version must be one of {GRAPH_VERSIONS}")

        self.graph_pair = graph_pair
        self.size = size
        self.version = version

        # ensure zip file is present
        zip_path = OPEN_EA_MODULE.ensure(
            url=OpenEA._FIGSHARE_LINK,
            name="OpenEA_dataset_v2.0.zip",
            download_kwargs=dict(hexdigests=dict(sha512=OpenEA._SHA512)),
        )

        # save relative paths beforehand so they are present for loading
        inner_cache_path = f"{graph_pair}_{size}_{version}"
        inner_path = pathlib.PurePosixPath(
            "OpenEA_dataset_v2.0",
            inner_cache_path
        )
        actual_cache_path = self.create_cache_path(OPEN_EA_MODULE, inner_cache_path, cache_path)
        super().__init__(
            cache_path=actual_cache_path,
            use_cache=use_cache,
            zip_path=zip_path,
            inner_path=inner_path,
            backend=backend,
            npartitions=npartitions,
            dataset_names=OpenEA._GRAPH_PAIR_TO_DS_NAMES[graph_pair],
        )

    @property
    def _canonical_name(self) -> str:
        return f"{self.__class__.__name__}_{self.graph_pair}_{self.size}_{self.version}"

    @property
    def _param_repr(self) -> str:
        return (
            f"graph_pair={self.graph_pair}, size={self.size}, version={self.version}, "
        )
