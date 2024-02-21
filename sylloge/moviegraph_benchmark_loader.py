import pathlib
from typing import Literal, Optional, Tuple

from moviegraphbenchmark import load_data

from .base import (
    BACKEND_LITERAL,
    BASE_DATASET_MODULE,
    CacheableEADataset,
    TrainTestValSplit,
)

MOVIEGRAPH_MODULE = BASE_DATASET_MODULE.module("moviegraphbenchmark")

# graph pairs
GraphPair = Literal["imdb-tmdb", "imdb-tvdb", "tmdb-tvdb"]
IMDB_TMDB: GraphPair = "imdb-tmdb"
IMDB_TVDB: GraphPair = "imdb-tvdb"
TMDB_TVDB: GraphPair = "tmdb-tvdb"

GRAPH_PAIRS: Tuple[GraphPair, ...] = (IMDB_TMDB, IMDB_TVDB, TMDB_TVDB)


class MovieGraphBenchmark(CacheableEADataset):
    """Class containing the movie graph benchmark.

    Published in `Obraczka, D. et. al. (2021) Embedding-Assisted Entity Resolution for Knowledge Graphs <http://ceur-ws.org/Vol-2873/paper8.pdf>`_,
    *Proceedings of the 2nd International Workshop on Knowledge Graph Construction co-located with 18th Extended Semantic Web Conference*
    """

    def __init__(
        self,
        graph_pair: GraphPair = "imdb-tmdb",
        backend: BACKEND_LITERAL = "pandas",
        npartitions: int = 1,
        use_cache: bool = True,
        cache_path: Optional[pathlib.Path] = None,
    ):
        """Initialize a MovieGraphBenchmark dataset.

        :param graph_pair: which graph pair to use of "imdb-tdmb","imdb-tvdb" or "tmdb-tvdb"
        :param backend: Whether to use "pandas" or "dask"
        :param npartitions: how many partitions to use for each frame, when using dask
        :param use_cache: whether to use cache or not
        :param cache_path: Path where cache will be stored/loaded
        :raises ValueError: if unknown graph pair
        """
        # Input validation.
        if graph_pair not in GRAPH_PAIRS:
            raise ValueError(f"Invalid graph pair: Allowed are: {GRAPH_PAIRS}")

        self.graph_pair = graph_pair

        actual_cache_path = self.create_cache_path(
            MOVIEGRAPH_MODULE, graph_pair, cache_path
        )
        left_name, right_name = self.graph_pair.split("-")
        super().__init__(
            cache_path=actual_cache_path,
            use_cache=use_cache,
            backend=backend,
            npartitions=npartitions,
            dataset_names=(left_name, right_name),
        )

    def initial_read(self, backend: BACKEND_LITERAL):
        ds = load_data(pair=self.graph_pair, data_path=str(MOVIEGRAPH_MODULE.base))
        folds = [
            TrainTestValSplit(
                train=fold.train_links, test=fold.test_links, val=fold.valid_links
            )
            for fold in ds.folds
        ]
        return {
            "rel_triples_left": ds.rel_triples_1,
            "rel_triples_right": ds.rel_triples_2,
            "attr_triples_left": ds.attr_triples_1,
            "attr_triples_right": ds.attr_triples_2,
            "ent_links": ds.ent_links,
            "folds": folds,
        }

    @property
    def _canonical_name(self) -> str:
        return f"{self.__class__.__name__}_{self.graph_pair}"

    @property
    def _param_repr(self) -> str:
        return f"graph_pair={self.graph_pair}, "
