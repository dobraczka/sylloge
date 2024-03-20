import os
import pathlib
from collections import OrderedDict
from typing import Dict, Literal, Optional, Tuple

import pandas as pd
from eche import PrefixedClusterHelper
from moviegraphbenchmark import load_data

from .base import (
    BACKEND_LITERAL,
    BASE_DATASET_MODULE,
    CacheableEADataset,
    TrainTestValSplit,
)

MOVIEGRAPH_MODULE = BASE_DATASET_MODULE.module("moviegraphbenchmark")

# graph pairs
GraphPair = Literal["imdb-tmdb", "imdb-tvdb", "tmdb-tvdb", "multi"]
IMDB_TMDB: GraphPair = "imdb-tmdb"
IMDB_TVDB: GraphPair = "imdb-tvdb"
TMDB_TVDB: GraphPair = "tmdb-tvdb"
MULTI: GraphPair = "multi"

GRAPH_PAIRS: Tuple[GraphPair, ...] = (IMDB_TMDB, IMDB_TVDB, TMDB_TVDB, MULTI)

# graph names
GraphName = Literal["imdb", "tmdb", "tvdb"]
IMDB: GraphName = "imdb"
TMDB: GraphName = "tmdb"
TVDB: GraphName = "tvdb"

# prefixes
BASE_PREFIX = "https://www.scads.de/movieBenchmark/resource/"
IMDB_PREFIX = f"{BASE_PREFIX}IMDB/"
TMDB_PREFIX = f"{BASE_PREFIX}TMDB/"
TVDB_PREFIX = f"{BASE_PREFIX}TVDB/"


GP_TO_DS_PREFIX: Dict[GraphPair, OrderedDict[str, str]] = {
    IMDB_TMDB: OrderedDict({IMDB: IMDB_PREFIX, TMDB: TMDB_PREFIX}),
    IMDB_TVDB: OrderedDict({IMDB: IMDB_PREFIX, TVDB: TVDB_PREFIX}),
    TMDB_TVDB: OrderedDict({TMDB: TMDB_PREFIX, TVDB: TVDB_PREFIX}),
    MULTI: OrderedDict({IMDB: IMDB_PREFIX, TMDB: TMDB_PREFIX, TVDB: TVDB_PREFIX}),
}


class MovieGraphBenchmark(CacheableEADataset[pd.DataFrame]):
    """Class containing the movie graph benchmark.

    Published in `Obraczka, D. et. al. (2021) Embedding-Assisted Entity Resolution for Knowledge Graphs <http://ceur-ws.org/Vol-2873/paper8.pdf>`_,
    *Proceedings of the 2nd International Workshop on Knowledge Graph Construction co-located with 18th Extended Semantic Web Conference*
    """

    ent_links: PrefixedClusterHelper

    def __init__(
        self,
        graph_pair: GraphPair = "imdb-tmdb",
        use_cache: bool = True,
        cache_path: Optional[pathlib.Path] = None,
    ):
        """Initialize a MovieGraphBenchmark dataset.

        :param graph_pair: which graph pair to use of "imdb-tdmb","imdb-tvdb" or "tmdb-tvdb" or "multi" for multi-source setting
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
        ds_names = (
            tuple(self.graph_pair.split("-"))
            if graph_pair != MULTI
            else ("imdb", "tmdb", "tvdb")
        )
        super().__init__(
            cache_path=actual_cache_path,
            use_cache=use_cache,
            backend="pandas",
            dataset_names=ds_names,
        )

    def initial_read(self, backend: BACKEND_LITERAL):
        ds_prefixes = GP_TO_DS_PREFIX[self.graph_pair]
        data_path = str(MOVIEGRAPH_MODULE.base)
        if self.graph_pair == MULTI:
            ds = load_data(pair=IMDB_TMDB, data_path=data_path)
            ds2 = load_data(pair=TMDB_TVDB, data_path=data_path)
            rel_triples = [ds.rel_triples_1, ds.rel_triples_2, ds2.rel_triples_2]
            attr_triples = [ds.attr_triples_1, ds.attr_triples_2, ds2.attr_triples_2]
            ent_links = PrefixedClusterHelper.from_file(
                os.path.join(data_path, "multi_source_cluster"), ds_prefixes=ds_prefixes
            )
        else:
            ds = load_data(pair=self.graph_pair, data_path=data_path)
            rel_triples = [ds.rel_triples_1, ds.rel_triples_2]
            attr_triples = [ds.attr_triples_1, ds.attr_triples_2]
            ent_links = PrefixedClusterHelper.from_file(
                os.path.join(data_path, self.graph_pair, "cluster"),
                ds_prefixes=ds_prefixes,
            )
        folds = [
            TrainTestValSplit(
                train=PrefixedClusterHelper.from_numpy(
                    fold.train_links.to_numpy(), ds_prefixes=ds_prefixes
                ),
                test=PrefixedClusterHelper.from_numpy(
                    fold.test_links.to_numpy(), ds_prefixes=ds_prefixes
                ),
                val=PrefixedClusterHelper.from_numpy(
                    fold.valid_links.to_numpy(), ds_prefixes=ds_prefixes
                ),
            )
            for fold in ds.folds
        ]
        return {
            "rel_triples": rel_triples,
            "attr_triples": attr_triples,
            "ent_links": ent_links,
            "folds": folds,
        }

    @property
    def _canonical_name(self) -> str:
        return f"{self.__class__.__name__}_{self.graph_pair}"

    @property
    def _param_repr(self) -> str:
        return f"graph_pair={self.graph_pair}, "
