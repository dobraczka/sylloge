import logging
import os
import pathlib
from typing import Dict, List, Literal, Optional, Tuple, Union, cast, overload

import pandas as pd
from eche import PrefixedClusterHelper
from moviegraphbenchmark import load_data
from moviegraphbenchmark.loading import ERData

from .base import (
    BACKEND_LITERAL,
    BASE_DATASET_MODULE,
    CacheableEADataset,
    TrainTestValSplit,
)
from .my_typing import EA_SIDES, LinkType

logger = logging.getLogger(__name__)
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


GP_TO_DS_PREFIX: Dict[GraphPair, Tuple[str, ...]] = {
    IMDB_TMDB: (IMDB_PREFIX, TMDB_PREFIX),
    IMDB_TVDB: (IMDB_PREFIX, TVDB_PREFIX),
    TMDB_TVDB: (TMDB_PREFIX, TVDB_PREFIX),
    MULTI: (IMDB_PREFIX, TMDB_PREFIX, TVDB_PREFIX),
}


class MovieGraphBenchmark(CacheableEADataset[pd.DataFrame, LinkType]):
    """Class containing the movie graph benchmark.

    Published in `Obraczka, D. et. al. (2021) Embedding-Assisted Entity Resolution for Knowledge Graphs <http://ceur-ws.org/Vol-2873/paper8.pdf>`_,
    *Proceedings of the 2nd International Workshop on Knowledge Graph Construction co-located with 18th Extended Semantic Web Conference*
    """

    ent_links: LinkType
    folds: Optional[List[TrainTestValSplit[LinkType]]]

    @overload
    def __init__(
        self: "MovieGraphBenchmark[PrefixedClusterHelper]",
        graph_pair: GraphPair = "imdb-tmdb",
        use_cache: bool = True,
        cache_path: Optional[Union[str, pathlib.Path]] = None,
        use_cluster_helper: Literal[True] = True,
    ):
        ...

    @overload
    def __init__(
        self: "MovieGraphBenchmark[pd.DataFrame]",
        graph_pair: GraphPair = "imdb-tmdb",
        use_cache: bool = True,
        cache_path: Optional[Union[str, pathlib.Path]] = None,
        use_cluster_helper: Literal[False] = False,
    ):
        ...

    def __init__(
        self,
        graph_pair: GraphPair = "imdb-tmdb",
        use_cache: bool = True,
        cache_path: Optional[Union[str, pathlib.Path]] = None,
        use_cluster_helper: bool = True,
    ):
        """Initialize a MovieGraphBenchmark dataset.

        :param graph_pair: which graph pair to use of "imdb-tdmb","imdb-tvdb" or "tmdb-tvdb" or "multi" for multi-source setting
        :param use_cache: whether to use cache or not
        :param cache_path: Path where cache will be stored/loaded
        :param cache_path: Path where cache will be stored/loaded
        :param use_cluster_helper: if True uses ClusterHelper to load links
        :raises ValueError: if unknown graph pair
        """
        # Input validation.
        if graph_pair not in GRAPH_PAIRS:
            raise ValueError(f"Invalid graph pair: Allowed are: {GRAPH_PAIRS}")
        if not use_cluster_helper and graph_pair == MULTI:
            logging.info(
                "Must use ClusterHelper with multi setting! Will ignore the supplied option and use ClusterHelper!"
            )
            use_cluster_helper = True

        self.graph_pair = graph_pair

        actual_cache_path = self.create_cache_path(
            MOVIEGRAPH_MODULE, graph_pair, cache_path
        )
        ds_names = (
            tuple(self.graph_pair.split("-"))
            if graph_pair != MULTI
            else ("imdb", "tmdb", "tvdb")
        )
        super().__init__(  # type: ignore[misc, call-overload]
            cache_path=actual_cache_path,
            use_cache=use_cache,
            backend="pandas",
            dataset_names=ds_names,
            ds_prefix_tuples=GP_TO_DS_PREFIX[graph_pair],
            use_cluster_helper=use_cluster_helper,
        )
        if graph_pair != MULTI:
            self.rel_triples_left = self.rel_triples[0]
            self.rel_triples_right = self.rel_triples[1]
            self.attr_triples_left = self.attr_triples[0]
            self.attr_triples_right = self.attr_triples[1]

    def _read_links(self, data_path: str) -> Union[PrefixedClusterHelper, pd.DataFrame]:
        if self.use_cluster_helper:
            assert self._ds_prefixes
            return PrefixedClusterHelper.from_file(
                os.path.join(data_path, self.graph_pair, "cluster"),
                ds_prefixes=self._ds_prefixes,
            )  # type: ignore[return-value]
        return pd.read_csv(
            os.path.join(data_path, self.graph_pair, "ent_links"),
            sep="\t",
            names=EA_SIDES,
        )  # type: ignore[return-value]

    def _create_folds(self, ds: ERData) -> Optional[List[TrainTestValSplit]]:
        if self.graph_pair == MULTI:
            return None
        if not self.use_cluster_helper:
            return [
                TrainTestValSplit(
                    train=fold.train_links, test=fold.test_links, val=fold.valid_links
                )
                for fold in ds.folds
            ]
        assert self._ds_prefixes is not None
        return [
            TrainTestValSplit(
                train=PrefixedClusterHelper.from_numpy(
                    fold.train_links.to_numpy(), ds_prefixes=self._ds_prefixes
                ),
                test=PrefixedClusterHelper.from_numpy(
                    fold.test_links.to_numpy(), ds_prefixes=self._ds_prefixes
                ),
                val=PrefixedClusterHelper.from_numpy(
                    fold.valid_links.to_numpy(), ds_prefixes=self._ds_prefixes
                ),
            )
            for fold in ds.folds
        ]

    def initial_read(self, backend: BACKEND_LITERAL):
        assert self._ds_prefixes
        data_path = str(MOVIEGRAPH_MODULE.base)
        if self.graph_pair == MULTI:
            ds = load_data(pair=IMDB_TMDB, data_path=data_path)
            ds2 = load_data(pair=TMDB_TVDB, data_path=data_path)
            rel_triples = [ds.rel_triples_1, ds.rel_triples_2, ds2.rel_triples_2]
            attr_triples = [ds.attr_triples_1, ds.attr_triples_2, ds2.attr_triples_2]
            ent_links: Union[
                PrefixedClusterHelper, pd.DataFrame
            ] = PrefixedClusterHelper.from_file(
                os.path.join(data_path, "multi_source_cluster"),
                ds_prefixes=self._ds_prefixes,
            )
        else:
            ds = load_data(pair=self.graph_pair, data_path=data_path)
            rel_triples = [ds.rel_triples_1, ds.rel_triples_2]
            attr_triples = [ds.attr_triples_1, ds.attr_triples_2]
            ent_links = self._read_links(data_path)
        folds = self._create_folds(ds)
        return {
            "rel_triples": rel_triples,
            "attr_triples": attr_triples,
            "ent_links": cast(LinkType, ent_links),
            "folds": folds,
        }

    def __repr__(self) -> str:
        repr_str = super().__repr__()
        if self.graph_pair != MULTI:
            return repr_str.replace("triples_0", "triples_left").replace(
                "triples_1", "triples_right"
            )
        return repr_str

    @property
    def _canonical_name(self) -> str:
        return f"{self.__class__.__name__}_{self.graph_pair}"

    @property
    def _param_repr(self) -> str:
        return f"graph_pair={self.graph_pair}, "
