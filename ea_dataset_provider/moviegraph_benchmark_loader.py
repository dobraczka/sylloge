from typing import Literal, Tuple

from moviegraphbenchmark import load_data

from .base import BASE_DATASET_MODULE, EADataset, TrainTestValSplit

MOVIEGRAPH_PATH = BASE_DATASET_MODULE.join("moviegraphbenchmark")

# graph pairs
GraphPair = Literal["imdb-tmdb", "imdb-tvdb", "tmdb-tvdb"]
IMDB_TMDB: GraphPair = "imdb-tmdb"
IMDB_TVDB: GraphPair = "imdb-tvdb"
TMDB_TVDB: GraphPair = "tmdb-tvdb"

GRAPH_PAIRS: Tuple[GraphPair, ...] = (IMDB_TMDB, IMDB_TVDB, TMDB_TVDB)


class MovieGraphBenchmark(EADataset):
    """Class containing the movie graph benchmark published in
    `Obraczka, D. et. al. (2021) Embedding-Assisted Entity Resolution for Knowledge Graphs <http://ceur-ws.org/Vol-2873/paper8.pdf>`_,
    *Proceedings of the 2nd International Workshop on Knowledge Graph Construction co-located with 18th Extended Semantic Web Conference*"""

    def __init__(
        self,
        graph_pair: str = "imdb-tmdb",
    ):
        """Initialize a MovieGraphBenchmark dataset.

        :param graph_pair: which graph pair to use of "imdb-tdmb","imdb-tvdb" or "tmdb-tvdb"
        :raises ValueError: if unknown graph pair
        """
        # Input validation.
        if graph_pair not in GRAPH_PAIRS:
            raise ValueError(f"Invalid graph pair: Allowed are: {GRAPH_PAIRS}")

        self.graph_pair = graph_pair
        ds = load_data(pair=graph_pair, data_path=str(MOVIEGRAPH_PATH))
        folds = [
            TrainTestValSplit(
                train=fold.train_links, test=fold.test_links, val=fold.valid_links
            )
            for fold in ds.folds
        ]
        super().__init__(
            rel_triples_left=ds.rel_triples_1,
            rel_triples_right=ds.rel_triples_2,
            attr_triples_left=ds.attr_triples_1,
            attr_triples_right=ds.attr_triples_2,
            ent_links=ds.ent_links,
            folds=folds,
        )

    def _param_repr(self) -> str:
        return f"graph_pair={self.graph_pair}, "
