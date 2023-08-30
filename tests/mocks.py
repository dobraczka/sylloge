from typing import Iterable

import dask.bag as db
import dask.dataframe as dd
import pandas as pd
from moviegraphbenchmark.loading import ERData, Fold
from strawman import dummy_df, dummy_triples
from util import DatasetStatistics

from sylloge.typing import EA_SIDE_LEFT, EA_SIDE_RIGHT, EA_SIDES


class ResourceMocker:
    def __init__(
        self, statistic: DatasetStatistics = None, fraction: float = 1.0, seed: int = 17
    ):
        if statistic is None:
            self.statistic = DatasetStatistics(
                num_rel_triples_left=15,
                num_rel_triples_right=15,
                num_attr_triples_left=15,
                num_attr_triples_right=15,
                num_ent_links=15,
            )
        else:
            self.statistic = statistic
        self.fraction = fraction
        self.seed = seed

    def mock_read(self, path: str, names: Iterable):
        return self.mock_read_zipfile_csv(path="", inner_path=path)

    def mock_load_data(self, pair: str, data_path: str) -> ERData:
        ent_links = self.mock_read_zipfile_csv("ent_links")
        train_links = ent_links[:2]
        test_links = ent_links[2:5]
        valid_links = ent_links[5:]
        return ERData(
            attr_triples_1=self.mock_read_zipfile_csv("attr_triples_1"),
            attr_triples_2=self.mock_read_zipfile_csv("attr_triples_2"),
            rel_triples_1=self.mock_read_zipfile_csv("rel_triples_1"),
            rel_triples_2=self.mock_read_zipfile_csv("rel_triples_2"),
            ent_links=self.mock_read_zipfile_csv("ent_links"),
            folds=[
                Fold(
                    train_links=train_links,
                    test_links=test_links,
                    valid_links=valid_links,
                )
                for _ in range(5)
            ],
        )

    def mock_read_zipfile_csv(self, inner_path: str, **kwargs) -> pd.DataFrame:
        if "rel_triples_1" in inner_path:
            return dummy_triples(
                int(self.statistic.num_rel_triples_left * self.fraction),
                entity_prefix=EA_SIDE_LEFT,
                seed=self.seed,
            )
        elif "rel_triples_2" in inner_path:
            return dummy_triples(
                int(self.statistic.num_rel_triples_right * self.fraction),
                entity_prefix=EA_SIDE_RIGHT,
                seed=self.seed,
            )
        elif "attr_triples_1" in inner_path:
            return dummy_triples(
                int(self.statistic.num_attr_triples_left * self.fraction),
                entity_prefix=EA_SIDE_LEFT,
                relation_triples=False,
                seed=self.seed,
            )
        elif "attr_triples_2" in inner_path:
            return dummy_triples(
                int(self.statistic.num_attr_triples_right * self.fraction),
                entity_prefix=EA_SIDE_RIGHT,
                relation_triples=False,
                seed=self.seed,
            )
        elif "_links" in inner_path:
            return dummy_df(
                shape=(int(self.statistic.num_ent_links * self.fraction), 2),
                content_length=100,
                columns=list(EA_SIDES),
                seed=self.seed,
            )

    def mock_read_dask_bag_from_archive_text(
        self, archive_path: str, inner_path: str, protocol: str
    ) -> db.Bag:
        if "ref" in inner_path:
            return db.from_sequence(
                [
                    (
                        "_:N7a75889e6f604aef9e0c2937692822d9 <http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity1> <http://dbkwik.webdatacommons.org/starwars.wikia.com/property/weapons> .\n"
                    ),
                    (
                        "_:N7a75889e6f604aef9e0c2937692822d9 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://knowledgeweb.semanticweb.org/heterogeneity/alignmentCell> .\n"
                    ),
                    (
                        '_:N7a75889e6f604aef9e0c2937692822d9 <http://knowledgeweb.semanticweb.org/heterogeneity/alignmentrelation> "=" .\n'
                    ),
                    (
                        "_:N937bec96d6684a2b9b7af88509bd07b4 <http://knowledgeweb.semanticweb.org/heterogeneity/alignmentmap> _:N7a75889e6f604aef9e0c2937692822d9 .\n"
                    ),
                    (
                        '_:N7a75889e6f604aef9e0c2937692822d9 <http://knowledgeweb.semanticweb.org/heterogeneity/alignmentmeasure> "1.0"^^<xsd:float> .\n'
                    ),
                    (
                        "_:N7a75889e6f604aef9e0c2937692822d9 <http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity2> <http://dbkwik.webdatacommons.org/swg.wikia.com/property/weapons> .\n"
                    ),
                    (
                        "_:Na53a488f3e0648d6a181817a992fb395 <http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity2> <http://dbkwik.webdatacommons.org/swg.wikia.com/resource/TransGalMeg_%22Ixiyen%22_Fast_Attack_Craft> .\n"
                    ),
                    (
                        "_:Na53a488f3e0648d6a181817a992fb395 <http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity1> <http://dbkwik.webdatacommons.org/starwars.wikia.com/resource/Ixiyen-class_fast_attack_craft> .\n"
                    ),
                    (
                        "_:Nb2fe1639010d4f81bfed1f845dd38e8f <http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity1> <http://dbkwik.webdatacommons.org/starwars.wikia.com/class/weapon> .\n"
                    ),
                    (
                        "_:Nb2fe1639010d4f81bfed1f845dd38e8f <http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity2> <http://dbkwik.webdatacommons.org/swg.wikia.com/class/weapon> .\n"
                    ),
                ]
            )
        else:
            return db.from_sequence(
                [
                    (
                        "<http://dbkwik.webdatacommons.org/starwars.wikia.com/resource/Charis_system> <http://purl.org/dc/terms/subject> <http://dbkwik.webdatacommons.org/starwars.wikia.com/resource/Category:Kathol_sector_star_systems> .\n"
                    ),
                    (
                        "<http://dbkwik.webdatacommons.org/starwars.wikia.com/resource/Catalyst:_A_Rogue_One_Novel> <http://dbkwik.webdatacommons.org/starwars.wikia.com/property/miscellanea> <http://dbkwik.webdatacommons.org/starwars.wikia.com/resource/Methane> .\n"
                    ),
                    (
                        "<http://dbkwik.webdatacommons.org/starwars.wikia.com/resource/ConJob%27s_Trifles> <http://dbkwik.webdatacommons.org/ontology/wikiPageWikiLink> <http://dbkwik.webdatacommons.org/starwars.wikia.com/resource/Hydroponic_garden> .\n"
                    ),
                ]
            )

    def mock_read_dask_df_archive_csv(self, inner_path: str, **kwargs) -> dd.DataFrame:
        return dd.from_pandas(
            self.mock_read_zipfile_csv(inner_path=inner_path), npartitions=1
        )

    def assert_not_called(self, **kwargs):
        assert False
