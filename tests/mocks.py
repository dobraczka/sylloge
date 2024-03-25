import pathlib
from collections import OrderedDict
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import dask.bag as db
import dask.dataframe as dd
import pandas as pd
import pytest
from eche import ClusterHelper, PrefixedClusterHelper
from moviegraphbenchmark.loading import ERData, Fold
from strawman import dummy_df, dummy_triples
from util import EATaskStatistics

from sylloge.moviegraph_benchmark_loader import GP_TO_DS_PREFIX, GraphPair
from sylloge.my_typing import EA_SIDE_LEFT, EA_SIDE_RIGHT, EA_SIDES
from sylloge.oaei_loader import TASK_NAME_TO_PREFIX


class ResourceMocker:
    def __init__(
        self,
        statistic: Optional[EATaskStatistics] = None,
        fraction: float = 1.0,
        seed: int = 17,
    ):
        if statistic is None:
            self.statistic = EATaskStatistics(
                num_rel_triples=(15, 15),
                num_attr_triples=(15, 15),
                num_ent_links=15,
                num_intra_ent_links=(0, 0),
            )
        else:
            self.statistic = statistic
        self.fraction = fraction
        self.seed = seed

    def mock_ensure(
        self,
        *subkeys: str,
        url: str,
        name: Optional[str] = None,
        force: bool = False,
        download_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> pathlib.Path:
        return pathlib.Path("mocked")

    def mock_read(self, path: str, names: Iterable):
        return self.mock_read_zipfile_csv(path="", inner_path=path)

    def mock_clusterhelper_from_file(self, path: str, ds_prefixes: OrderedDict):
        ent_links = self.mock_read_zipfile_csv("ent_links")
        ent_links = self._add_prefixes(ent_links, ds_prefixes)
        if len(ds_prefixes) == 2:
            return PrefixedClusterHelper.from_numpy(
                ent_links.to_numpy(), ds_prefixes=ds_prefixes
            )
        middle = dummy_df(
            (int(len(ent_links) / 3), 1), columns=["middle"], content_length=50
        )
        mpref = tuple(ds_prefixes.values())[2]
        middle["middle"] = mpref + middle["middle"]
        clusters = []
        for idx, (lp, rp) in enumerate(ent_links.itertuples(index=False, name=None)):
            if idx < len(middle):
                clusters.append({lp, rp, middle.iloc[idx].to_numpy()[0]})
            else:
                clusters.append({lp, rp})
        return PrefixedClusterHelper(clusters, ds_prefixes=ds_prefixes)

    def _add_prefixes(
        self, ent_links: pd.DataFrame, ds_prefixes: Union[OrderedDict, Tuple]
    ) -> pd.DataFrame:
        if isinstance(ds_prefixes, OrderedDict):
            left_pref, right_pref = tuple(ds_prefixes.values())[:2]
        else:
            left_pref, right_pref = ds_prefixes
        ent_links["left"] = left_pref + ent_links["left"]
        ent_links["right"] = right_pref + ent_links["right"]
        return ent_links

    def mock_load_data(self, pair: GraphPair, data_path: str) -> ERData:
        ent_links = self.mock_read_zipfile_csv("ent_links")
        ent_links = self._add_prefixes(ent_links, GP_TO_DS_PREFIX[pair])
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

    def mock_read_zipfile_csv_multi(self, inner_path: str, **kwargs) -> pd.DataFrame:
        inner_path = pathlib.Path(inner_path).name
        if inner_path.startswith("rel_triples"):
            idx = int(inner_path.replace("rel_triples_", "")) - 1
            statistic = int(self.statistic.num_rel_triples[idx] * self.fraction)
        elif inner_path.startswith("attr_triples"):
            idx = int(inner_path.replace("attr_triples_", "")) - 1
            statistic = int(self.statistic.num_attr_triples[idx] * self.fraction)
        else:
            raise ValueError(f"Unknown case {inner_path}!")
        return dummy_triples(
            statistic,
            entity_prefix=f"ds_{idx}",
            seed=self.seed,
        )

    def mock_read_zipfile_csv(self, inner_path: str, **kwargs) -> pd.DataFrame:
        if "rel_triples_1" in inner_path:
            return dummy_triples(
                int(self.statistic.num_rel_triples[0] * self.fraction),
                entity_prefix=EA_SIDE_LEFT,
                seed=self.seed,
            )
        if "rel_triples_2" in inner_path:
            return dummy_triples(
                int(self.statistic.num_rel_triples[1] * self.fraction),
                entity_prefix=EA_SIDE_RIGHT,
                seed=self.seed,
            )
        if "attr_triples_1" in inner_path:
            return dummy_triples(
                int(self.statistic.num_attr_triples[0] * self.fraction),
                entity_prefix=EA_SIDE_LEFT,
                relation_triples=False,
                seed=self.seed,
            )
        if "attr_triples_2" in inner_path:
            return dummy_triples(
                int(self.statistic.num_attr_triples[1] * self.fraction),
                entity_prefix=EA_SIDE_RIGHT,
                relation_triples=False,
                seed=self.seed,
            )
        if "ent_links" in inner_path:
            return dummy_df(
                (int(self.statistic.num_ent_links * self.fraction), 2),
                content_length=100,
                columns=list(EA_SIDES),
                seed=self.seed,
            )
        raise ValueError("Unknown case!")

    def mock_cluster_helper_from_zipped_file(self, *args, **kwargs):
        if "ds_prefixes" in kwargs:
            dataset_names = list(kwargs["ds_prefixes"])
            ds_prefixes = list(kwargs["ds_prefixes"].values())
            pairs = [
                {f"{pref}_{idx}" for pref in ds_prefixes}
                for idx in range(int(self.statistic.num_ent_links * self.fraction))
            ]
            return PrefixedClusterHelper(
                data=pairs, ds_prefixes=OrderedDict(zip(dataset_names, ds_prefixes))
            )
        return ClusterHelper.from_numpy(
            dummy_df(
                shape=(int(self.statistic.num_ent_links * self.fraction), 2),
                content_length=100,
                columns=list(EA_SIDES),
                seed=self.seed,
            ).to_numpy()
        )

    def mock_read_dask_bag_from_archive_text(
        self, archive_path: str, inner_path: str, protocol: str
    ) -> db.Bag:
        if "ref" in inner_path:
            left_pref, right_pref = TASK_NAME_TO_PREFIX[inner_path.split("/")[0]]  # type: ignore[index]
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
                        f"_:Na53a488f3e0648d6a181817a992fb395 <http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity2> <{left_pref}TransGalMeg_%22Ixiyen%22_Fast_Attack_Craft> .\n"
                    ),
                    (
                        f"_:Na53a488f3e0648d6a181817a992fb395 <http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity1> <{right_pref}Ixiyen-class_fast_attack_craft> .\n"
                    ),
                    (
                        "_:Nb2fe1639010d4f81bfed1f845dd38e8f <http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity1> <http://dbkwik.webdatacommons.org/starwars.wikia.com/class/weapon> .\n"
                    ),
                    (
                        "_:Nb2fe1639010d4f81bfed1f845dd38e8f <http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity2> <http://dbkwik.webdatacommons.org/swg.wikia.com/class/weapon> .\n"
                    ),
                ]
            )
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

    def mock_read_dask_df_archive_csv_multi(
        self, inner_path: str, **kwargs
    ) -> dd.DataFrame:
        return dd.from_pandas(
            self.mock_read_zipfile_csv_multi(inner_path=inner_path), npartitions=1
        )

    def assert_not_called(self, **kwargs):
        pytest.fail("Assert should have called!")
