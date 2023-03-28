from pathlib import Path
from typing import Iterable, Union

import pandas as pd
from strawman import dummy_df, dummy_triples
from util import DatasetStatistics

from sylloge.base import EA_SIDE_LEFT, EA_SIDE_RIGHT, EA_SIDES


class DataPathMocker:
    def __init__(self, data_path):
        self.data_path = data_path

    def mock_data_path(self):
        return self.data_path


class ResourceMocker:
    def __init__(self, statistic: DatasetStatistics, fraction: float):
        self.statistic = statistic
        self.fraction = fraction

    def mock_read(self, path: str, names: Iterable):
        return self.mock_read_zipfile_csv(path="", inner_path=path)

    def mock_read_zipfile_csv(
        self, path: Union[str, Path], inner_path: str, sep: str = "\t", **kwargs
    ) -> pd.DataFrame:
        if "rel_triples_1" in inner_path:
            return dummy_triples(
                int(self.statistic.num_rel_triples_left * self.fraction),
                entity_prefix=EA_SIDE_LEFT,
            )
        elif "rel_triples_2" in inner_path:
            return dummy_triples(
                int(self.statistic.num_rel_triples_right * self.fraction),
                entity_prefix=EA_SIDE_RIGHT,
            )
        elif "attr_triples_1" in inner_path:
            return dummy_triples(
                int(self.statistic.num_attr_triples_left * self.fraction),
                entity_prefix=EA_SIDE_LEFT,
                relation_triples=False,
            )
        elif "attr_triples_2" in inner_path:
            return dummy_triples(
                int(self.statistic.num_attr_triples_right * self.fraction),
                entity_prefix=EA_SIDE_RIGHT,
                relation_triples=False,
            )
        elif "_links" in inner_path:
            return dummy_df(
                shape=(int(self.statistic.num_ent_links * self.fraction), 2),
                content_length=100,
                columns=list(EA_SIDES),
            )
