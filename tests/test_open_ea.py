from collections import namedtuple

import pytest

from ea_dataset_provider import OpenEA

DatasetStatistics = namedtuple(
    "DatasetStatistics",
    [
        "params",
        "num_rel_triples_left",
        "num_rel_triples_right",
        "num_attr_triples_left",
        "num_attr_triples_right",
    ],
)

all_ds_statistics = [
    DatasetStatistics(
        params=dict(graph_pair="D_W", size="15K", version="V1"),
        num_rel_triples_left=38265,
        num_rel_triples_right=42746,
        num_attr_triples_left=52134,
        num_attr_triples_right=138246,
    ),
    DatasetStatistics(
        params=dict(graph_pair="D_W", size="15K", version="V2"),
        num_rel_triples_left=73983,
        num_rel_triples_right=83365,
        num_attr_triples_left=51378,
        num_attr_triples_right=175686,
    ),
    DatasetStatistics(
        params=dict(graph_pair="D_Y", size="15K", version="V1"),
        num_rel_triples_left=30291,
        num_rel_triples_right=26638,
        num_attr_triples_left=52093,
        num_attr_triples_right=117114,
    ),
    DatasetStatistics(
        params=dict(graph_pair="D_Y", size="15K", version="V2"),
        num_rel_triples_left=68063,
        num_rel_triples_right=60970,
        num_attr_triples_left=49602,
        num_attr_triples_right=116151,
    ),
    DatasetStatistics(
        params=dict(graph_pair="EN_DE", size="15K", version="V1"),
        num_rel_triples_left=47676,
        num_rel_triples_right=50419,
        num_attr_triples_left=62403,
        num_attr_triples_right=133776,
    ),
    DatasetStatistics(
        params=dict(graph_pair="EN_DE", size="15K", version="V2"),
        num_rel_triples_left=84867,
        num_rel_triples_right=92632,
        num_attr_triples_left=59511,
        num_attr_triples_right=161315,
    ),
    DatasetStatistics(
        params=dict(graph_pair="EN_FR", size="15K", version="V1"),
        num_rel_triples_left=47334,
        num_rel_triples_right=40864,
        num_attr_triples_left=57164,
        num_attr_triples_right=54401,
    ),
    DatasetStatistics(
        params=dict(graph_pair="EN_FR", size="15K", version="V2"),
        num_rel_triples_left=96318,
        num_rel_triples_right=80112,
        num_attr_triples_left=52396,
        num_attr_triples_right=56114,
    ),
    DatasetStatistics(
        params=dict(graph_pair="D_W", size="100K", version="V1"),
        num_rel_triples_left=293990,
        num_rel_triples_right=251708,
        num_attr_triples_left=334911,
        num_attr_triples_right=687860,
    ),
    DatasetStatistics(
        params=dict(graph_pair="D_W", size="100K", version="V2"),
        num_rel_triples_left=616457,
        num_rel_triples_right=588203,
        num_attr_triples_left=360696,
        num_attr_triples_right=878219,
    ),
    DatasetStatistics(
        params=dict(graph_pair="D_Y", size="100K", version="V1"),
        num_rel_triples_left=294188,
        num_rel_triples_right=400518,
        num_attr_triples_left=360415,
        num_attr_triples_right=649787,
    ),
    DatasetStatistics(
        params=dict(graph_pair="D_Y", size="100K", version="V2"),
        num_rel_triples_left=576547,
        num_rel_triples_right=865265,
        num_attr_triples_left=374785,
        num_attr_triples_right=755161,
    ),
    DatasetStatistics(
        params=dict(graph_pair="EN_DE", size="100K", version="V1"),
        num_rel_triples_left=335359,
        num_rel_triples_right=336240,
        num_attr_triples_left=423666,
        num_attr_triples_right=586207,
    ),
    DatasetStatistics(
        params=dict(graph_pair="EN_DE", size="100K", version="V2"),
        num_rel_triples_left=622588,
        num_rel_triples_right=629395,
        num_attr_triples_left=430752,
        num_attr_triples_right=656458,
    ),
    DatasetStatistics(
        params=dict(graph_pair="EN_FR", size="100K", version="V1"),
        num_rel_triples_left=309607,
        num_rel_triples_right=258285,
        num_attr_triples_left=384248,
        num_attr_triples_right=340725,
    ),
    DatasetStatistics(
        params=dict(graph_pair="EN_FR", size="100K", version="V2"),
        num_rel_triples_left=649902,
        num_rel_triples_right=561391,
        num_attr_triples_left=396150,
        num_attr_triples_right=342768,
    ),
]


@pytest.mark.parametrize("ds_statistic", all_ds_statistics)
def test_open_ea(ds_statistic: DatasetStatistics):
    ds = OpenEA(**ds_statistic.params)
    assert len(ds.rel_triples_left) == ds_statistic.num_rel_triples_left
    assert len(ds.rel_triples_right) == ds_statistic.num_rel_triples_right
    assert len(ds.attr_triples_left) == ds_statistic.num_attr_triples_left
    assert len(ds.attr_triples_right) == ds_statistic.num_attr_triples_right
