from typing import Dict

import pytest
from util import DatasetStatistics

from sylloge import OpenEA
from sylloge.open_ea_loader import D_W, D_Y, EN_DE, EN_FR, SIZE_15K, SIZE_100K, V1, V2

statistics_with_params = [
    (
        dict(graph_pair=D_W, size=SIZE_15K, version=V1),
        DatasetStatistics(
            num_rel_triples_left=38265,
            num_rel_triples_right=42746,
            num_attr_triples_left=52134,
            num_attr_triples_right=138246,
            num_ent_links=15000,
        ),
    ),
    (
        dict(graph_pair=D_W, size=SIZE_15K, version=V2),
        DatasetStatistics(
            num_rel_triples_left=73983,
            num_rel_triples_right=83365,
            num_attr_triples_left=51378,
            num_attr_triples_right=175686,
            num_ent_links=15000,
        ),
    ),
    (
        dict(graph_pair=D_Y, size=SIZE_15K, version=V1),
        DatasetStatistics(
            num_rel_triples_left=30291,
            num_rel_triples_right=26638,
            num_attr_triples_left=52093,
            num_attr_triples_right=117114,
            num_ent_links=15000,
        ),
    ),
    (
        dict(graph_pair=D_Y, size=SIZE_15K, version=V2),
        DatasetStatistics(
            num_rel_triples_left=68063,
            num_rel_triples_right=60970,
            num_attr_triples_left=49602,
            num_attr_triples_right=116151,
            num_ent_links=15000,
        ),
    ),
    (
        dict(graph_pair=EN_DE, size=SIZE_15K, version=V1),
        DatasetStatistics(
            num_rel_triples_left=47676,
            num_rel_triples_right=50419,
            num_attr_triples_left=62403,
            num_attr_triples_right=133776,
            num_ent_links=15000,
        ),
    ),
    (
        dict(graph_pair=EN_DE, size=SIZE_15K, version=V2),
        DatasetStatistics(
            num_rel_triples_left=84867,
            num_rel_triples_right=92632,
            num_attr_triples_left=59511,
            num_attr_triples_right=161315,
            num_ent_links=15000,
        ),
    ),
    (
        dict(graph_pair=EN_FR, size=SIZE_15K, version=V1),
        DatasetStatistics(
            num_rel_triples_left=47334,
            num_rel_triples_right=40864,
            num_attr_triples_left=57164,
            num_attr_triples_right=54401,
            num_ent_links=15000,
        ),
    ),
    (
        dict(graph_pair=EN_FR, size=SIZE_15K, version=V2),
        DatasetStatistics(
            num_rel_triples_left=96318,
            num_rel_triples_right=80112,
            num_attr_triples_left=52396,
            num_attr_triples_right=56114,
            num_ent_links=15000,
        ),
    ),
    (
        dict(graph_pair=D_W, size=SIZE_100K, version=V1),
        DatasetStatistics(
            num_rel_triples_left=293990,
            num_rel_triples_right=251708,
            num_attr_triples_left=334911,
            num_attr_triples_right=687860,
            num_ent_links=100000,
        ),
    ),
    (
        dict(graph_pair=D_W, size=SIZE_100K, version=V2),
        DatasetStatistics(
            num_rel_triples_left=616457,
            num_rel_triples_right=588203,
            num_attr_triples_left=360696,
            num_attr_triples_right=878219,
            num_ent_links=100000,
        ),
    ),
    (
        dict(graph_pair=D_Y, size=SIZE_100K, version=V1),
        DatasetStatistics(
            num_rel_triples_left=294188,
            num_rel_triples_right=400518,
            num_attr_triples_left=360415,
            num_attr_triples_right=649787,
            num_ent_links=100000,
        ),
    ),
    (
        dict(graph_pair=D_Y, size=SIZE_100K, version=V2),
        DatasetStatistics(
            num_rel_triples_left=576547,
            num_rel_triples_right=865265,
            num_attr_triples_left=374785,
            num_attr_triples_right=755161,
            num_ent_links=100000,
        ),
    ),
    (
        dict(graph_pair=EN_DE, size=SIZE_100K, version=V1),
        DatasetStatistics(
            num_rel_triples_left=335359,
            num_rel_triples_right=336240,
            num_attr_triples_left=423666,
            num_attr_triples_right=586207,
            num_ent_links=100000,
        ),
    ),
    (
        dict(graph_pair=EN_DE, size=SIZE_100K, version=V2),
        DatasetStatistics(
            num_rel_triples_left=622588,
            num_rel_triples_right=629395,
            num_attr_triples_left=430752,
            num_attr_triples_right=656458,
            num_ent_links=100000,
        ),
    ),
    (
        dict(graph_pair=EN_FR, size=SIZE_100K, version=V1),
        DatasetStatistics(
            num_rel_triples_left=309607,
            num_rel_triples_right=258285,
            num_attr_triples_left=384248,
            num_attr_triples_right=340725,
            num_ent_links=100000,
        ),
    ),
    (
        dict(graph_pair=EN_FR, size=SIZE_100K, version=V2),
        DatasetStatistics(
            num_rel_triples_left=649902,
            num_rel_triples_right=561391,
            num_attr_triples_left=396150,
            num_attr_triples_right=342768,
            num_ent_links=100000,
        ),
    ),
]


@pytest.mark.parametrize("params,statistic", statistics_with_params)
def test_open_ea(params: Dict, statistic: DatasetStatistics):
    ds = OpenEA(**params)
    assert len(ds.rel_triples_left) == statistic.num_rel_triples_left
    assert len(ds.rel_triples_right) == statistic.num_rel_triples_right
    assert len(ds.attr_triples_left) == statistic.num_attr_triples_left
    assert len(ds.attr_triples_right) == statistic.num_attr_triples_right
    assert len(ds.ent_links) == statistic.num_ent_links
    assert ds.folds is not None
    for fold in ds.folds:
        if params["size"] == SIZE_15K:
            assert len(fold.train) == 3000
            assert len(fold.test) == 10500
            assert len(fold.val) == 1500
        if params["size"] == SIZE_100K:
            assert len(fold.train) == 20000
            assert len(fold.test) == 70000
            assert len(fold.val) == 10000
