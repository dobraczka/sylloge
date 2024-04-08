from typing import Dict

import dask.dataframe as dd
import pandas as pd
import pytest
from mocks import ResourceMocker
from util import EATaskStatistics

from sylloge import OpenEA
from sylloge.open_ea_loader import D_W, D_Y, EN_DE, EN_FR, SIZE_15K, SIZE_100K, V1, V2

statistics_with_params = [
    (
        {"graph_pair": D_W, "size": SIZE_15K, "version": V1},
        EATaskStatistics(
            num_rel_triples=(38265, 42746),
            num_attr_triples=(52134, 138246),
            num_ent_links=15000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": D_W, "size": SIZE_15K, "version": V2},
        EATaskStatistics(
            num_rel_triples=(73983, 83365),
            num_attr_triples=(51378, 175686),
            num_ent_links=15000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": D_Y, "size": SIZE_15K, "version": V1},
        EATaskStatistics(
            num_rel_triples=(30291, 26638),
            num_attr_triples=(52093, 117114),
            num_ent_links=15000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": D_Y, "size": SIZE_15K, "version": V2},
        EATaskStatistics(
            num_rel_triples=(68063, 60970),
            num_attr_triples=(49602, 116151),
            num_ent_links=15000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": EN_DE, "size": SIZE_15K, "version": V1},
        EATaskStatistics(
            num_rel_triples=(47676, 50419),
            num_attr_triples=(62403, 133776),
            num_ent_links=15000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": EN_DE, "size": SIZE_15K, "version": V2},
        EATaskStatistics(
            num_rel_triples=(84867, 92632),
            num_attr_triples=(59511, 161315),
            num_ent_links=15000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": EN_FR, "size": SIZE_15K, "version": V1},
        EATaskStatistics(
            num_rel_triples=(47334, 40864),
            num_attr_triples=(57164, 54401),
            num_ent_links=15000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": EN_FR, "size": SIZE_15K, "version": V2},
        EATaskStatistics(
            num_rel_triples=(96318, 80112),
            num_attr_triples=(52396, 56114),
            num_ent_links=15000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": D_W, "size": SIZE_100K, "version": V1},
        EATaskStatistics(
            num_rel_triples=(293990, 251708),
            num_attr_triples=(334911, 687860),
            num_ent_links=100000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": D_W, "size": SIZE_100K, "version": V2},
        EATaskStatistics(
            num_rel_triples=(616457, 588203),
            num_attr_triples=(360696, 878219),
            num_ent_links=100000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": D_Y, "size": SIZE_100K, "version": V1},
        EATaskStatistics(
            num_rel_triples=(294188, 400518),
            num_attr_triples=(360415, 649787),
            num_ent_links=100000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": D_Y, "size": SIZE_100K, "version": V2},
        EATaskStatistics(
            num_rel_triples=(576547, 865265),
            num_attr_triples=(374785, 755161),
            num_ent_links=100000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": EN_DE, "size": SIZE_100K, "version": V1},
        EATaskStatistics(
            num_rel_triples=(335359, 336240),
            num_attr_triples=(423666, 586207),
            num_ent_links=100000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": EN_DE, "size": SIZE_100K, "version": V2},
        EATaskStatistics(
            num_rel_triples=(622588, 629395),
            num_attr_triples=(430752, 656458),
            num_ent_links=100000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": EN_FR, "size": SIZE_100K, "version": V1},
        EATaskStatistics(
            num_rel_triples=(309607, 258285),
            num_attr_triples=(384248, 340725),
            num_ent_links=100000,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {"graph_pair": EN_FR, "size": SIZE_100K, "version": V2},
        EATaskStatistics(
            num_rel_triples=(649902, 561391),
            num_attr_triples=(396150, 342768),
            num_ent_links=100000,
            num_intra_ent_links=(0, 0),
        ),
    ),
]


@pytest.mark.slow()
@pytest.mark.parametrize(("params", "statistic"), statistics_with_params)
def test_open_ea(params: Dict, statistic: EATaskStatistics):
    ds = OpenEA(**params, use_cache=False)
    assert len(ds.rel_triples_left) == statistic.num_rel_triples[0]
    assert len(ds.rel_triples_right) == statistic.num_rel_triples[1]
    assert len(ds.attr_triples_left) == statistic.num_attr_triples[0]
    assert len(ds.attr_triples_right) == statistic.num_attr_triples[1]
    assert len(ds.ent_links) == statistic.num_ent_links
    assert ds.folds is not None
    assert len(ds.folds) == 5
    for fold in ds.folds:
        if params["size"] == SIZE_15K:
            assert len(fold.train) == 3000
            assert len(fold.test) == 10500
            assert len(fold.val) == 1500
        if params["size"] == SIZE_100K:
            assert len(fold.train) == 20000
            assert len(fold.test) == 70000
            assert len(fold.val) == 10000


@pytest.mark.parametrize(("params", "statistic"), statistics_with_params)
@pytest.mark.parametrize("backend", ["pandas", "dask"])
def test_open_ea_mock(
    params: Dict, statistic: EATaskStatistics, backend, mocker, tmp_path
):
    left_name, right_name = params["graph_pair"].split("_")
    fraction = 0.001 if params["size"] == SIZE_15K else 0.0001
    rm = ResourceMocker(statistic=statistic, fraction=fraction)
    mocker.patch("sylloge.base.read_zipfile_csv", rm.mock_read_zipfile_csv)
    mocker.patch(
        "sylloge.base.read_dask_df_archive_csv", rm.mock_read_dask_df_archive_csv
    )
    mocker.patch("sylloge.open_ea_loader.OPEN_EA_MODULE.ensure", rm.mock_ensure)
    # TODO mock zip ensure
    for use_cache, cache_exists in [(False, False), (True, False), (True, True)]:
        if cache_exists:
            # ensure these methods don't get called
            mocker.patch("sylloge.base.read_zipfile_csv", rm.assert_not_called)
            mocker.patch("sylloge.base.read_dask_df_archive_csv", rm.assert_not_called)

        ds = OpenEA(backend=backend, use_cache=use_cache, cache_path=tmp_path, **params)
        assert ds.__repr__() is not None
        assert ds.canonical_name
        assert ds.rel_triples_left is not None
        assert ds.rel_triples_right is not None
        assert ds.attr_triples_left is not None
        assert ds.attr_triples_right is not None
        assert ds.ent_links is not None
        assert ds.folds is not None
        assert len(ds.folds) == 5
        assert ds.dataset_names == OpenEA._GRAPH_PAIR_TO_DS_NAMES[params["graph_pair"]]
        for fold in ds.folds:
            assert fold.train is not None
            assert fold.test is not None
            assert fold.val is not None

        if backend == "pandas":
            assert isinstance(ds.rel_triples_left, pd.DataFrame)
        else:
            assert isinstance(ds.rel_triples_left, dd.DataFrame)

        assert ds.ds_prefix_tuples is not None
        assert ds._ds_prefixes is not None
