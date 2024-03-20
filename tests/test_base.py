import pathlib
from typing import Dict

import pytest
from mocks import ResourceMocker
from util import EATaskStatistics

from sylloge import ZipEADatasetWithPreSplitFolds

statistics_with_params = [
    (
        {
            "dataset_names": ("ds1", "ds2"),
            "ds_prefixes": ("left", "right"),
        },
        EATaskStatistics(
            num_rel_triples=(100, 200),
            num_attr_triples=(150, 200),
            num_ent_links=100,
            num_intra_ent_links=(0, 0),
        ),
    ),
    (
        {
            "dataset_names": ("ds1", "ds2", "ds3"),
            "ds_prefixes": ("left", "right", "middle"),
            "file_names_rel_triples": (
                "rel_triples_1",
                "rel_triples_2",
                "rel_triples_3",
            ),
            "file_names_attr_triples": (
                "attr_triples_1",
                "attr_triples_2",
                "attr_triples_3",
            ),
        },
        EATaskStatistics(
            num_rel_triples=(100, 200, 120),
            num_attr_triples=(150, 200, 130),
            num_ent_links=100,
            num_intra_ent_links=(0, 0, 0),
        ),
    ),
    (
        {
            "dataset_names": tuple(f"ds_{idx}" for idx in range(20)),
            "ds_prefixes": tuple(f"ds_{idx}" for idx in range(20)),
            "file_names_rel_triples": tuple(
                f"rel_triples_{idx}" for idx in range(1, 21)
            ),
            "file_names_attr_triples": tuple(
                f"rel_triples_{idx}" for idx in range(1, 21)
            ),
        },
        EATaskStatistics(
            num_rel_triples=tuple(100 + idx for idx in range(20)),
            num_attr_triples=tuple(100 + idx for idx in range(20)),
            num_ent_links=100,
            num_intra_ent_links=(0,) * 19,
        ),
    ),
]


@pytest.mark.parametrize(("params", "statistic"), statistics_with_params)
@pytest.mark.parametrize("backend", ["pandas", "dask"])
def test_zip_ea_dataset_with_pre_split_folds(
    params: Dict, statistic: EATaskStatistics, backend, mocker, tmp_path
):
    rm = ResourceMocker(statistic=statistic)
    mocker.patch(
        "sylloge.base.ClusterHelper.from_zipped_file",
        rm.mock_cluster_helper_from_zipped_file,
    )
    mocker.patch(
        "sylloge.base.PrefixedClusterHelper.from_zipped_file",
        rm.mock_cluster_helper_from_zipped_file,
    )
    if len(statistic.num_rel_triples) == 2:
        mocker.patch("sylloge.base.read_zipfile_csv", rm.mock_read_zipfile_csv)
    else:
        mocker.patch("sylloge.base.read_zipfile_csv", rm.mock_read_zipfile_csv_multi)

    if len(statistic.num_rel_triples) == 2:
        mocker.patch(
            "sylloge.base.read_dask_df_archive_csv", rm.mock_read_dask_df_archive_csv
        )
    else:
        mocker.patch(
            "sylloge.base.read_dask_df_archive_csv",
            rm.mock_read_dask_df_archive_csv_multi,
        )
    for use_cache, cache_exists in [(False, False), (True, False), (True, True)]:
        if cache_exists:
            # ensure these methods don't get called
            mocker.patch("sylloge.base.read_zipfile_csv", rm.assert_not_called)
            mocker.patch(
                "sylloge.base.ClusterHelper.from_zipped_file",
                rm.mock_cluster_helper_from_zipped_file,
            )
            mocker.patch(
                "sylloge.base.PrefixedClusterHelper.from_zipped_file",
                rm.mock_cluster_helper_from_zipped_file,
            )
            mocker.patch("sylloge.base.read_dask_df_archive_csv", rm.assert_not_called)
        ds = ZipEADatasetWithPreSplitFolds(
            backend=backend,
            use_cache=use_cache,
            cache_path=tmp_path,
            zip_path="mocked",
            inner_path=pathlib.PurePosixPath("mocked"),
            **params,
        )
        for idx, _ in enumerate(statistic.num_rel_triples):
            assert len(ds.rel_triples[idx]) == statistic.num_rel_triples[idx]
            assert len(ds.attr_triples[idx]) == statistic.num_attr_triples[idx]
        assert len(ds.ent_links) == statistic.num_ent_links
        assert ds.folds is not None
        assert len(ds.folds) == 5
