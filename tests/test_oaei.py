import dask.dataframe as dd
import pytest
from mocks import ResourceMocker

from sylloge import OAEI


@pytest.mark.parametrize(
    "task",
    [
        "starwars-swg",
        "starwars-swtor",
        "marvelcinematicuniverse-marvel",
        "memoryalpha-memorybeta",
        "memoryalpha-stexpanded",
    ],
)
def test_oaei_mock(task, mocker, tmp_path):
    rm = ResourceMocker()
    mocker.patch(
        "sylloge.oaei_loader.read_dask_bag_from_archive_text",
        rm.mock_read_dask_bag_from_archive_text,
    )

    mocker.patch("sylloge.oaei_loader.OAEI_MODULE.ensure", rm.mock_ensure)
    mocker.patch(
        "sylloge.base.PrefixedClusterHelper.from_file",
        rm.mock_cluster_helper_from_zipped_file,
    )
    for use_cache, cache_exists in [(False, False), (True, False), (True, True)]:
        if cache_exists:
            # ensure this method doesn't get called
            mocker.patch(
                "sylloge.oaei_loader.read_dask_bag_from_archive_text",
                rm.assert_not_called,
            )
        ds = OAEI(task=task, use_cache=use_cache, cache_path=tmp_path)
        assert ds.__repr__() is not None
        assert ds.canonical_name
        assert ds.rel_triples_left is not None
        assert ds.rel_triples_right is not None
        assert ds.attr_triples_left is not None
        assert ds.attr_triples_right is not None
        assert ds.ent_links is not None
        assert ds.dataset_names == tuple(task.split("-"))
        assert isinstance(ds.rel_triples_left, dd.DataFrame)
        assert ds._ds_prefixes is not None
