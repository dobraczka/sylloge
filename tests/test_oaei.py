import dask.dataframe as dd
import pandas as pd
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
@pytest.mark.parametrize("backend", ["pandas", "dask"])
def test_oaei_mock(task, backend, mocker):
    rm = ResourceMocker()
    mocker.patch(
        "sylloge.oaei_loader.read_dask_bag_from_archive_text",
        rm.mock_read_dask_bag_from_archive_text,
    )
    ds = OAEI(backend=backend, task=task)
    assert ds.__repr__() is not None
    assert ds.canonical_name
    assert ds.rel_triples_left is not None
    assert ds.rel_triples_right is not None
    assert ds.attr_triples_left is not None
    assert ds.attr_triples_right is not None
    assert ds.ent_links is not None

    if backend == "pandas":
        assert isinstance(ds.rel_triples_left, pd.DataFrame)
    else:
        assert isinstance(ds.rel_triples_left, dd.DataFrame)
