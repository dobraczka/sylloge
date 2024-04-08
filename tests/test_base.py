import dask.dataframe as dd
import pandas as pd
import pytest
from eche import PrefixedClusterHelper

from sylloge import (
    MED_BBK,
    OAEI,
    MovieGraphBenchmark,
    OpenEA,
)


@pytest.mark.slow()
@pytest.mark.parametrize(
    ("cls", "args"),
    [
        (OpenEA, {}),
        (OpenEA, {"graph_pair": "D_W", "size": "15K", "version": "V1"}),
        (OpenEA, {"graph_pair": "D_W", "size": "15K", "version": "V2"}),
        (OpenEA, {"graph_pair": "D_Y", "size": "15K", "version": "V1"}),
        (OpenEA, {"graph_pair": "D_Y", "size": "15K", "version": "V2"}),
        (OpenEA, {"graph_pair": "EN_DE", "size": "15K", "version": "V1"}),
        (OpenEA, {"graph_pair": "EN_DE", "size": "15K", "version": "V2"}),
        (OpenEA, {"graph_pair": "EN_FR", "size": "15K", "version": "V1"}),
        (OpenEA, {"graph_pair": "EN_FR", "size": "15K", "version": "V2"}),
        (OpenEA, {"graph_pair": "D_W", "size": "100K", "version": "V1"}),
        (OpenEA, {"graph_pair": "D_W", "size": "100K", "version": "V2"}),
        (OpenEA, {"graph_pair": "D_Y", "size": "100K", "version": "V1"}),
        (OpenEA, {"graph_pair": "D_Y", "size": "100K", "version": "V2"}),
        (OpenEA, {"graph_pair": "EN_DE", "size": "100K", "version": "V1"}),
        (OpenEA, {"graph_pair": "EN_DE", "size": "100K", "version": "V2"}),
        (OpenEA, {"graph_pair": "EN_FR", "size": "100K", "version": "V1"}),
        (OpenEA, {"graph_pair": "EN_FR", "size": "100K", "version": "V2"}),
        (MovieGraphBenchmark, {}),
        (MovieGraphBenchmark, {"graph_pair": "imdb-tmdb"}),
        (MovieGraphBenchmark, {"graph_pair": "imdb-tvdb"}),
        (MovieGraphBenchmark, {"graph_pair": "tmdb-tvdb"}),
        (MovieGraphBenchmark, {"graph_pair": "multi"}),
        (MED_BBK, {}),
        (OAEI, {}),
        (OAEI, {"task": "marvelcinematicuniverse-marvel"}),
        (OAEI, {"task": "memoryalpha-memorybeta"}),
        (OAEI, {"task": "memoryalpha-stexpanded"}),
        (OAEI, {"task": "starwars-swg"}),
        (OAEI, {"task": "starwars-swtor"}),
    ],
)
@pytest.mark.parametrize("existing_cache", [True, False])
def test_all_load_from_existing(cls, args, existing_cache, tmp_path):
    """Smoketest loading."""
    ds = cls(**args) if existing_cache else cls(**args, cache_path=tmp_path)
    assert ds.__repr__() is not None
    assert ds.canonical_name
    assert ds.rel_triples is not None
    assert ds.rel_triples is not None
    assert ds.attr_triples is not None
    assert ds.attr_triples is not None
    assert ds.ent_links is not None
    assert ds.dataset_names is not None
    if cls == MovieGraphBenchmark:
        assert isinstance(ds.ent_links, PrefixedClusterHelper)
    elif cls == OAEI:
        assert isinstance(ds.ent_links, dd.DataFrame)
    else:
        assert isinstance(ds.ent_links, pd.DataFrame)
