from typing import Dict

import pytest
from mocks import ResourceMocker
from util import DatasetStatistics

from sylloge import MovieGraphBenchmark
from sylloge.moviegraph_benchmark_loader import IMDB_TMDB, IMDB_TVDB, TMDB_TVDB

statistics_with_params = [
    (
        dict(graph_pair=IMDB_TMDB),
        DatasetStatistics(
            num_rel_triples_left=17507,
            num_rel_triples_right=27903,
            num_attr_triples_left=20800,
            num_attr_triples_right=23761,
            num_ent_links=1978,
        ),
    ),
    (
        dict(graph_pair=IMDB_TVDB),
        DatasetStatistics(
            num_rel_triples_left=17507,
            num_rel_triples_right=15455,
            num_attr_triples_left=20800,
            num_attr_triples_right=20902,
            num_ent_links=2488,
        ),
    ),
    (
        dict(graph_pair=TMDB_TVDB),
        DatasetStatistics(
            num_rel_triples_left=27903,
            num_rel_triples_right=15455,
            num_attr_triples_left=23761,
            num_attr_triples_right=20902,
            num_ent_links=2483,
        ),
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("params,statistic", statistics_with_params)
def test_movie_benchmark(params: Dict, statistic: DatasetStatistics):
    ds = MovieGraphBenchmark(**params, use_cache=False)
    assert len(ds.rel_triples_left) == statistic.num_rel_triples_left
    assert len(ds.rel_triples_right) == statistic.num_rel_triples_right
    assert len(ds.attr_triples_left) == statistic.num_attr_triples_left
    assert len(ds.attr_triples_right) == statistic.num_attr_triples_right
    assert len(ds.ent_links) == statistic.num_ent_links
    assert ds.folds is not None
    if params["graph_pair"] == IMDB_TMDB:
        for fold in ds.folds:
            assert len(fold.train) == 396 or len(fold.train) == 395
            assert len(fold.test) == 1384 or len(fold.test) == 1385
            assert len(fold.val) == 198 or len(fold.val) == 197
    elif params["graph_pair"] == IMDB_TVDB:
        for fold in ds.folds:
            assert len(fold.train) == 498 or len(fold.train) == 497
            assert len(fold.test) == 1742 or len(fold.test) == 1741
            assert len(fold.val) == 249 or len(fold.val) == 248
    elif params["graph_pair"] == TMDB_TVDB:
        for fold in ds.folds:
            assert len(fold.train) == 496 or len(fold.train) == 497
            assert len(fold.test) == 1738
            assert len(fold.val) == 249 or len(fold.val) == 248


@pytest.mark.parametrize("params,statistic", statistics_with_params)
def test_movie_benchmark_mock(
    params: Dict, statistic: DatasetStatistics, mocker, tmpdir, tmp_path
):
    rm = ResourceMocker(statistic=statistic, fraction=0.1)
    mocker.patch("sylloge.moviegraph_benchmark_loader.load_data", rm.mock_load_data)
    for (use_cache, cache_exists) in [(False, False), (True, False), (True, True)]:
        if cache_exists:
            # ensure this method doesn't get called
            mocker.patch(
                "sylloge.moviegraph_benchmark_loader.load_data", rm.assert_not_called
            )
        ds = MovieGraphBenchmark(**params, use_cache=use_cache, cache_path=tmp_path)
        assert ds.__repr__() is not None
        assert ds.canonical_name
        assert ds.rel_triples_left is not None
        assert ds.rel_triples_right is not None
        assert ds.attr_triples_left is not None
        assert ds.attr_triples_right is not None
        assert ds.ent_links is not None
        assert ds.folds is not None
        if ds.graph_pair == IMDB_TMDB:
            assert ds.dataset_names == ("imdb", "tmdb")
        elif ds.graph_pair == IMDB_TVDB:
            assert ds.dataset_names == ("imdb", "tvdb")
        elif ds.graph_pair == TMDB_TVDB:
            assert ds.dataset_names == ("tmdb", "tvdb")
        for fold in ds.folds:
            assert fold.train is not None
            assert fold.test is not None
            assert fold.val is not None
