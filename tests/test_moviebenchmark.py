from typing import Dict, Iterable, Optional, Tuple, cast

import pytest
from eche import PrefixedClusterHelper
from mocks import ResourceMocker
from util import EATaskStatistics

from sylloge import MovieGraphBenchmark, TrainTestValSplit
from sylloge.moviegraph_benchmark_loader import IMDB_TMDB, IMDB_TVDB, MULTI, TMDB_TVDB

statistics_with_params = [
    (
        {"graph_pair": IMDB_TMDB},
        EATaskStatistics(
            num_rel_triples=(17507, 27903),
            num_attr_triples=(20800, 23761),
            num_ent_links=2172,
            num_intra_ent_links=(1, 64),
        ),
    ),
    (
        {"graph_pair": IMDB_TVDB},
        EATaskStatistics(
            num_rel_triples=(17507, 15455),
            num_attr_triples=(20800, 20902),
            num_ent_links=2919,
            num_intra_ent_links=(1, 22663),
        ),
    ),
    (
        {"graph_pair": TMDB_TVDB},
        EATaskStatistics(
            num_rel_triples=(27903, 15455),
            num_attr_triples=(23761, 20902),
            num_ent_links=3411,
            num_intra_ent_links=(64, 22663),
        ),
    ),
    (
        {"graph_pair": MULTI},
        EATaskStatistics(
            num_rel_triples=(17507, 27903, 15455),
            num_attr_triples=(20800, 23761, 20902),
            num_ent_links=3598,
            num_intra_ent_links=(1, 64, 22663),
        ),
    ),
]


def _test_fold_number(
    folds: Iterable[TrainTestValSplit],
    train_test_val: Tuple[int, int, int],
    approx_bounds: Optional[Tuple[int, int, int]] = None,
):
    train_num, test_num, val_num = train_test_val
    train_bound, test_bound, val_bound = (
        (1, 1, 1) if approx_bounds is None else approx_bounds
    )
    train_range = list(range(train_num, train_num + train_bound))
    test_range = list(range(test_num, test_num + test_bound))
    val_range = list(range(val_num, val_num + val_bound))
    for fold in folds:
        train = cast(PrefixedClusterHelper, fold.train)
        test = cast(PrefixedClusterHelper, fold.test)
        val = cast(PrefixedClusterHelper, fold.val)
        assert train.number_of_no_intra_links in train_range
        assert test.number_of_no_intra_links in test_range
        assert val.number_of_no_intra_links in val_range


@pytest.mark.slow()
@pytest.mark.parametrize(("params", "statistic"), statistics_with_params)
def test_movie_benchmark(params: Dict, statistic: EATaskStatistics):
    ds = MovieGraphBenchmark(**params, use_cache=False)
    for idx in range(len(statistic.num_rel_triples)):
        assert len(ds.rel_triples[idx]) == statistic.num_rel_triples[idx]
        assert len(ds.attr_triples[idx]) == statistic.num_attr_triples[idx]
        total_links = statistic.num_ent_links + sum(statistic.num_intra_ent_links)
        assert ds.ent_links.number_of_links == total_links
        assert ds.ent_links.number_of_no_intra_links == statistic.num_ent_links
    assert ds.folds is not None
    if params["graph_pair"] == IMDB_TMDB:
        _test_fold_number(ds.folds, (434, 1520, 218))
    elif params["graph_pair"] == IMDB_TVDB:
        _test_fold_number(ds.folds, (584, 2043, 292))
    elif params["graph_pair"] == TMDB_TVDB:
        _test_fold_number(ds.folds, (682, 2388, 341), (20, 70, 10))


@pytest.mark.parametrize(("params", "statistic"), statistics_with_params)
def test_movie_benchmark_mock(
    params: Dict, statistic: EATaskStatistics, mocker, tmp_path
):
    rm = ResourceMocker(statistic=statistic, fraction=0.1)
    mocker.patch("sylloge.moviegraph_benchmark_loader.load_data", rm.mock_load_data)
    mocker.patch(
        "sylloge.moviegraph_benchmark_loader.PrefixedClusterHelper.from_file",
        rm.mock_clusterhelper_from_file,
    )
    for use_cache, cache_exists in [(False, False), (True, False), (True, True)]:
        if cache_exists:
            # ensure this method doesn't get called
            mocker.patch(
                "sylloge.moviegraph_benchmark_loader.load_data", rm.assert_not_called
            )
        ds = MovieGraphBenchmark(**params, use_cache=use_cache, cache_path=tmp_path)
        assert ds.__repr__() is not None
        assert ds.canonical_name
        assert ds.rel_triples is not None
        assert ds.rel_triples is not None
        assert ds.attr_triples is not None
        assert ds.attr_triples is not None
        assert ds.ent_links is not None
        assert ds.folds is not None
        if ds.graph_pair == IMDB_TMDB:
            assert ds.dataset_names == ("imdb", "tmdb")
        elif ds.graph_pair == IMDB_TVDB:
            assert ds.dataset_names == ("imdb", "tvdb")
        elif ds.graph_pair == TMDB_TVDB:
            assert ds.dataset_names == ("tmdb", "tvdb")
        elif ds.graph_pair == MULTI:
            assert ds.dataset_names == ("imdb", "tmdb", "tvdb")
        for fold in ds.folds:
            assert fold.train is not None
            assert fold.test is not None
            assert fold.val is not None
