import pytest

from sylloge import MED_BBK


@pytest.mark.slow()
def test_med_bbk():
    ds = MED_BBK(use_cache=False)
    assert len(ds.rel_triples_left) == 158357
    assert len(ds.rel_triples_right) == 50307
    assert len(ds.attr_triples_left) == 11467
    assert len(ds.attr_triples_right) == 44987
    assert ds.dataset_names == ("MED", "BBK")

    # check if switch was correct
    assert "集聚肠杆菌感染" in set(ds.rel_triples_right["head"])
