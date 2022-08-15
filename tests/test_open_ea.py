from ea_dataset_provider import OpenEA


def test_open_ea():
    ds = OpenEA()
    assert len(ds.rel_triples_left) == 38265
    assert len(ds.rel_triples_right) == 42746
    assert len(ds.attr_triples_left) == 52134
    assert len(ds.attr_triples_right) == 138246
