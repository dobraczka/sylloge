from sylloge.utils import load_from_rdf
from sylloge.typing import COLUMNS, LABEL_TAIL

def test_load_from_rdf():
    rel, attr = load_from_rdf("http://www.w3.org/People/Berners-Lee/card")
    assert attr.shape == (34,3)
    assert rel.shape == (52,3)
    assert list(rel.columns) == COLUMNS
    assert list(attr.columns) == COLUMNS
    assert all(rel[LABEL_TAIL].apply(type) == str)
    assert any(attr[LABEL_TAIL].apply(type) == int)
    assert any(attr[LABEL_TAIL] == "Tim Berners-Lee")
