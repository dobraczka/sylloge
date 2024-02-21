from typing import NamedTuple


class DatasetStatistics(NamedTuple):
    num_rel_triples_left: int
    num_rel_triples_right: int
    num_attr_triples_left: int
    num_attr_triples_right: int
    num_ent_links: int
