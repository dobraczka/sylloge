from typing import NamedTuple, Tuple


class EATaskStatistics(NamedTuple):
    num_rel_triples: Tuple[int, ...]
    num_attr_triples: Tuple[int, ...]
    num_ent_links: int
    num_intra_ent_links: Tuple[int, ...]
