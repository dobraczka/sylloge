from typing import NamedTuple, Optional, Tuple


class EATaskStatistics(NamedTuple):
    num_rel_triples: Tuple[int, ...]
    num_attr_triples: Tuple[int, ...]
    num_ent_links: int
    num_intra_ent_links: Tuple[int, ...]
    num_total_links: Optional[int] = None
