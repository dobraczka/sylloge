from collections import namedtuple

DatasetStatistics = namedtuple(
    "DatasetStatistics",
    [
        "num_rel_triples_left",
        "num_rel_triples_right",
        "num_attr_triples_left",
        "num_attr_triples_right",
        "num_ent_links",
    ],
)
