from typing import Dict, Iterable, Tuple

import pandas as pd
from eche import ClusterHelper, PrefixedClusterHelper

from sylloge import MED_BBK, OAEI, MovieGraphBenchmark, MultiSourceEADataset, OpenEA


def create_statistics_df(
    datasets: Iterable[MultiSourceEADataset], seperate_attribute_relations: bool = True
):
    rows = []
    triples_col = (
        ["Relation Triples", "Attribute Triples"]
        if seperate_attribute_relations
        else ["Triples"]
    )
    index_cols = ["Dataset family", "Task Name", "Dataset Name"]
    columns = [
        *index_cols,
        "Entities",
        *triples_col,
        "Relations",
        "Properties",
        "Literals",
        "Clusters",
        "Intra-dataset Matches",
        "All Matches",
    ]
    for ds in datasets:
        ds_family = str(ds.__class__.__name__).split(".")[-1]
        ds_stats, num_clusters = ds.statistics()
        intra_dataset_matches = (0,) * len(ds.dataset_names)
        if isinstance(ds.ent_links, ClusterHelper):
            all_matches = ds.ent_links.number_of_links
            if isinstance(ds.ent_links, PrefixedClusterHelper):
                intra_dataset_matches = ds.ent_links.number_of_intra_links
        else:
            all_matches = len(ds.ent_links)
        for i, (ds_side, ds_side_name) in enumerate(zip(ds_stats, ds.dataset_names)):
            if seperate_attribute_relations:
                triples = [ds_side.rel_triples, ds_side.attr_triples]
            else:
                triples = [ds_side.triples]
            rows.append(
                [
                    ds_family,
                    ds.canonical_name,
                    ds_side_name,
                    ds_side.entities,
                    *triples,
                    ds_side.relations,
                    ds_side.properties,
                    ds_side.literals,
                    num_clusters,
                    intra_dataset_matches[i],
                    all_matches,
                ]
            )
    statistics_df = pd.DataFrame(
        rows,
        columns=columns,
    )
    return statistics_df.set_index(index_cols)


all_classes_with_args: Tuple[Tuple[type[MultiSourceEADataset], Dict[str, str]], ...] = (
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
    (MovieGraphBenchmark, {"graph_pair": "imdb-tmdb"}),
    (MovieGraphBenchmark, {"graph_pair": "imdb-tvdb"}),
    (MovieGraphBenchmark, {"graph_pair": "tmdb-tvdb"}),
    (MovieGraphBenchmark, {"graph_pair": "multi"}),
    (MED_BBK, {}),
    (OAEI, {"task": "marvelcinematicuniverse-marvel"}),
    (OAEI, {"task": "memoryalpha-memorybeta"}),
    (OAEI, {"task": "memoryalpha-stexpanded"}),
    (OAEI, {"task": "starwars-swg"}),
    (OAEI, {"task": "starwars-swtor"}),
)


def create_and_write_statistic(
    classes_with_args: Iterable[
        Tuple[type[MultiSourceEADataset], Dict[str, str]]
    ] = all_classes_with_args,
    output_path: str = "dataset_statistics.csv",
    seperate_attribute_relations: bool = True,
) -> pd.DataFrame:
    datasets = iter(cls(**args) for cls, args in classes_with_args)  # type: ignore[arg-type]
    stats = create_statistics_df(
        datasets, seperate_attribute_relations=seperate_attribute_relations
    )
    stats.to_csv(output_path)
    print(f"Wrote stats to {output_path}")
    return stats


if __name__ == "__main__":
    create_and_write_statistic()
