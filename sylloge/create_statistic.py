from typing import Dict, Iterable, Tuple

import pandas as pd

from sylloge import MED_BBK, OAEI, MovieGraphBenchmark, MultiSourceEADataset, OpenEA
from sylloge.base import create_statistics_df

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
    (MED_BBK, {}),
    (OAEI, {"task": "marvelcinematicuniverse-marvel"}),
    (OAEI, {"task": "memoryalpha-memorybeta"}),
    (OAEI, {"task": "memoryalpha-stexpanded"}),
    (OAEI, {"task": "starwars-swg"}),
    (OAEI, {"task": "starwars-swtor"}),
)


def create_statistic(
    classes_with_args: Iterable[
        Tuple[type[MultiSourceEADataset], Dict[str, str]]
    ] = all_classes_with_args,
    output_path: str = "dataset_statistics.csv",
    seperate_attribute_relations: bool = True,
) -> pd.DataFrame:
    datasets = iter(cls(**args) for cls, args in classes_with_args)  # type: ignore[arg-type]
    stats = create_statistics_df(datasets)
    stats.to_csv(output_path)
    print(f"Wrote stats to {output_path}")
    return stats


if __name__ == "__main__":
    create_statistic()
