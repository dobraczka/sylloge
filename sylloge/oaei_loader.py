"""OAEI knowledge graph track dataset class."""
import logging
import pathlib
import typing
from collections import namedtuple
from typing import Any, Dict, Literal, Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd

from .base import BASE_DATASET_MODULE, CacheableEADataset, DataFrameType
from .dask import read_dask_bag_from_archive_text
from .typing import (
    BACKEND_LITERAL,
    COLUMNS,
    EA_SIDE_LEFT,
    EA_SIDE_RIGHT,
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
)

OAEI_MODULE = BASE_DATASET_MODULE.module("oaei")

OAEI_TASK_NAME = Literal[
    "starwars-swg",
    "starwars-swtor",
    "marvelcinematicuniverse-marvel",
    "memoryalpha-memorybeta",
    "memoryalpha-stexpanded",
]

logger = logging.getLogger(__name__)

REFLINE_TYPE_LITERAL = Literal["dismiss", "entity", "property", "class", "unknown"]
REL_ATTR_LITERAL = Literal["rel", "attr"]

reflinetype = pd.CategoricalDtype(categories=typing.get_args(REFLINE_TYPE_LITERAL))
relattrlinetype = pd.CategoricalDtype(categories=typing.get_args(REL_ATTR_LITERAL))

URL_SHA512_HASH = namedtuple("URL_SHA512_HASH", ["url", "hash"])


def parse_ref_line(line: str) -> Tuple[str, str, str, REFLINE_TYPE_LITERAL]:
    subj, pred, obj = line.split(" ", maxsplit=2)
    pred = pred[1:-1]
    obj = obj[1:-4]
    disallowed_pred = [
        "http://knowledgeweb.semanticweb.org/heterogeneity/alignmentmap",
        "http://knowledgeweb.semanticweb.org/heterogeneity/alignmentrelation",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://knowledgeweb.semanticweb.org/heterogeneity/alignmentmeasure",
        "http://knowledgeweb.semanticweb.org/heterogeneity/alignmenttype",
        "http://knowledgeweb.semanticweb.org/heterogeneity/alignmentlevel",
        "http://knowledgeweb.semanticweb.org/heterogeneity/alignmentxml",
    ]
    line_type: REFLINE_TYPE_LITERAL = "unknown"
    if pred in disallowed_pred:
        line_type = "dismiss"
    if "/resource/" in obj:
        line_type = "entity"
    elif "/property/" in obj:
        line_type = "property"
    elif "/class/" in obj:
        line_type = "class"
    return subj, pred, obj, line_type


def clean_and_categorize_part(
    part: str, remove_dt: bool = True
) -> Tuple[str, REL_ATTR_LITERAL]:
    if part.endswith(" .\n"):
        part = part[:-3]
    if part.startswith("<") and part.endswith(">"):
        return part[1:-1], "rel"
    if remove_dt:
        part = part.split("^^")[0]
        if part.startswith('"') and part.endswith('"'):
            return part[1:-1], "attr"
    return part, "attr"


def fault_tolerant_parse_nt(
    line: str, remove_dt: bool = True
) -> Tuple[str, str, str, REL_ATTR_LITERAL]:
    subj, pred, obj = line.split(" ", maxsplit=2)
    subj, _ = clean_and_categorize_part(subj, remove_dt)
    pred, _ = clean_and_categorize_part(pred, remove_dt)
    obj, triple_type = clean_and_categorize_part(obj, remove_dt)
    return subj, pred, obj, triple_type


class OAEI(CacheableEADataset[DataFrameType]):
    """The  OAEI (Ontology Alignment Evaluation Initiative) Knowledge Graph Track tasks contain graphs created from fandom wikis.

    Five integration tasks are available:
        - starwars-swg
        - starwars-swtor
        - marvelcinematicuniverse-marvel
        - memoryalpha-memorybeta
        - memoryalpha-stexpanded

    More information can be found at the `website <http://oaei.ontologymatching.org/2019/knowledgegraph/index.html>`_.
    """

    class_links: pd.DataFrame
    property_links: pd.DataFrame

    _CLASS_LINKS_PATH: str = "class_links_parquet"
    _PROPERTY_LINKS_PATH: str = "property_links_parquet"

    _TASK_URLS: Dict[OAEI_TASK_NAME, URL_SHA512_HASH] = {
        "starwars-swg": URL_SHA512_HASH(
            "https://cloud.scadsai.uni-leipzig.de/index.php/s/WzRxffgoq2qqn7q/download/starwars-swg.tar.gz",
            "ac6c1ca1b374e24fbeaf368a4ecf5254c54be38d0cbc1880815310ffa534c2266fa0da840014facf5fff430f777ef342a2d432fe00cf6fc14a0a12419d6966be",
        ),
        "starwars-swtor": URL_SHA512_HASH(
            "https://cloud.scadsai.uni-leipzig.de/index.php/s/CcG6tANX4JqacNY/download/starwars-swtor.tar.gz",
            "6b7913c5613bbc8c2aa849936d8b86c4613b73cebd06f8ee1c5e26f3605f95e61115a8a4d2d1399dbeb047579440899e9ecfce31f8c95c6013260dd3785cac80",
        ),
        "marvelcinematicuniverse-marvel": URL_SHA512_HASH(
            "https://cloud.scadsai.uni-leipzig.de/index.php/s/CH3NNin95BGjgb3/download/marvelcinematicuniverse-marvel.tar.gz",
            "d67936030c5ada48e6415774210527af51bf964828edd3f18beed51146ea9f70425cb760d073803189c4079dc68a4e39168fb1105c42fa6e82b790eb01e4a207",
        ),
        "memoryalpha-memorybeta": URL_SHA512_HASH(
            "https://cloud.scadsai.uni-leipzig.de/index.php/s/d6TYKFfDKxZEpiy/download/memoryalpha-memorybeta.tar.gz",
            "afb0a06225afcc2ae6edf5dcd513e48d4159e3d9bf94fc3cda6495a8c98a26fbc369413c587daec09e6ff7d045bf061a16ab8b65af4e22d0c91113390dc5d8d3",
        ),
        "memoryalpha-stexpanded": URL_SHA512_HASH(
            "https://cloud.scadsai.uni-leipzig.de/index.php/s/wSaRamAT5R5ieFZ/download/memoryalpha-stexpanded.tar.gz",
            "3d6f93c46425b4cb06f5a532601af9ee0a9fcdc3928709fa0ac66617099a27e26c244c024ae097274173cf2c910b045ea412fb225f51323eb7cdad86ad2461cc",
        ),
    }

    # avoid triggering dask compute
    _precalc_ds_statistics = {
        "starwars-swg": dict(
            rel_triples_left=6675247,
            rel_triples_right=178085,
            attr_triples_left=1570786,
            attr_triples_right=76269,
            ent_links=1096,
            folds=None,
            property_links=20,
            class_links=5,
        ),
        "starwars-swtor": dict(
            rel_triples_left=6675247,
            rel_triples_right=105543,
            attr_triples_left=1570786,
            attr_triples_right=40605,
            ent_links=1358,
            folds=None,
            property_links=56,
            class_links=15,
        ),
        "marvelcinematicuniverse-marvel": dict(
            rel_triples_left=1094598,
            rel_triples_right=5152898,
            attr_triples_left=130517,
            attr_triples_right=1580468,
            ent_links=1654,
            folds=None,
            property_links=11,
            class_links=2,
        ),
        "memoryalpha-memorybeta": dict(
            rel_triples_left=2096198,
            rel_triples_right=2048728,
            attr_triples_left=430730,
            attr_triples_right=494181,
            ent_links=9296,
            folds=None,
            property_links=55,
            class_links=14,
        ),
        "memoryalpha-stexpanded": dict(
            rel_triples_left=2096198,
            rel_triples_right=412179,
            attr_triples_left=430730,
            attr_triples_right=155207,
            ent_links=1725,
            folds=None,
            property_links=41,
            class_links=13,
        ),
    }

    def __init__(
        self,
        task: OAEI_TASK_NAME = "starwars-swg",
        backend: BACKEND_LITERAL = "dask",
        npartitions: int = 1,
        use_cache: bool = True,
        cache_path: Optional[pathlib.Path] = None,
    ):
        """Initialize a OAEI Knowledge Graph Track task.

        :param task: Name of the task. Has to be one of {starwars-swg,starwars-swtor,marvelcinematicuniverse-marvel,memoryalpha-memorybeta, memoryalpha-stexpanded}
        :param backend: Whether to use "pandas" or "dask"
        :param npartitions: how many partitions to use for each frame, when using dask
        :param use_cache: whether to use cache or not
        :param cache_path: Path where cache will be stored/loaded
        :raises ValueError: if unknown task value is provided
        """
        if task not in typing.get_args(OAEI_TASK_NAME):
            raise ValueError(f"Task has to be one of {OAEI_TASK_NAME}, but got{task}")
        self.task = task

        left_name, right_name = task.split("-")
        inner_cache_path = f"{left_name}_{right_name}"
        actual_cache_path = self.create_cache_path(
            OAEI_MODULE, inner_cache_path, cache_path
        )

        super().__init__(
            use_cache=use_cache,
            cache_path=actual_cache_path,
            dataset_names=(left_name, right_name),
            backend=backend,
            npartitions=npartitions,
        )

    def initial_read(self, backend: BACKEND_LITERAL):
        archive_url, sha512hash = OAEI._TASK_URLS[self.task]

        # ensure archive file is present
        archive_path = str(
            OAEI_MODULE.ensure(
                url=archive_url,
                download_kwargs=dict(hexdigests=dict(sha512=sha512hash)),
            )
        )
        entity_mapping_df, property_mapping_df, class_mapping_df = self._mapping_dfs(
            archive_path
        )
        left_attr, left_rel = self._read_triples(archive_path, left=True)
        right_attr, right_rel = self._read_triples(archive_path, left=False)

        # need to set before initialization
        # because of possible transforming
        self.property_links = property_mapping_df
        self.class_links = class_mapping_df
        return dict(
            rel_triples_left=left_rel,
            rel_triples_right=right_rel,
            attr_triples_left=left_attr,
            attr_triples_right=right_attr,
            ent_links=entity_mapping_df,
        )

    @property
    def _canonical_name(self) -> str:
        return f"{self.__class__.__name__}_{self.task}"

    @property
    def _param_repr(self) -> str:
        return f"task={self.task}, "

    @property
    def _statistics(self) -> str:
        this_ds_stats = self.__class__._precalc_ds_statistics[self.task]
        stat_str = ""
        for key, value in this_ds_stats.items():
            stat_str += f"{key}={value}, "
        return stat_str[:-2]  # remove last comma and space

    def _pair_dfs(self, mapping_df: dd.DataFrame) -> dd.DataFrame:
        left_side_uri = (
            "http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity1"
        )
        side_predicate = mapping_df[LABEL_RELATION] == left_side_uri
        map_left = mapping_df[side_predicate]
        map_right = mapping_df[~side_predicate]
        joined_left = LABEL_TAIL + EA_SIDE_LEFT
        joined_right = LABEL_TAIL + EA_SIDE_RIGHT
        return (
            map_left.set_index(LABEL_HEAD)
            .join(
                map_right.set_index(LABEL_HEAD),
                lsuffix=EA_SIDE_LEFT,
                rsuffix=EA_SIDE_RIGHT,
            )[[joined_left, joined_right]]
            .rename(columns={joined_left: EA_SIDE_LEFT, joined_right: EA_SIDE_RIGHT})
        )

    def _mapping_dfs(
        self, archive_path: str
    ) -> Tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        LINE_TYPE_COL = "line_type"
        schema_df = pd.DataFrame(
            {
                LABEL_HEAD: pd.Series([], dtype="str"),
                LABEL_RELATION: pd.Series([], dtype="str"),
                LABEL_TAIL: pd.Series([], dtype="str"),
                LINE_TYPE_COL: pd.Series(
                    [],
                    dtype=reflinetype,
                ),
            }
        )
        df = (
            read_dask_bag_from_archive_text(
                archive_path,
                inner_path=f"{self.task}/ref-{self.task}.nt",
                protocol="tar",
            )
            .map(parse_ref_line)
            .filter(lambda x: x[3] != "dismiss")
            .to_dataframe(schema_df)
        )

        entity_mapping_df = df[df[LINE_TYPE_COL] == "entity"]
        property_mapping_df = df[df[LINE_TYPE_COL] == "property"]
        class_mapping_df = df[df[LINE_TYPE_COL] == "class"]
        return (
            self._pair_dfs(entity_mapping_df),
            self._pair_dfs(property_mapping_df),
            self._pair_dfs(class_mapping_df),
        )

    def _read_triples(
        self, archive_path: str, left: bool
    ) -> Tuple[dd.DataFrame, dd.DataFrame]:
        task_index = 0 if left else 1
        task_side = self.task.split("-")[task_index]
        REL_ATTR_COL = "rel_attr_type"
        schema_df = pd.DataFrame(
            {
                LABEL_HEAD: pd.Series([], dtype="str"),
                LABEL_RELATION: pd.Series([], dtype="str"),
                LABEL_TAIL: pd.Series([], dtype="str"),
                REL_ATTR_COL: pd.Series(
                    [],
                    dtype=relattrlinetype,
                ),
            }
        )
        df = (
            read_dask_bag_from_archive_text(
                archive_path, inner_path=f"{self.task}/{task_side}.nt", protocol="tar"
            )
            .map(fault_tolerant_parse_nt)
            .to_dataframe(schema_df)
        )
        return (
            df[df[REL_ATTR_COL] == "attr"][COLUMNS],
            df[df[REL_ATTR_COL] == "rel"][COLUMNS],
        )

    def to_parquet(self, path: Union[str, pathlib.Path], **kwargs):
        """Write dataset to path as several parquet files.

        :param path: directory where dataset will be stored. Will be created if necessary.
        :param kwargs: will be handed through to `to_parquet` functions

        .. seealso:: :func:`read_parquet`
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        super().to_parquet(path=path, **kwargs)

        # write tables
        for table, table_path in zip(
            [
                self.property_links,
                self.class_links,
            ],
            [
                self.__class__._PROPERTY_LINKS_PATH,
                self.__class__._CLASS_LINKS_PATH,
            ],
        ):
            table.to_parquet(path.joinpath(table_path), **kwargs)

    @classmethod
    def _read_parquet_values(
        cls,
        path: Union[str, pathlib.Path],
        backend: BACKEND_LITERAL = "pandas",
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        init_args, additional_init_args = super()._read_parquet_values(
            path=path, backend=backend, **kwargs
        )

        for table, table_path in zip(
            [
                "property_links",
                "class_links",
            ],
            [
                cls._PROPERTY_LINKS_PATH,
                cls._CLASS_LINKS_PATH,
            ],
        ):
            additional_init_args[table] = pd.read_parquet(
                path.joinpath(table_path), **kwargs
            )
        return init_args, additional_init_args
