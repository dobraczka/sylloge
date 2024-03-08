import logging
import os
import pathlib
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import dask.dataframe as dd
import pandas as pd
import pystow
from eche import ClusterHelper
from pystow.utils import read_zipfile_csv
from slugify import slugify

from .dask import read_dask_df_archive_csv
from .typing import (
    BACKEND_LITERAL,
    COLUMNS,
    EA_SIDES,
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    DataFrameType,
)
from .utils import fix_dataclass_init_docs

BASE_DATASET_KEY = "sylloge"

BASE_DATASET_MODULE = pystow.module(BASE_DATASET_KEY)


logger = logging.getLogger(__name__)


@dataclass
class DatasetStatistics:
    rel_triples: int
    attr_triples: int
    entities: int
    relations: int
    properties: int
    literals: int

    @property
    def triples(self) -> int:
        return self.rel_triples + self.attr_triples


@fix_dataclass_init_docs
@dataclass
class TrainTestValSplit:
    """Dataclass holding split of gold standard entity links."""

    #: entity links for training
    train: ClusterHelper
    #: entity links for testing
    test: ClusterHelper
    #: entity links for validation
    val: ClusterHelper


@fix_dataclass_init_docs
@dataclass
class MultiSourceEADataset(Generic[DataFrameType]):
    """Dataset class holding information of the alignment class."""

    rel_triples: List[DataFrameType]
    attr_triples: List[DataFrameType]
    ent_links: ClusterHelper
    dataset_names: Tuple[str, str]
    folds: Optional[Sequence[TrainTestValSplit]] = None

    @property
    def _canonical_name(self) -> str:
        raise NotImplementedError

    @property
    def canonical_name(self) -> str:
        """A canonical name for this dataset instance.

        This includes all the necessary information
        to distinguish this specific dataset as string.
        This can be used e.g. to create folders with this
        dataset name to store results.

        :return: concise string representation for this dataset instance
        """
        name = self._canonical_name
        assert isinstance(name, str)  # for mypy
        return slugify(name, separator="_")

    def _statistics_side(self, index: int) -> DatasetStatistics:
        attr_triples = self.attr_triples[index]
        rel_triples = self.rel_triples[index]
        num_attr_triples = len(attr_triples)
        num_rel_triples = len(rel_triples)
        num_entities = len(
            set(attr_triples[LABEL_HEAD]).union(
                set(rel_triples[LABEL_HEAD]).union(set(rel_triples[LABEL_TAIL]))
            )
        )
        num_literals = len(set(attr_triples[LABEL_TAIL]))
        num_relations = len(set(rel_triples[LABEL_RELATION]))
        num_properties = len(set(attr_triples[LABEL_RELATION]))
        return DatasetStatistics(
            rel_triples=num_rel_triples,
            attr_triples=num_attr_triples,
            entities=num_entities,
            relations=num_relations,
            properties=num_properties,
            literals=num_literals,
        )

    def statistics(self) -> Tuple[List[DatasetStatistics], int]:
        """Provide statistics of datasets.

        :return: statistics of left dataset, statistics of right dataset and number of gold standard matches
        """
        return (
            [self._statistics_side(idx) for idx in range(len(self.attr_triples))],
            len(self.ent_links),
        )

    def _create_ds_repr(self) -> str:
        ds_stats, num_ent_links = self.statistics()
        ds_stat_repr = " ".join(
            f"rel_triples_{idx}={stat.rel_triples}, rel_triples_{idx}={stat.rel_triples}"
            for idx, stat in enumerate(ds_stats)
        )
        return f"{ds_stat_repr}, ent_links={num_ent_links}, folds={len(self.folds) if self.folds else None}"

    def __repr__(self) -> str:
        ds_stat_repr = self._create_ds_repr()
        return f"{self.__class__.__name__}({ds_stat_repr})"


class ParquetEADataset(MultiSourceEADataset[DataFrameType]):
    """Dataset class holding information of the alignment task."""

    _REL_TRIPLES_PATH: str = "rel_triples"
    _ATTR_TRIPLES_PATH: str = "attr_triples"
    _ENT_LINKS_PATH: str = "ent_links"
    _FOLD_DIR: str = "folds"
    _TRAIN_LINKS_PATH: str = "train"
    _TEST_LINKS_PATH: str = "test"
    _VAL_LINKS_PATH: str = "val"
    _DATASET_NAMES_PATH: str = "dataset_names.txt"

    @overload
    def __init__(
        self: "ParquetEADataset[pd.DataFrame]",
        *,
        rel_triples: Sequence[DataFrameType],
        attr_triples: Sequence[DataFrameType],
        ent_links: DataFrameType,
        dataset_names: Tuple[str, str],
        folds: Optional[Sequence[TrainTestValSplit]] = None,
        backend: Literal["pandas"] = "pandas",
    ):
        ...

    @overload
    def __init__(
        self: "ParquetEADataset[dd.DataFrame]",
        *,
        rel_triples: Sequence[DataFrameType],
        attr_triples: Sequence[DataFrameType],
        ent_links: DataFrameType,
        dataset_names: Tuple[str, str],
        folds: Optional[Sequence[TrainTestValSplit]] = None,
        backend: Literal["dask"] = "dask",
    ):
        ...

    def __init__(
        self,
        *,
        rel_triples: Sequence[DataFrameType],
        attr_triples: Sequence[DataFrameType],
        ent_links: DataFrameType,
        dataset_names: Tuple[str, str],
        folds: Optional[Sequence[TrainTestValSplit]] = None,
        backend: BACKEND_LITERAL = "pandas",
    ) -> None:
        """Create an entity aligment dataclass.

        :param rel_triples: relation triples of knowledge graph
        :param attr_triples: attribute triples of knowledge graph
        :param dataset_names: tuple of dataset names
        :param ent_links: gold standard entity links of alignment
        :param folds: optional pre-split folds of the gold standard
        :param backend: which backend is used of either 'pandas' or 'dask'
        """
        super().__init__(
            rel_triples=rel_triples,  # type: ignore[arg-type]
            attr_triples=attr_triples,  # type: ignore[arg-type]
            ent_links=ent_links,  # type: ignore[arg-type]
            dataset_names=dataset_names,
            folds=folds,  # type: ignore[arg-type]
        )
        self.backend = backend

    @property
    def _param_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        ds_stat_repr = self._create_ds_repr()
        return f"{self.__class__.__name__}(backend={self.backend},{self._param_repr}{ds_stat_repr})"

    def _triple_path_modifier(self, triple_list: List) -> List[str]:
        return [f"_{idx}_" for idx in range(len(triple_list))]

    def to_parquet(self, path: Union[str, pathlib.Path], **kwargs):
        """Write dataset to path as several parquet files.

        :param path: directory where dataset will be stored. Will be created if necessary.
        :param kwargs: will be handed through to `to_parquet` functions

        .. seealso:: :func:`read_parquet`
        """
        if not os.path.exists(path):
            os.makedirs(path)
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        # write dataset names
        with open(path.joinpath(self.__class__._DATASET_NAMES_PATH), "w") as fh:
            for side, name in zip(EA_SIDES, self.dataset_names):
                fh.write(f"{side}:{name}\n")

        # write tables
        for tables, table_prefix in [
            (self.rel_triples, self.__class__._REL_TRIPLES_PATH),
            (self.attr_triples, self.__class__._ATTR_TRIPLES_PATH),
        ]:
            for table, infix in zip(tables, self._triple_path_modifier(tables)):
                table.to_parquet(
                    path.joinpath(f"{table_prefix}{infix}parquet"), **kwargs
                )

        self.ent_links.to_file(path.joinpath(self.__class__._ENT_LINKS_PATH))

        # write folds
        if self.folds:
            fold_path = path.joinpath(self.__class__._FOLD_DIR)
            for fold_number, fold in enumerate(self.folds, start=1):
                fold_dir = fold_path.joinpath(str(fold_number))
                os.makedirs(fold_dir)
                for fold_links, link_path in zip(
                    [fold.train, fold.test, fold.val],
                    [
                        self.__class__._TRAIN_LINKS_PATH,
                        self.__class__._TEST_LINKS_PATH,
                        self.__class__._VAL_LINKS_PATH,
                    ],
                ):
                    fold_links.to_file(
                        fold_dir.joinpath(link_path), write_cluster_id=False
                    )

    @classmethod
    def _read_parquet_values(
        cls,
        path: Union[str, pathlib.Path],
        backend: BACKEND_LITERAL = "pandas",
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        read_parquet_fn = cast(  # did not find another way to get the correct type
            Callable[[Any], DataFrameType],
            pd.read_parquet if backend == "pandas" else dd.read_parquet,
        )

        # read dataset names
        with open(path.joinpath(cls._DATASET_NAMES_PATH)) as fh:
            dataset_names = tuple(line.strip().split(":")[1] for line in fh)
            # for mypy
            dataset_names = cast(Tuple[str, str], dataset_names)

        tables = defaultdict(list)
        # read tables
        for table, table_prefix in [
            ("rel_triples", cls._REL_TRIPLES_PATH),
            ("attr_triples", cls._ATTR_TRIPLES_PATH),
        ]:
            table_glob = f"{path.joinpath(table_prefix).absolute()}_*_parquet"
            for table_path in glob(table_glob):
                tables[table].append(
                    read_parquet_fn(path.joinpath(table_path), **kwargs)
                )

        ent_links = ClusterHelper.from_file(path.joinpath(cls._ENT_LINKS_PATH))

        # read folds
        fold_path = path.joinpath(cls._FOLD_DIR)
        folds = None
        if os.path.exists(fold_path):
            folds = []
            for tmp_fold_dir in sorted(sub_dir for sub_dir in os.listdir(fold_path)):
                fold_dir = fold_path.joinpath(tmp_fold_dir)
                train_test_val: Dict[str, ClusterHelper] = {}
                for links, link_path in zip(
                    ["train", "test", "val"],
                    [
                        cls._TRAIN_LINKS_PATH,
                        cls._TEST_LINKS_PATH,
                        cls._VAL_LINKS_PATH,
                    ],
                ):
                    train_test_val[links] = ClusterHelper.from_file(
                        fold_dir.joinpath(link_path)
                    )
                folds.append(TrainTestValSplit(**train_test_val))
        return (
            {
                "dataset_names": dataset_names,
                "folds": folds,
                "backend": backend,
                "rel_triples": tables["rel_triples"],
                "attr_triples": tables["attr_triples"],
                "ent_links": ent_links,
            },
            {},
        )

    @overload
    @classmethod
    def read_parquet(
        cls,
        path: Union[str, pathlib.Path],
        backend: Literal["dask"],
        **kwargs,
    ) -> "ParquetEADataset[dd.DataFrame]":
        ...

    @overload
    @classmethod
    def read_parquet(
        cls,
        path: Union[str, pathlib.Path],
        backend: Literal["pandas"] = "pandas",
        **kwargs,
    ) -> "ParquetEADataset[pd.DataFrame]":
        ...

    @classmethod
    def read_parquet(
        cls,
        path: Union[str, pathlib.Path],
        backend: BACKEND_LITERAL = "pandas",
        **kwargs,
    ) -> "ParquetEADataset":
        """Read dataset from parquet files in given `path`.

        This function expects the left/right attribute/relation triples and entity links as well as a `dataset_names.txt`

        Optionally folds are read from a `folds` directory, with numbered fold subdirectories containing train/test/val links.

        :param path: Directory with files
        :param backend: Whether to use pandas or dask for reading
        :param kwargs: passed on to the respective read function
        :return: EADataset read from parquet

        .. seealso:: :func:`to_parquet`
        """
        init_kwargs, additional_kwargs = cls._read_parquet_values(
            path=path, backend=backend, **kwargs
        )
        instance = cls(**init_kwargs)
        instance.__dict__.update(additional_kwargs)
        return instance


class CacheableEADataset(ParquetEADataset[DataFrameType]):
    @overload
    def __init__(
        self: "CacheableEADataset[pd.DataFrame]",
        *,
        cache_path: pathlib.Path,
        use_cache: bool = True,
        parquet_load_options: Optional[Mapping] = None,
        parquet_store_options: Optional[Mapping] = None,
        backend: Literal["pandas"],
        **init_kwargs,
    ):
        ...

    @overload
    def __init__(
        self: "CacheableEADataset[dd.DataFrame]",
        *,
        cache_path: pathlib.Path,
        use_cache: bool = True,
        parquet_load_options: Optional[Mapping] = None,
        parquet_store_options: Optional[Mapping] = None,
        backend: Literal["dask"],
        **init_kwargs,
    ):
        ...

    def __init__(
        self,
        *,
        cache_path: pathlib.Path,
        use_cache: bool = True,
        parquet_load_options: Optional[Mapping] = None,
        parquet_store_options: Optional[Mapping] = None,
        backend: BACKEND_LITERAL = "pandas",
        **init_kwargs,
    ):
        """EADataset that uses caching after initial read.

        :param cache_path: Path where cache will be stored/loaded
        :param use_cache: whether to use cache
        :param parquet_load_options: handed through to parquet loading function
        :param parquet_store_options: handed through to parquet writing function
        :param backend: Whether to use pandas or dask for reading/writing
        :param init_kwargs: other arguments for creating the EADataset instance
        """
        self.cache_path = cache_path
        self.parquet_load_options = parquet_load_options or {}
        self.parquet_store_options = parquet_store_options or {}
        update_cache = False
        additional_kwargs: Dict[str, Any] = {}
        if use_cache:
            if self.cache_path.exists():
                logger.info(f"Loading from cache at {self.cache_path}")
                ea_ds_kwargs, new_additional_kwargs = self.load_from_cache(
                    backend=backend
                )
                init_kwargs.update(ea_ds_kwargs)
                additional_kwargs.update(new_additional_kwargs)
            else:
                init_kwargs.update(self.initial_read(backend=backend))
                update_cache = True
        else:
            init_kwargs.update(self.initial_read(backend=backend))
        self.__dict__.update(additional_kwargs)
        if "backend" in init_kwargs:
            backend = init_kwargs.pop("backend")
        super().__init__(backend=backend, **init_kwargs)  # type: ignore[misc,arg-type]
        if update_cache:
            logger.info(f"Caching dataset at {self.cache_path}")
            self.store_cache()

    def create_cache_path(
        self,
        pystow_module: pystow.Module,
        inner_cache_path: str,
        cache_path: Optional[pathlib.Path] = None,
    ) -> pathlib.Path:
        """Use either pystow module or cache_path to create cache path.

        :param pystow_module: module where data is stored
        :param inner_cache_path: path relative to pystow/cache path
        :param cache_path: alternative to pystow module
        :return: cache path as `pathlib.Path`
        """
        if cache_path is None:
            return pystow_module.join("cached", inner_cache_path, ensure_exists=False)
        return cache_path.joinpath(inner_cache_path)

    def load_from_cache(
        self, backend: BACKEND_LITERAL = "pandas"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.__class__._read_parquet_values(
            path=self.cache_path, backend=backend
        )

    @abstractmethod
    def initial_read(self, backend: BACKEND_LITERAL) -> Dict[str, Any]:
        """Read data for initialising EADataset."""

    def store_cache(self):
        self.to_parquet(self.cache_path, **self.parquet_store_options)


class ZipEADataset(CacheableEADataset[DataFrameType]):
    """Dataset created from zip file which is downloaded."""

    @overload
    def __init__(
        self: "ZipEADataset[pd.DataFrame]",
        *,
        cache_path: pathlib.Path,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        dataset_names: Tuple[str, str],
        file_names_rel_triples: Sequence[str] = ("rel_triples_1", "rel_triples_2"),
        file_names_attr_triples: Sequence[str] = ("attr_triples_1", "attr_triples_2"),
        file_name_ent_links: str = "ent_links",
        backend: Literal["pandas"],
        use_cache: bool = True,
    ):
        ...

    @overload
    def __init__(
        self: "ZipEADataset[dd.DataFrame]",
        *,
        cache_path: pathlib.Path,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        dataset_names: Tuple[str, str],
        file_names_rel_triples: Sequence[str] = ("rel_triples_1", "rel_triples_2"),
        file_names_attr_triples: Sequence[str] = ("attr_triples_1", "attr_triples_2"),
        file_name_ent_links: str = "ent_links",
        backend: Literal["dask"],
        use_cache: bool = True,
    ):
        ...

    def __init__(
        self,
        *,
        cache_path: pathlib.Path,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        dataset_names: Tuple[str, str],
        file_names_rel_triples: Sequence[str] = ("rel_triples_1", "rel_triples_2"),
        file_names_attr_triples: Sequence[str] = ("attr_triples_1", "attr_triples_2"),
        file_name_ent_links: str = "ent_links",
        backend: BACKEND_LITERAL = "pandas",
        use_cache: bool = True,
    ):
        """Initialize ZipEADataset.

        :param cache_path: Path where cache will be stored/loaded
        :param zip_path: path to zip archive containing data
        :param inner_path: base path inside zip archive
        :param dataset_names: tuple of dataset names
        :param file_name_rel_triples: file names of relation triples
        :param file_name_attr_triples: file names of attribute triples
        :param file_name_ent_links: file name gold standard containing all entity links
        :param backend: Whether to use "pandas" or "dask"
        :param use_cache: whether to use cache or not
        """
        self.zip_path = zip_path
        self.inner_path = inner_path
        self.file_names_rel_triples = file_names_rel_triples
        self.file_names_attr_triples = file_names_attr_triples
        self.file_name_ent_links = file_name_ent_links

        super().__init__(  # type: ignore[misc]
            dataset_names=dataset_names,
            cache_path=cache_path,
            backend=backend,  # type: ignore[arg-type]
            use_cache=use_cache,
        )

    def initial_read(self, backend: BACKEND_LITERAL) -> Dict[str, Any]:
        rel_triples = [
            self._read_triples(file_name=fn, backend=backend)
            for fn in self.file_names_rel_triples
        ]
        attr_triples = [
            self._read_triples(file_name=fn, backend=backend)
            for fn in self.file_names_attr_triples
        ]
        ent_links = self._read_links(self.inner_path, self.file_name_ent_links)
        return {
            "rel_triples": rel_triples,
            "attr_triples": attr_triples,
            "ent_links": ent_links,
        }

    def _read_links(
        self,
        inner_folder: pathlib.PurePosixPath,
        file_name: Union[str, pathlib.Path],
        sep: str = "\t",
        encoding: str = "utf8",
    ) -> ClusterHelper:
        return ClusterHelper.from_zipped_file(
            path=self.zip_path,
            inner_path=str(inner_folder.joinpath(file_name)),
            has_cluster_id=False,
            sep=sep,
            encoding=encoding,
        )

    @overload
    def _read_triples(
        self,
        file_name: Union[str, pathlib.Path],
        backend: Literal["dask"],
        is_links: bool = False,
    ) -> dd.DataFrame:
        ...

    @overload
    def _read_triples(
        self,
        file_name: Union[str, pathlib.Path],
        backend: Literal["pandas"] = "pandas",
        is_links: bool = False,
    ) -> pd.DataFrame:
        ...

    def _read_triples(
        self,
        file_name: Union[str, pathlib.Path],
        backend: BACKEND_LITERAL = "pandas",
        is_links: bool = False,
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        columns = list(EA_SIDES) if is_links else COLUMNS
        read_csv_kwargs = dict(  # noqa: C408
            header=None,
            names=columns,
            sep="\t",
            encoding="utf8",
            dtype=str,
        )
        if backend == "pandas":
            trip = read_zipfile_csv(
                path=self.zip_path,
                inner_path=str(self.inner_path.joinpath(file_name)),
                **read_csv_kwargs,
            )
        else:
            trip = read_dask_df_archive_csv(
                path=self.zip_path,
                inner_path=str(self.inner_path.joinpath(file_name)),
                protocol="zip",
                **read_csv_kwargs,
            )
        return cast(DataFrameType, trip)


class ZipEADatasetWithPreSplitFolds(ZipEADataset[DataFrameType]):
    """Dataset with pre-split folds created from zip file which is downloaded."""

    @overload
    def __init__(
        self: "ZipEADatasetWithPreSplitFolds[pd.DataFrame]",
        *,
        cache_path: pathlib.Path,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        dataset_names: Tuple[str, str],
        file_names_rel_triples: Sequence[str] = ("rel_triples_1", "rel_triples_2"),
        file_names_attr_triples: Sequence[str] = ("attr_triples_1", "attr_triples_2"),
        file_name_ent_links: str = "ent_links",
        backend: Literal["pandas"],
        directory_name_folds: str = "721_5fold",
        directory_names_individual_folds: Sequence[str] = ("1", "2", "3", "4", "5"),
        file_name_test_links: str = "test_links",
        file_name_train_links: str = "train_links",
        file_name_valid_links: str = "valid_links",
        use_cache: bool = True,
    ):
        ...

    @overload
    def __init__(
        self: "ZipEADatasetWithPreSplitFolds[dd.DataFrame]",
        *,
        cache_path: pathlib.Path,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        dataset_names: Tuple[str, str],
        file_names_rel_triples: Sequence[str] = ("rel_triples_1", "rel_triples_2"),
        file_names_attr_triples: Sequence[str] = ("attr_triples_1", "attr_triples_2"),
        file_name_ent_links: str = "ent_links",
        backend: Literal["dask"],
        directory_name_folds: str = "721_5fold",
        directory_names_individual_folds: Sequence[str] = ("1", "2", "3", "4", "5"),
        file_name_test_links: str = "test_links",
        file_name_train_links: str = "train_links",
        file_name_valid_links: str = "valid_links",
        use_cache: bool = True,
    ):
        ...

    def __init__(
        self,
        *,
        cache_path: pathlib.Path,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        dataset_names: Tuple[str, str],
        file_names_rel_triples: Sequence[str] = ("rel_triples_1", "rel_triples_2"),
        file_names_attr_triples: Sequence[str] = ("attr_triples_1", "attr_triples_2"),
        file_name_ent_links: str = "ent_links",
        backend: BACKEND_LITERAL = "pandas",
        directory_name_folds: str = "721_5fold",
        directory_names_individual_folds: Sequence[str] = ("1", "2", "3", "4", "5"),
        file_name_test_links: str = "test_links",
        file_name_train_links: str = "train_links",
        file_name_valid_links: str = "valid_links",
        use_cache: bool = True,
    ):
        """Initialize ZipEADatasetWithPreSplitFolds.

        :param cache_path: Path where cache will be stored/loaded
        :param zip_path: path to zip archive containing data
        :param inner_path: base path inside zip archive
        :param dataset_names: tuple of dataset names
        :param file_names_rel_triples: file names of relation triples
        :param file_names_attr_triples: file names of attribute triples
        :param file_name_ent_links: file name gold standard containing all entity links
        :param backend: Whether to use "pandas" or "dask"
        :param directory_name_folds: name of the folds directory
        :param directory_names_individual_folds: name of individual folds
        :param file_name_test_links: name of test link file
        :param file_name_train_links: name of train link file
        :param file_name_valid_links: name of valid link file
        :param use_cache: whether to use cache or not
        """
        self.zip_path = zip_path
        self.inner_path = inner_path
        self.directory_names_individual_folds = directory_names_individual_folds
        self.directory_name_folds = directory_name_folds
        self.file_name_train_links = file_name_train_links
        self.file_name_test_links = file_name_test_links
        self.file_name_valid_links = file_name_valid_links

        super().__init__(  # type: ignore[misc]
            dataset_names=dataset_names,
            zip_path=zip_path,
            inner_path=inner_path,
            cache_path=cache_path,
            backend=backend,  # type: ignore[arg-type]
            use_cache=use_cache,
            file_names_rel_triples=file_names_rel_triples,
            file_names_attr_triples=file_names_attr_triples,
            file_name_ent_links=file_name_ent_links,
        )

    def initial_read(self, backend: BACKEND_LITERAL) -> Dict[str, Any]:
        folds = []
        for fold in self.directory_names_individual_folds:
            fold_folder = self.inner_path.joinpath(
                pathlib.Path(self.directory_name_folds).joinpath(fold)
            )
            train = self._read_links(fold_folder, self.file_name_train_links)
            test = self._read_links(fold_folder, self.file_name_test_links)
            val = self._read_links(fold_folder, self.file_name_valid_links)
            folds.append(TrainTestValSplit(train=train, test=test, val=val))
        return {**super().initial_read(backend=backend), "folds": folds}


class BinaryEADataset(MultiSourceEADataset[DataFrameType]):
    """Binary class to get left and right triples easier."""

    @property
    def rel_triples_left(self) -> DataFrameType:
        return self.rel_triples[0]

    @property
    def rel_triples_right(self) -> DataFrameType:
        return self.rel_triples[1]

    @property
    def attr_triples_left(self) -> DataFrameType:
        return self.attr_triples[0]

    @property
    def attr_triples_right(self) -> DataFrameType:
        return self.attr_triples[1]


class BinaryParquetEADataset(
    ParquetEADataset[DataFrameType], BinaryEADataset[DataFrameType]
):
    """Binary version of ParquetEADataset."""


class BinaryCacheableEADataset(CacheableEADataset[DataFrameType], BinaryEADataset):
    """Binary version of CacheableEADataset."""


class BinaryZipEADataset(ZipEADataset[DataFrameType], BinaryEADataset):
    """Binary version of ZipEADataset."""


class BinaryZipEADatasetWithPreSplitFolds(
    ZipEADatasetWithPreSplitFolds[DataFrameType], BinaryEADataset
):
    """Binary version of ZipEADataset."""


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
    ]
    for ds in datasets:
        ds_family = str(ds.__class__.__name__).split(".")[-1]
        ds_stats, num_ent_links = ds.statistics()
        for ds_side, ds_side_name in zip(ds_stats, ds.dataset_names):
            if seperate_attribute_relations:
                rows.append(
                    [
                        ds_family,
                        ds.canonical_name,
                        ds_side_name,
                        ds_side.entities,
                        ds_side.rel_triples,
                        ds_side.attr_triples,
                        ds_side.relations,
                        ds_side.properties,
                        ds_side.literals,
                        num_ent_links,
                    ]
                )
            else:
                rows.append(
                    [
                        ds_family,
                        ds.canonical_name,
                        ds_side_name,
                        ds_side.entities,
                        ds_side.triples,
                        ds_side.relations,
                        ds_side.properties,
                        ds_side.literals,
                        num_ent_links,
                    ]
                )
    statistics_df = pd.DataFrame(
        rows,
        columns=columns,
    )
    return statistics_df.set_index(index_cols)
