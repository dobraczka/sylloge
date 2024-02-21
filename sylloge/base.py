import logging
import os
import pathlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import dask.dataframe as dd
import pandas as pd
import pystow
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
)
from .utils import fix_dataclass_init_docs

BASE_DATASET_KEY = "sylloge"

BASE_DATASET_MODULE = pystow.module(BASE_DATASET_KEY)

DataFrameType = TypeVar("DataFrameType", pd.DataFrame, dd.DataFrame)


if TYPE_CHECKING:
    import dask.dataframe as dd

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
class TrainTestValSplit(Generic[DataFrameType]):
    """Dataclass holding split of gold standard entity links."""

    #: entity links for training
    train: DataFrameType
    #: entity links for testing
    test: DataFrameType
    #: entity links for validation
    val: DataFrameType


@fix_dataclass_init_docs
class EADataset(Generic[DataFrameType]):
    """Dataset class holding information of the alignment class."""

    rel_triples_left: DataFrameType
    rel_triples_right: DataFrameType
    attr_triples_left: DataFrameType
    attr_triples_right: DataFrameType
    ent_links: DataFrameType
    dataset_names: Tuple[str, str]
    folds: Optional[Sequence[TrainTestValSplit[DataFrameType]]] = None

    _REL_TRIPLES_LEFT_PATH: str = "rel_triples_left_parquet"
    _REL_TRIPLES_RIGHT_PATH: str = "rel_triples_right_parquet"
    _ATTR_TRIPLES_LEFT_PATH: str = "attr_triples_left_parquet"
    _ATTR_TRIPLES_RIGHT_PATH: str = "attr_triples_right_parquet"
    _ENT_LINKS_PATH: str = "ent_links_parquet"
    _FOLD_DIR: str = "folds"
    _TRAIN_LINKS_PATH: str = "train_parquet"
    _TEST_LINKS_PATH: str = "test_parquet"
    _VAL_LINKS_PATH: str = "val_parquet"
    _DATASET_NAMES_PATH: str = "dataset_names.txt"

    def __init__(
        self,
        *,
        rel_triples_left: DataFrameType,
        rel_triples_right: DataFrameType,
        attr_triples_left: DataFrameType,
        attr_triples_right: DataFrameType,
        ent_links: DataFrameType,
        dataset_names: Tuple[str, str],
        folds: Optional[Sequence[TrainTestValSplit[DataFrameType]]] = None,
        backend: BACKEND_LITERAL = "pandas",
        npartitions: int = 1,
    ) -> None:
        """Create an entity aligment dataclass.

        :param rel_triples_left: relation triples of left knowledge graph
        :param rel_triples_right: relation triples of right knowledge graph
        :param attr_triples_left: attribute triples of left knowledge graph
        :param attr_triples_right: attribute triples of right knowledge graph
        :param dataset_names: tuple of dataset names
        :param ent_links: gold standard entity links of alignment
        :param folds: optional pre-split folds of the gold standard
        :param backend: which backend is used of either 'pandas' or 'dask'
        :param npartitions: how many partitions to use for each frame, when using dask
        """
        self.rel_triples_left = rel_triples_left
        self.rel_triples_right = rel_triples_right
        self.attr_triples_left = attr_triples_left
        self.attr_triples_right = attr_triples_right
        self.ent_links = ent_links
        self.dataset_names = dataset_names
        self.folds = folds
        self.npartitions: int = npartitions
        self._backend: BACKEND_LITERAL = backend
        # trigger possible transformation
        self.backend = backend

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

    def _statistics_side(self, left: bool) -> DatasetStatistics:
        if left:
            attr_triples = self.attr_triples_left
            rel_triples = self.rel_triples_left
        else:
            attr_triples = self.attr_triples_right
            rel_triples = self.rel_triples_right
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

    def statistics(self) -> Tuple[DatasetStatistics, DatasetStatistics, int]:
        """Provide statistics of datasets.

        :return: statistics of left dataset, statistics of right dataset and number of gold standard matches
        """
        return (
            self._statistics_side(True),
            self._statistics_side(False),
            len(self.ent_links),
        )

    @property
    def _param_repr(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        left_ds_stats, right_ds_stats, num_ent_links = self.statistics()
        return f"{self.__class__.__name__}(backend={self.backend}, {self._param_repr}rel_triples_left={left_ds_stats.rel_triples}, rel_triples_right={right_ds_stats.rel_triples}, attr_triples_left={left_ds_stats.attr_triples}, attr_triples_right={right_ds_stats.attr_triples}, ent_links={num_ent_links}, folds={len(self.folds) if self.folds else None})"

    def _additional_backend_handling(self, backend: BACKEND_LITERAL):
        pass

    @property
    def backend(self) -> BACKEND_LITERAL:
        return self._backend

    @backend.setter
    def backend(self, backend: BACKEND_LITERAL):
        """Set backend and transform data if needed."""
        if backend == "pandas":
            self._backend = "pandas"
            if isinstance(self.rel_triples_left, pd.DataFrame):
                return
            self.rel_triples_left = self.rel_triples_left.compute()
            self.rel_triples_right = self.rel_triples_right.compute()
            self.attr_triples_left = self.attr_triples_left.compute()
            self.attr_triples_right = self.attr_triples_right.compute()
            self.ent_links = self.ent_links.compute()
            if self.folds:
                for fold in self.folds:
                    fold.train = fold.train.compute()
                    fold.test = fold.test.compute()
                    fold.val = fold.val.compute()

        elif backend == "dask":
            self._backend = "dask"
            if isinstance(self.rel_triples_left, dd.DataFrame):
                if self.rel_triples_left.npartitions != self.npartitions:
                    self.rel_triples_left = self.rel_triples_left.repartition(
                        npartitions=self.npartitions
                    )
                    self.rel_triples_right = self.rel_triples_right.repartition(
                        npartitions=self.npartitions
                    )
                    self.attr_triples_left = self.attr_triples_left.repartition(
                        npartitions=self.npartitions
                    )
                    self.attr_triples_right = self.attr_triples_right.repartition(
                        npartitions=self.npartitions
                    )
                    self.ent_links = self.ent_links.repartition(
                        npartitions=self.npartitions
                    )
                    if self.folds:
                        for fold in self.folds:
                            fold.train = fold.train.repartition(
                                npartitions=self.npartitions
                            )
                            fold.test = fold.test.repartition(
                                npartitions=self.npartitions
                            )
                            fold.val = fold.val.repartition(
                                npartitions=self.npartitions
                            )
                else:
                    return

            else:
                self.rel_triples_left = dd.from_pandas(
                    self.rel_triples_left, npartitions=self.npartitions
                )
                self.rel_triples_right = dd.from_pandas(
                    self.rel_triples_right, npartitions=self.npartitions
                )
                self.attr_triples_left = dd.from_pandas(
                    self.attr_triples_left, npartitions=self.npartitions
                )
                self.attr_triples_right = dd.from_pandas(
                    self.attr_triples_right, npartitions=self.npartitions
                )
                self.ent_links = dd.from_pandas(
                    self.ent_links, npartitions=self.npartitions
                )
                if self.folds:
                    for fold in self.folds:
                        fold.train = dd.from_pandas(
                            fold.train, npartitions=self.npartitions
                        )
                        fold.test = dd.from_pandas(
                            fold.test, npartitions=self.npartitions
                        )
                        fold.val = dd.from_pandas(
                            fold.val, npartitions=self.npartitions
                        )
        else:
            raise ValueError(f"Unknown backend {backend}")
        self._additional_backend_handling(backend)

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
        for table, table_path in zip(
            [
                self.rel_triples_left,
                self.rel_triples_right,
                self.attr_triples_left,
                self.attr_triples_right,
                self.ent_links,
            ],
            [
                self.__class__._REL_TRIPLES_LEFT_PATH,
                self.__class__._REL_TRIPLES_RIGHT_PATH,
                self.__class__._ATTR_TRIPLES_LEFT_PATH,
                self.__class__._ATTR_TRIPLES_RIGHT_PATH,
                self.__class__._ENT_LINKS_PATH,
            ],
        ):
            table.to_parquet(path.joinpath(table_path), **kwargs)

        # write folds
        if self.folds:
            fold_path = path.joinpath(self.__class__._FOLD_DIR)
            for fold_number, fold in enumerate(self.folds, start=1):
                fold_dir = fold_path.joinpath(str(fold_number))
                os.makedirs(fold_dir)
                for _, link_path in zip(
                    [fold.train, fold.test, fold.val],
                    [
                        self.__class__._TRAIN_LINKS_PATH,
                        self.__class__._TEST_LINKS_PATH,
                        self.__class__._VAL_LINKS_PATH,
                    ],
                ):
                    table.to_parquet(fold_dir.joinpath(link_path), **kwargs)

    @classmethod
    def _read_parquet_values(
        cls,
        path: Union[str, pathlib.Path],
        backend: BACKEND_LITERAL = "pandas",
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        read_parquet_fn = pd.read_parquet if backend == "pandas" else dd.read_parquet

        # read dataset names
        with open(path.joinpath(cls._DATASET_NAMES_PATH)) as fh:
            dataset_names = tuple(line.strip().split(":")[1] for line in fh)
            # for mypy
            dataset_names = cast(Tuple[str, str], dataset_names)

        tables = {}
        # read tables
        for table, table_path in zip(
            [
                "rel_triples_left",
                "rel_triples_right",
                "attr_triples_left",
                "attr_triples_right",
                "ent_links",
            ],
            [
                cls._REL_TRIPLES_LEFT_PATH,
                cls._REL_TRIPLES_RIGHT_PATH,
                cls._ATTR_TRIPLES_LEFT_PATH,
                cls._ATTR_TRIPLES_RIGHT_PATH,
                cls._ENT_LINKS_PATH,
            ],
        ):
            tables[table] = read_parquet_fn(path.joinpath(table_path), **kwargs)

        # read folds
        fold_path = path.joinpath(cls._FOLD_DIR)
        folds = None
        if os.path.exists(fold_path):
            folds = []
            for tmp_fold_dir in sorted(sub_dir for sub_dir in os.listdir(fold_path)):
                fold_dir = fold_path.joinpath(tmp_fold_dir)
                train_test_val = {}
                for links, link_path in zip(
                    ["train", "test", "val"],
                    [
                        cls._TRAIN_LINKS_PATH,
                        cls._TEST_LINKS_PATH,
                        cls._VAL_LINKS_PATH,
                    ],
                ):
                    train_test_val[links] = read_parquet_fn(
                        fold_dir.joinpath(link_path), **kwargs
                    )
                folds.append(TrainTestValSplit(**train_test_val))
        npartitions = 1
        if backend == "dask":
            npartitions = tables["rel_triples_left"].npartitions
        return (
            dict(
                dataset_names=dataset_names,
                folds=folds,
                backend=backend,
                npartitions=npartitions,
                **tables,
            ),
            {},
        )

    @classmethod
    def read_parquet(
        cls,
        path: Union[str, pathlib.Path],
        backend: BACKEND_LITERAL = "pandas",
        **kwargs,
    ) -> "EADataset":
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


class CacheableEADataset(EADataset[DataFrameType]):
    def __init__(
        self,
        *,
        cache_path: pathlib.Path,
        use_cache: bool = True,
        parquet_load_options: Optional[Mapping] = None,
        parquet_store_options: Optional[Mapping] = None,
        **init_kwargs,
    ):
        """EADataset that uses caching after initial read.

        :param cache_path: Path where cache will be stored/loaded
        :param use_cache: whether to use cache
        :param parquet_load_options: handed through to parquet loading function
        :param parquet_store_options: handed through to parquet writing function
        :param init_kwargs: other arguments for creating the EADataset instance
        """
        self.cache_path = cache_path
        self.parquet_load_options = parquet_load_options or {}
        self.parquet_store_options = parquet_store_options or {}
        backend = init_kwargs["backend"]
        specific_npartitions = init_kwargs["npartitions"]
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
        if specific_npartitions != 1:
            init_kwargs["npartitions"] = specific_npartitions
        self.__dict__.update(additional_kwargs)
        super().__init__(**init_kwargs)
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


class ZipEADataset(CacheableEADataset):
    """Dataset created from zip file which is downloaded."""

    def __init__(
        self,
        *,
        cache_path: pathlib.Path,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        dataset_names: Tuple[str, str],
        file_name_rel_triples_left: str = "rel_triples_1",
        file_name_rel_triples_right: str = "rel_triples_2",
        file_name_attr_triples_left: str = "attr_triples_1",
        file_name_attr_triples_right: str = "attr_triples_2",
        file_name_ent_links: str = "ent_links",
        backend: BACKEND_LITERAL = "pandas",
        npartitions: int = 1,
        use_cache: bool = True,
    ):
        """Initialize ZipEADataset.

        :param cache_path: Path where cache will be stored/loaded
        :param zip_path: path to zip archive containing data
        :param inner_path: base path inside zip archive
        :param dataset_names: tuple of dataset names
        :param file_name_rel_triples_left: file name of left relation triples
        :param file_name_rel_triples_right: file name of right relation triples
        :param file_name_attr_triples_left: file name of left attribute triples
        :param file_name_attr_triples_right: file name of right attribute triples
        :param file_name_ent_links: file name gold standard containing all entity links
        :param backend: Whether to use "pandas" or "dask"
        :param npartitions: how many partitions to use for each frame, when using dask
        :param use_cache: whether to use cache or not
        """
        self.zip_path = zip_path
        self.inner_path = inner_path
        self.file_name_rel_triples_left = file_name_rel_triples_left
        self.file_name_rel_triples_right = file_name_rel_triples_right
        self.file_name_ent_links = file_name_ent_links
        self.file_name_attr_triples_left = file_name_attr_triples_left
        self.file_name_attr_triples_right = file_name_attr_triples_right

        super().__init__(
            dataset_names=dataset_names,
            cache_path=cache_path,
            backend=backend,
            npartitions=npartitions,
            use_cache=use_cache,
        )

    def initial_read(self, backend: BACKEND_LITERAL) -> Dict[str, Any]:
        return {
            "rel_triples_left": self._read_triples(
                file_name=self.file_name_rel_triples_left, backend=backend
            ),
            "rel_triples_right": self._read_triples(
                file_name=self.file_name_rel_triples_right, backend=backend
            ),
            "attr_triples_left": self._read_triples(
                file_name=self.file_name_attr_triples_left, backend=backend
            ),
            "attr_triples_right": self._read_triples(
                file_name=self.file_name_attr_triples_right, backend=backend
            ),
            "ent_links": self._read_triples(
                file_name=self.file_name_ent_links, is_links=True, backend=backend
            ),
        }

    @overload
    def _read_triples(
        self,
        file_name: Union[str, pathlib.Path],
        backend: Literal["pandas"],
        is_links: bool = False,
    ) -> pd.DataFrame:
        ...

    @overload
    def _read_triples(
        self,
        file_name: Union[str, pathlib.Path],
        backend: Literal["dask"],
        is_links: bool = False,
    ) -> "dd.DataFrame":
        ...

    def _read_triples(
        self,
        file_name: Union[str, pathlib.Path],
        backend: BACKEND_LITERAL,
        is_links: bool = False,
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        columns = list(EA_SIDES) if is_links else COLUMNS
        read_csv_kwargs = dict(  # noqa: C408
            header=None,
            names=columns,
            sep="\t",
            encoding="utf8",
            dtype=str,
        )
        if backend == "pandas":
            return read_zipfile_csv(
                path=self.zip_path,
                inner_path=str(self.inner_path.joinpath(file_name)),
                **read_csv_kwargs,
            )
        return read_dask_df_archive_csv(
            path=self.zip_path,
            inner_path=str(self.inner_path.joinpath(file_name)),
            protocol="zip",
            **read_csv_kwargs,
        )


class ZipEADatasetWithPreSplitFolds(ZipEADataset):
    """Dataset with pre-split folds created from zip file which is downloaded."""

    def __init__(
        self,
        *,
        cache_path: pathlib.Path,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        dataset_names: Tuple[str, str],
        file_name_rel_triples_left: str = "rel_triples_1",
        file_name_rel_triples_right: str = "rel_triples_2",
        file_name_ent_links: str = "ent_links",
        file_name_attr_triples_left: str = "attr_triples_1",
        file_name_attr_triples_right: str = "attr_triples_2",
        backend: BACKEND_LITERAL = "pandas",
        npartitions: int = 1,
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
        :param file_name_rel_triples_left: file name of left relation triples
        :param file_name_rel_triples_right: file name of right relation triples
        :param file_name_attr_triples_left: file name of left attribute triples
        :param file_name_attr_triples_right: file name of right attribute triples
        :param file_name_ent_links: file name gold standard containing all entity links
        :param backend: Whether to use "pandas" or "dask"
        :param npartitions: how many partitions to use for each frame, when using dask
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

        super().__init__(
            dataset_names=dataset_names,
            zip_path=zip_path,
            inner_path=inner_path,
            cache_path=cache_path,
            backend=backend,
            npartitions=npartitions,
            use_cache=use_cache,
            file_name_rel_triples_left=file_name_rel_triples_left,
            file_name_rel_triples_right=file_name_rel_triples_right,
            file_name_ent_links=file_name_ent_links,
            file_name_attr_triples_left=file_name_attr_triples_left,
            file_name_attr_triples_right=file_name_attr_triples_right,
        )

    def initial_read(self, backend: BACKEND_LITERAL) -> Dict[str, Any]:
        folds = []
        for fold in self.directory_names_individual_folds:
            fold_folder = pathlib.Path(self.directory_name_folds).joinpath(fold)
            train = self._read_triples(
                fold_folder.joinpath(self.file_name_train_links),
                is_links=True,
                backend=backend,
            )
            test = self._read_triples(
                fold_folder.joinpath(self.file_name_test_links),
                is_links=True,
                backend=backend,
            )
            val = self._read_triples(
                fold_folder.joinpath(self.file_name_valid_links),
                is_links=True,
                backend=backend,
            )
            folds.append(TrainTestValSplit(train=train, test=test, val=val))
        return {**super().initial_read(backend=backend), "folds": folds}


def create_statistics_df(
    datasets: Iterable[EADataset], seperate_attribute_relations: bool = True
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
        "Matches",
    ]
    for ds in datasets:
        ds_family = str(ds.__class__.__name__).split(".")[-1]
        ds_left_stats, ds_right_stats, num_ent_links = ds.statistics()
        for ds_side, ds_side_name in zip(
            [ds_left_stats, ds_right_stats], ds.dataset_names
        ):
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
