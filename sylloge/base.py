import logging
import pathlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)

import pandas as pd
import pystow
from pystow.utils import read_zipfile_csv
from slugify import slugify

from .dask import read_dask_df_archive_csv
from .typing import EA_SIDES, LABEL_HEAD, LABEL_RELATION, LABEL_TAIL
from .utils import fix_dataclass_init_docs, load_from_rdf

BASE_DATASET_KEY = "sylloge"

BASE_DATASET_MODULE = pystow.module(BASE_DATASET_KEY)

T = TypeVar("T")

BACKEND_LITERAL = Literal["pandas", "dask"]

if TYPE_CHECKING:
    import dask.dataframe as dd

logger = logging.getLogger(__name__)


@fix_dataclass_init_docs
@dataclass
class TrainTestValSplit(Generic[T]):
    """Dataclass holding split of gold standard entity links."""

    #: entity links for training
    train: T
    #: entity links for testing
    test: T
    #: entity links for validation
    val: T


@fix_dataclass_init_docs
@dataclass
class EADataset(Generic[T]):
    """Dataclass holding information of the alignment class."""

    #: relation triples of left knowledge graph
    rel_triples_left: T
    #: relation triples of right knowledge graph
    rel_triples_right: T
    #: attribute triples of left knowledge graph
    attr_triples_left: T
    #: attribute triples of right knowledge graph
    attr_triples_right: T
    #: gold standard entity links of alignment
    ent_links: T
    #: optional pre-split folds of the gold standard
    folds: Optional[Sequence[TrainTestValSplit[T]]] = None

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

    def _param_repr(self) -> str:
        raise NotImplementedError

    @property
    def _statistics(self) -> str:
        if hasattr(self.rel_triples_left, "__len__"):
            return f"rel_triples_left={len(self.rel_triples_left)}, rel_triples_right={len(self.rel_triples_right)}, attr_triples_left={len(self.attr_triples_left)}, attr_triples_right={len(self.attr_triples_right)}, ent_links={len(self.ent_links)}, folds={len(self.folds) if self.folds else None}"  # type: ignore
        else:
            unknown = "unknown_len"
            return f"rel_triples_left={unknown}, rel_triples_right={unknown}, attr_triples_left={unknown}, attr_triples_right={unknown}, ent_links={unknown}, folds={unknown if self.folds else None}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._param_repr()}{self._statistics})"


class ZipEADataset(EADataset[pd.DataFrame]):
    """Dataset created from zip file which is downloaded."""

    def __init__(
        self,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        file_name_rel_triples_left: str = "rel_triples_1",
        file_name_rel_triples_right: str = "rel_triples_2",
        file_name_attr_triples_left: str = "attr_triples_1",
        file_name_attr_triples_right: str = "attr_triples_2",
        file_name_ent_links: str = "ent_links",
        backend: BACKEND_LITERAL = "pandas",
    ):
        """Initialize ZipEADataset.

        :param zip_path: path to zip archive containing data
        :param inner_path: base path inside zip archive
        :param file_name_rel_triples_left: file name of left relation triples
        :param file_name_rel_triples_right: file name of right relation triples
        :param file_name_attr_triples_left: file name of left attribute triples
        :param file_name_attr_triples_right: file name of right attribute triples
        :param file_name_ent_links: file name gold standard containing all entity links
        :param backend: Whether to use "pandas" or "dask"
        """
        self.zip_path = zip_path
        self.inner_path = inner_path
        self.file_name_rel_triples_left = file_name_rel_triples_left
        self.file_name_rel_triples_right = file_name_rel_triples_right
        self.file_name_ent_links = file_name_ent_links
        self.file_name_attr_triples_left = file_name_attr_triples_left
        self.file_name_attr_triples_right = file_name_attr_triples_right
        self.backend = backend

        # load data
        rel_triples_left = self._read_triples(
            file_name=self.file_name_rel_triples_left, backend=self.backend
        )
        rel_triples_right = self._read_triples(
            file_name=self.file_name_rel_triples_right, backend=self.backend
        )
        attr_triples_left = self._read_triples(
            file_name=self.file_name_attr_triples_left, backend=self.backend
        )
        attr_triples_right = self._read_triples(
            file_name=self.file_name_attr_triples_right, backend=self.backend
        )
        ent_links = self._read_triples(
            file_name=self.file_name_ent_links, is_links=True, backend=self.backend
        )
        super().__init__(
            rel_triples_left=rel_triples_left,
            rel_triples_right=rel_triples_right,
            attr_triples_left=attr_triples_left,
            attr_triples_right=attr_triples_right,
            ent_links=ent_links,
        )

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
        columns = (
            list(EA_SIDES) if is_links else (LABEL_HEAD, LABEL_RELATION, LABEL_TAIL)
        )
        read_csv_kwargs = dict(
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
        else:
            return read_dask_df_archive_csv(
                path=self.zip_path,
                inner_path=str(self.inner_path.joinpath(file_name)),
                protocol="zip",
                **read_csv_kwargs,
            )

    def _param_repr(self) -> str:
        return f"backend={self.backend}, "


class ZipEADatasetWithPreSplitFolds(ZipEADataset):
    """Dataset with pre-split folds created from zip file which is downloaded."""

    def __init__(
        self,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        file_name_rel_triples_left: str = "rel_triples_1",
        file_name_rel_triples_right: str = "rel_triples_2",
        file_name_ent_links: str = "ent_links",
        file_name_attr_triples_left: str = "attr_triples_1",
        file_name_attr_triples_right: str = "attr_triples_2",
        backend: BACKEND_LITERAL = "pandas",
        directory_name_folds: str = "721_5fold",
        directory_names_individual_folds: Sequence[str] = ("1", "2", "3", "4", "5"),
        file_name_test_links: str = "test_links",
        file_name_train_links: str = "train_links",
        file_name_valid_links: str = "valid_links",
    ):
        """Initialize ZipEADatasetWithPreSplitFolds.

        :param zip_path: path to zip archive containing data
        :param inner_path: base path inside zip archive
        :param file_name_rel_triples_left: file name of left relation triples
        :param file_name_rel_triples_right: file name of right relation triples
        :param file_name_attr_triples_left: file name of left attribute triples
        :param file_name_attr_triples_right: file name of right attribute triples
        :param backend: Whether to use "pandas" or "dask"
        :param file_name_ent_links: file name gold standard containing all entity links
        :param directory_name_folds: directory name containing folds
        :param directory_names_individual_folds: directory names of individual folds
        :param file_name_test_links: name of test links file
        :param file_name_train_links: name of train links file
        :param file_name_valid_links: name of valid links file
        """
        super().__init__(
            zip_path=zip_path,
            inner_path=inner_path,
            file_name_rel_triples_left=file_name_rel_triples_left,
            file_name_rel_triples_right=file_name_rel_triples_right,
            file_name_ent_links=file_name_ent_links,
            file_name_attr_triples_left=file_name_attr_triples_left,
            file_name_attr_triples_right=file_name_attr_triples_right,
            backend=backend,
        )
        self.folds = []
        for fold in directory_names_individual_folds:
            fold_folder = pathlib.Path(directory_name_folds).joinpath(fold)
            train = self._read_triples(
                fold_folder.joinpath(file_name_train_links),
                is_links=True,
                backend=self.backend,
            )
            test = self._read_triples(
                fold_folder.joinpath(file_name_test_links),
                is_links=True,
                backend=self.backend,
            )
            val = self._read_triples(
                fold_folder.joinpath(file_name_valid_links),
                is_links=True,
                backend=self.backend,
            )
            self.folds.append(TrainTestValSplit(train=train, test=test, val=val))


class RDFBasedEADataset(EADataset):
    def __init__(
        self,
        left_file: str,
        right_file: str,
        links_file: str,
        left_format: str,
        right_format: str,
    ):
        logger.info("Loading left graph...")
        left_rel, left_attr = load_from_rdf(left_file, format=left_format)
        logger.info("Loading right graph...")
        right_rel, right_attr = load_from_rdf(right_file, format=right_format)
        ent_links = self._load_entity_links(links_file)
        super().__init__(
            rel_triples_left=left_rel,
            rel_triples_right=right_rel,
            attr_triples_left=left_attr,
            attr_triples_right=right_attr,
            ent_links=ent_links,
        )

    @abstractmethod
    def _load_entity_links(self, ref_path: str) -> pd.DataFrame:
        raise NotImplementedError
