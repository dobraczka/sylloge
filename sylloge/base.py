import pathlib
import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import pandas as pd
import pystow
from pystow.utils import read_zipfile_csv

from .typing import EA_SIDES, LABEL_HEAD, LABEL_RELATION, LABEL_TAIL
from .utils import fix_dataclass_init_docs
from .utils import load_from_rdf

BASE_DATASET_MODULE = pystow.module("sylloge")

logger = logging.getLogger(__name__)

@fix_dataclass_init_docs
@dataclass
class TrainTestValSplit:
    """Dataclass holding split of gold standard entity links."""

    #: entity links for training
    train: pd.DataFrame
    #: entity links for testing
    test: pd.DataFrame
    #: entity links for validation
    val: pd.DataFrame


@fix_dataclass_init_docs
@dataclass
class EADataset:
    """Dataclass holding information of the alignment class."""

    #: relation triples of left knowledge graph
    rel_triples_left: pd.DataFrame
    #: relation triples of right knowledge graph
    rel_triples_right: pd.DataFrame
    #: attribute triples of left knowledge graph
    attr_triples_left: pd.DataFrame
    #: attribute triples of right knowledge graph
    attr_triples_right: pd.DataFrame
    #: gold standard entity links of alignment
    ent_links: pd.DataFrame
    #: optional pre-split folds of the gold standard
    folds: Optional[Sequence[TrainTestValSplit]] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rel_triples_left={len(self.rel_triples_left)}, rel_triples_right={len(self.rel_triples_right)}, attr_triples_left={len(self.attr_triples_left)},attr_triples_right={len(self.attr_triples_right)},ent_links={len(self.ent_links)})"


class ZipEADataset(EADataset):
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
    ):
        """Initialize ZipEADataset.

        :param zip_path: path to zip archive containing data
        :param inner_path: base path inside zip archive
        :param file_name_rel_triples_left: file name of left relation triples
        :param file_name_rel_triples_right: file name of right relation triples
        :param file_name_attr_triples_left: file name of left attribute triples
        :param file_name_attr_triples_right: file name of right attribute triples
        :param file_name_ent_links: file name gold standard containing all entity links
        """
        self.zip_path = zip_path
        self.inner_path = inner_path
        self.file_name_rel_triples_left = file_name_rel_triples_left
        self.file_name_rel_triples_right = file_name_rel_triples_right
        self.file_name_ent_links = file_name_ent_links
        self.file_name_attr_triples_left = file_name_attr_triples_left
        self.file_name_attr_triples_right = file_name_attr_triples_right

        # load data
        rel_triples_left = self._read_triples(file_name=self.file_name_rel_triples_left)
        rel_triples_right = self._read_triples(
            file_name=self.file_name_rel_triples_right
        )
        attr_triples_left = self._read_triples(
            file_name=self.file_name_attr_triples_left
        )
        attr_triples_right = self._read_triples(
            file_name=self.file_name_attr_triples_right
        )
        ent_links = self._read_triples(
            file_name=self.file_name_ent_links, is_links=True
        )
        super().__init__(
            rel_triples_left=rel_triples_left,
            rel_triples_right=rel_triples_right,
            attr_triples_left=attr_triples_left,
            attr_triples_right=attr_triples_right,
            ent_links=ent_links,
        )

    def _read_triples(
        self, file_name: Union[str, pathlib.Path], is_links: bool = False
    ) -> pd.DataFrame:
        columns = (
            list(EA_SIDES) if is_links else (LABEL_HEAD, LABEL_RELATION, LABEL_TAIL)
        )
        return read_zipfile_csv(
            path=self.zip_path,
            inner_path=str(self.inner_path.joinpath(file_name)),
            header=None,
            names=columns,
            sep="\t",
            encoding="utf8",
            dtype=str,
        )

    @abstractmethod
    def _param_repr(self) -> str:
        raise NotImplementedError

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
        )
        self.folds = []
        for fold in directory_names_individual_folds:
            fold_folder = pathlib.Path(directory_name_folds).joinpath(fold)
            train = self._read_triples(
                fold_folder.joinpath(file_name_train_links), is_links=True
            )
            test = self._read_triples(
                fold_folder.joinpath(file_name_test_links), is_links=True
            )
            val = self._read_triples(
                fold_folder.joinpath(file_name_valid_links), is_links=True
            )
            self.folds.append(TrainTestValSplit(train=train, test=test, val=val))

    def __repr__(self) -> str:
        len_folds = None if not self.folds else len(self.folds)
        return f"{self.__class__.__name__}({self._param_repr()}rel_triples_left={len(self.rel_triples_left)}, rel_triples_right={len(self.rel_triples_right)}, attr_triples_left={len(self.attr_triples_left)}, attr_triples_right={len(self.attr_triples_right)}, ent_links={len(self.ent_links)}, folds={len_folds})"

class RDFBasedEADataset(EADataset):
    def __init__(self, left_file: str, right_file: str, links_file: str, left_format: str, right_format: str):
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

