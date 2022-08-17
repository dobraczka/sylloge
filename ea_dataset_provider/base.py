import pathlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import pandas as pd
import pystow
from pystow.utils import read_zipfile_csv

# borrowed from pykeen.typing
Target = Literal["head", "relation", "tail"]
LABEL_HEAD: Target = "head"
LABEL_RELATION: Target = "relation"
LABEL_TAIL: Target = "tail"
EASide = Literal["left", "right"]
EA_SIDE_LEFT: EASide = "left"
EA_SIDE_RIGHT: EASide = "right"
EA_SIDES: Tuple[EASide, EASide] = (EA_SIDE_LEFT, EA_SIDE_RIGHT)

BASE_DATASET_MODULE = pystow.module("ea-dataset-provider")


@dataclass
class TrainTestValSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    val: pd.DataFrame


@dataclass
class EADataset:
    rel_triples_left: pd.DataFrame
    rel_triples_right: pd.DataFrame
    attr_triples_left: pd.DataFrame
    attr_triples_right: pd.DataFrame
    ent_links: pd.DataFrame
    folds: Sequence[TrainTestValSplit] = None


class ZipEADataset(EADataset):
    def __init__(
        self,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        file_name_rel_triples_left: str = "rel_triples_1",
        file_name_rel_triples_right: str = "rel_triples_2",
        file_name_ent_links: str = "ent_links",
        file_name_attr_triples_left: str = "attr_triples_1",
        file_name_attr_triples_right: str = "attr_triples_2",
    ):
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

    def _read_triples(self, file_name: str, is_links: bool = False) -> pd.DataFrame:
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._param_repr()}rel_triples_left={len(self.rel_triples_left)}, rel_triples_right={len(self.rel_triples_right)}, attr_triples_left={len(self.attr_triples_left)}, attr_triples_right={len(self.attr_triples_right)}, ent_links={len(self.ent_links)})"


class ZipEADatasetWithPreSplitFolds(ZipEADataset):
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
        return f"{self.__class__.__name__}({self._param_repr()}rel_triples_left={len(self.rel_triples_left)}, rel_triples_right={len(self.rel_triples_right)}, attr_triples_left={len(self.attr_triples_left)}, attr_triples_right={len(self.attr_triples_right)}, ent_links={len(self.ent_links)}, folds={len(self.folds)})"
