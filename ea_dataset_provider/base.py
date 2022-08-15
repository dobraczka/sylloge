import pathlib
from typing import Literal, Optional, Tuple

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

BASE_DATASET_MODULE = pystow.module("ea-dataset-loader")


class EADataset:
    rel_triples_left: pd.DataFrame
    rel_triples_right: pd.DataFrame
    attr_triples_left: pd.DataFrame
    attr_triples_right: pd.DataFrame

    def __init__(
        self,
        zip_path: str,
        inner_path: pathlib.PurePosixPath,
        file_name_rel_triples_left: str = "rel_triples_1",
        file_name_rel_triples_right: str = "rel_triples_2",
        file_name_ent_links: str = "ent_links",
        file_name_attr_triples_left: Optional[str] = "attr_triples_1",
        file_name_attr_triples_right: Optional[str] = "attr_triples_2",
    ):
        self.zip_path = zip_path
        self.inner_path = inner_path
        self.file_name_rel_triples_left = file_name_rel_triples_left
        self.file_name_rel_triples_right = file_name_rel_triples_right
        self.file_name_ent_links = file_name_ent_links
        self.file_name_attr_triples_left = (
            file_name_attr_triples_left if file_name_rel_triples_left else None
        )
        self.file_name_attr_triples_right = (
            file_name_attr_triples_right if file_name_rel_triples_right else None
        )

        # load data
        self.rel_triples_left = self._read_triples(
            file_name=self.file_name_rel_triples_left
        )
        self.rel_triples_right = self._read_triples(
            file_name=self.file_name_rel_triples_right
        )
        self.attr_triples_left = self._read_triples(
            file_name=self.file_name_attr_triples_left
        )
        self.attr_triples_right = self._read_triples(
            file_name=self.file_name_attr_triples_right
        )

    def _read_triples(self, file_name: str, is_links: bool = False):
        columns = (
            list(EA_SIDES) if is_links else [LABEL_HEAD, LABEL_RELATION, LABEL_TAIL]
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

    def __repr__(self):
        return f"{self.__class__.__name__}(rel_triples_left={len(self.rel_triples_left)}, rel_triples_right={len(self.rel_triples_right)}, attr_triples_left={len(self.attr_triples_left)}, attr_triples_right={len(self.attr_triples_right)})"
