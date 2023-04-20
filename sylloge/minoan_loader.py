from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import pandas as pd
from pystow import Module
from tqdm import tqdm

from .base import BASE_DATASET_MODULE, EADataset
from .nt_reader import NTReader

MINOAN_MODULE = BASE_DATASET_MODULE.module("minoan")


@dataclass
class DataSetFileInfo:
    url: str
    inner_path_nt: str
    inner_path_map: str


@dataclass
class MinoanFileInfo:
    left: DataSetFileInfo
    right: DataSetFileInfo
    ground_truth_url: str
    ground_truth_inner_path: str


_BTC12DBpedia_URL = "http://csd.uoc.gr/~vefthym/minoanER/datasets/dbpedia37.tar.gz"
_BTC12DBpedia_inner_path_nt = "dbpedia37EntityIds.nt"
_BTC12DBpedia_inner_path_map = "dbpediaIds.txt"
_LOCAH_URL = "http://csd.uoc.gr/~vefthym/minoanER/datasets/locah.tar.gz"
_D5_GT_URL = "http://csd.uoc.gr/~vefthym/minoanER/datasets/groundTruths/D5_GT.tar.gz"

MinoanDSVariant = Literal["D2", "D3", "D4", "D5"]
_Variant_to_FileInfo: Dict[MinoanDSVariant, MinoanFileInfo] = {
    "D5": MinoanFileInfo(
        left=DataSetFileInfo(
            url=_BTC12DBpedia_URL,
            inner_path_nt=_BTC12DBpedia_inner_path_nt,
            inner_path_map=_BTC12DBpedia_inner_path_map,
        ),
        right=DataSetFileInfo(
            url="http://csd.uoc.gr/~vefthym/minoanER/datasets/locah.tar.gz",
            inner_path_nt="locahNewEntityIds.nt",
            inner_path_map="locahNewIds.txt",
        ),
        ground_truth_url="http://csd.uoc.gr/~vefthym/minoanER/datasets/groundTruths/D5_GT.tar.gz",
        ground_truth_inner_path="locahNew_groundTruth.txt",
    )
}


class IdMappedLineCleaner:
    def __init__(self, module: Module, url: str, inner_path: str):
        self.id_map = {}
        with module.ensure_open_tarfile(
            url=url,
            inner_path=inner_path,
            mode="r",
        ) as id_map_file:
            for line in tqdm(id_map_file, desc=f"Reading Id Map File {inner_path}"):
                uri, idx = line.decode("utf-8").strip().split("\t")

                self.id_map[idx] = uri

    def clean_line(self, line: str) -> str:
        idx, remainder = line.split(" ", maxsplit=1)
        if remainder.count("<") == 1:
            pred, obj = remainder.split(" ", maxsplit=1)
            return self.id_map[idx] + " " + pred + ' "' + obj + '".'
        return self.id_map[idx] + " " + remainder + " ."


class Minoan(EADataset):
    def __init__(self, dataset_variant: MinoanDSVariant = "D5") -> None:
        self.dataset_variant = dataset_variant
        file_info = _Variant_to_FileInfo[self.dataset_variant]
        left_attr, left_rel = self._load_side_dfs(file_info.left)
        right_attr, right_rel = self._load_side_dfs(file_info.right)
        entity_links = MINOAN_MODULE.ensure_tar_df(
            url=file_info.ground_truth_url,
            inner_path=file_info.ground_truth_inner_path,
            read_csv_kwargs=dict(
                sep=" ", header=None, names=["left", "right", "trailingwhitespace"]
            ),
        )[["left", "right"]]
        super().__init__(
            rel_triples_left=left_rel,
            rel_triples_right=right_rel,
            attr_triples_left=left_attr,
            attr_triples_right=right_attr,
            ent_links=entity_links,
        )

    def _load_side_dfs(
        self, file_info: DataSetFileInfo
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        parser = NTReader(
            IdMappedLineCleaner(
                module=MINOAN_MODULE,
                url=file_info.url,
                inner_path=file_info.inner_path_map,
            ).clean_line
        )
        return parser.read_remote_tarfile(
            MINOAN_MODULE, url=file_info.url, inner_path=file_info.inner_path_nt
        )

if __name__ == "__main__":
    ds = Minoan()
