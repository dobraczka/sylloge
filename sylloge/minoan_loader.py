from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import pandas as pd
from pystow import Module
from tqdm import tqdm

from .base import BASE_DATASET_MODULE, EADataset

# from .nt_reader import NTReader

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
        # parser = NTReader(
        #     IdMappedLineCleaner(
        #         module=MINOAN_MODULE,
        #         url=file_info.url,
        #         inner_path=file_info.inner_path_map,
        #     ).clean_line
        # )
        return parser.read_remote_tarfile(
            MINOAN_MODULE, url=file_info.url, inner_path=file_info.inner_path_nt
        )


if __name__ == "__main__":
    # ds = Minoan()
    from string import ascii_letters, digits
    from urllib.parse import unquote

    from .dask import read_dask_bag_from_archive_text

    path = "/home/dobraczka/.data/sylloge/minoan/dbpedia37.tar.gz"

    def clean_line(line: str) -> Tuple[str, str, str]:
        s, p, o = line.strip().split(" ", maxsplit=2)
        return s, p, o.strip()

    def clean_id_map(line: str) -> Tuple[str, str]:
        iri, idx = line.strip().split("\t")
        return iri, idx

    def clean_id_map_dbp(line: str) -> Tuple[str, str]:
        iri, idx = line.strip().split("\t")
        whitelist = set(digits + ascii_letters + "#_-[]<>:")
        iri = "".join(filter(lambda x: x in whitelist, unquote(iri)))
        return iri, idx

    def clean_gt(line: str) -> Tuple[str, str]:
        left, right = line.strip().split(" ")
        if "dbp" in right:
            right = right.lower()
        return left, right

    # bag = read_dask_bag_from_archive_text(path="/home/dobraczka/.data/sylloge/minoan/dbpedia37.tar.gz",inner_path="dbpedia37EntityIds.nt",protocol="tar")

    # bag3 = read_dask_bag_from_archive_text(path="/home/dobraczka/.data/sylloge/minoan/locah.tar.gz",inner_path="locahNewEntityIds.nt",protocol="tar")

    # print("==DBpedia==")
    # for line in bag.map(clean_line).take(10):
    #     print(line)

    # for line in bag2.take(10):
    #     print(line)

    # print("==locah==")
    # for line in bag3.map(clean_line).take(10):
    #     print(line)

    locah_id_df = (
        read_dask_bag_from_archive_text(
            path="/home/dobraczka/.data/sylloge/minoan/locah.tar.gz",
            inner_path="locahNewIds.txt",
            protocol="tar",
        )
        .map(clean_id_map)
        .to_dataframe(columns=["locah_iri", "locah_id"])
        .set_index("locah_iri")
    )
    dbpedia_id_df = (
        read_dask_bag_from_archive_text(
            path="/home/dobraczka/.data/sylloge/minoan/dbpedia37.tar.gz",
            inner_path="dbpediaIds.txt",
            protocol="tar",
        )
        .map(clean_id_map_dbp)
        .to_dataframe(columns=["dbpedia_iri", "dbpedia_id"])
        .set_index("dbpedia_iri")
    )

    df = (
        read_dask_bag_from_archive_text(
            path="/home/dobraczka/.data/sylloge/minoan/D5_GT.tar.gz",
            inner_path="locahNew_groundTruth.txt",
            protocol="tar",
        )
        .map(clean_gt)
        .to_dataframe(columns=["locah_iri", "dbpedia_iri"])
        .set_index("dbpedia_iri")
        .join(dbpedia_id_df,how="inner")
        .merge(locah_id_df, left_on="locah_iri", right_index=True, how="inner")
        .compute()
    )
    # base = "/home/dobraczka/Downloads/Datasets/MinoanER/mytestsamples/"
    # dbpidmap = (
    #     db.read_text(
    #         f"{base}dbpediaIds.txt",
    #     )
    #     .map(clean_id_map_dbp)
    #     .to_dataframe(columns=["dbpedia_iri", "dbpedia_id"])
    #     .compute()
    # )
    # dbptriples = (
    #     db.read_text(
    #         f"{base}dbpedia37EntityIds.nt",
    #     )
    #     .map(clean_line)
    #     .to_dataframe(columns=["s", "p", "o"])
    #     .compute()
    # )

    # locahidmap = pd.read_csv(
    #     f"{base}locahNewIds.txt",
    #     header=None,
    #     sep="\t",
    #     names=["locah_iri", "locah_id"],
    # )
    # locahtriples = (
    #     db.read_text(
    #         f"{base}locahNewEntityIds.nt",
    #     )
    #     .map(clean_line)
    #     .to_dataframe(columns=["s", "p", "o"])
    #     .compute()
    # )

    # gt = (
    #     db.read_text(
    #         f"{base}locahNew_groundTruth.txt",
    #     )
    #     .map(clean_gt)
    #     .to_dataframe(columns=["locah_iri", "dbpedia_iri"])
    #     .compute()
    # )
    import ipdb  # noqa: autoimport

    ipdb.set_trace()  # BREAKPOINT
