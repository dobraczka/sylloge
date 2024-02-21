from typing import Any, Mapping, Optional

import dask.bag as db
import dask.dataframe as dd


def read_dask_bag_from_archive_text(
    path: str,
    inner_path: str,
    protocol: str,
    read_text_kwargs: Optional[Mapping[str, Any]] = None,
) -> db.Bag:
    actual_path = f"{protocol}://{inner_path}::{path}"
    text_kwargs = read_text_kwargs or {}
    return db.read_text(actual_path, **text_kwargs)


def read_dask_df_archive_csv(
    path: str,
    inner_path: str,
    protocol: str,
    **read_csv_kwargs,
) -> dd.DataFrame:
    actual_path = f"{protocol}://{inner_path}::{path}"
    return dd.read_csv(actual_path, **read_csv_kwargs)
