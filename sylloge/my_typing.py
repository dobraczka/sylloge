from typing import Literal, Tuple, TypeVar

import dask.dataframe as dd
import pandas as pd
from eche import PrefixedClusterHelper

# borrowed from pykeen.typing
Target = Literal["head", "relation", "tail"]
LABEL_HEAD: Target = "head"
LABEL_RELATION: Target = "relation"
LABEL_TAIL: Target = "tail"
EASide = Literal["left", "right"]
EA_SIDE_LEFT: EASide = "left"
EA_SIDE_RIGHT: EASide = "right"
EA_SIDES: Tuple[EASide, EASide] = (EA_SIDE_LEFT, EA_SIDE_RIGHT)
COLUMNS = [LABEL_HEAD, LABEL_RELATION, LABEL_TAIL]

BACKEND_LITERAL = Literal["pandas", "dask"]
DataFrameType = TypeVar("DataFrameType", pd.DataFrame, dd.DataFrame)
LinkType = TypeVar("LinkType", pd.DataFrame, dd.DataFrame, PrefixedClusterHelper)
