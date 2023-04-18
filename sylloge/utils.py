import logging
from typing import Callable, Type, List, Any, Tuple

import pandas as pd
import rdflib
from rdflib import Graph
from rdflib.term import Literal
from tqdm import tqdm
from .typing import COLUMNS

logger = logging.getLogger(__name__)


def fix_dataclass_init_docs(cls: Type) -> Type:
    """Fix the ``__init__`` documentation for a :class:`dataclasses.dataclass`.
    :param cls: The class whose docstring needs fixing
    :returns: The class that was passed so this function can be used as a decorator

    .. seealso:: https://github.com/agronholm/sphinx-autodoc-typehints/issues/123
    """
    cls.__init__.__qualname__ = f"{cls.__name__}.__init__"
    return cls


def cast_to_python_type(lit: rdflib.term.Literal):
    """Casts a literal to the respective python type.

    :param lit : The literal that is to be cast.

    :returns:  The literal as the respective python object
    """
    if lit.datatype is not None and "langString" in lit.datatype:
        # lang strings are not automatically cast
        return str(lit)
    return lit.toPython()


def from_rdflib(
    graph: Graph,
    literal_cleaning_func: Callable = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create forayer knowledge graph object from rdflib graph.

    :param graph: Rdflib graph to transform.
    :param literal_cleaning_func: Function to preprocess literals,
        if None will simply cast to python types.
    :param format: Triple format ("xml”, “n3” (use for turtle), “nt” or “trix”).

    :returns: relation and attribute triples dataframe
    """
    if literal_cleaning_func is None:
        literal_cleaning_func = cast_to_python_type
    attr_triples: List[Tuple[str,str,Any]] = []
    rel_triples: List[Tuple[str,str,str]] = []
    for stmt in tqdm(graph, desc="Transforming graph", total=len(graph)):
        s, p, o = stmt
        if isinstance(o, Literal):
            value = literal_cleaning_func(o)
            attr_triples.append((str(s),str(p), value))
        else:
            rel_triples.append((str(s),str(o), str(p)))
    return pd.DataFrame(rel_triples, columns =  COLUMNS), pd.DataFrame(attr_triples, columns=COLUMNS)


def load_from_rdf(
    in_path: str,
    literal_cleaning_func: Callable = None,
    format: str = None,
    kg_name: str = None,
    multi_value: Callable = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create knowledge graph object from rdf source.

    :param in_path: Path of triple file.
    :param literal_cleaning_func: Function to preprocess literals,
        if None will simply cast to python types.
    :param format: Triple format ("xml”, “n3” (use for turtle), “nt” or “trix”).

    :returns: relation and attribute triples dataframe
    """
    g = Graph()
    logger.info(f"Reading graph from {in_path}. This might take a while...")
    g.parse(in_path, format=format)
    return from_rdflib(
        g,
        literal_cleaning_func=literal_cleaning_func,
    )
