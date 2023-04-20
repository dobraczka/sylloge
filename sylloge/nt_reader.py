import pandas as pd
import warnings
from tqdm import tqdm
from rdflib.plugins.parsers.ntriples import W3CNTriplesParser
from pystow import Module
from rdflib.exceptions import ParserError as ParseError
from rdflib.graph import _ObjectType, _PredicateType, _SubjectType
from rdflib.plugins.parsers.ntriples import NTGraphSink
from rdflib.term import URIRef, Literal, Node

from .base import COLUMNS
from typing import Optional, Callable, Tuple


class PandasNTGraphSink(NTGraphSink):
    def __init__(self):
        super().__init__(None)
        self.attr_store = []
        self.rel_store = []

    def triple(self, s: _SubjectType, p: _PredicateType, o: _ObjectType) -> None:
        assert isinstance(s, URIRef), "Subject %s must be a URIRef" % (s,)
        assert isinstance(p, URIRef), "Predicate %s must be a URIRef" % (p,)
        assert isinstance(o, Node), "Object %s must be an rdflib term" % (o,)
        if isinstance(o, Literal):
            self.attr_store.append((s.toPython(), p.toPython(), o.toPython()))
        else:
            assert isinstance(o, URIRef)
            self.rel_store.append((s.toPython(), p.toPython(), o.toPython()))


class NTReader:
    parser: W3CNTriplesParser

    def __init__(self, pre_parse_clean_fn: Optional[Callable[[str], str]] = None):
        self.pre_parse_clean_fn = pre_parse_clean_fn

    def parse_line(self, line: bytes) -> None:
        line_str = line.decode("utf-8").strip()
        pre_parse_line_str = line_str
        if self.pre_parse_clean_fn:
            line_str = self.pre_parse_clean_fn(line_str)
        self.parser.line = line_str
        try:
            self.parser.parseline()
        except ParseError:
            warnings.warn(f"Failed to parse line: {pre_parse_line_str}")

    def read_remote_tarfile(
        self, pystow_module: Module, url: str, inner_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sink = PandasNTGraphSink()
        self.parser = W3CNTriplesParser(sink=sink)
        with pystow_module.ensure_open_tarfile(
            url=url,
            inner_path=inner_path,
            mode="r",
        ) as file_stream:
            for line in tqdm(file_stream, desc=f"Reading nt file {inner_path}"):
                self.parse_line(line)
        return pd.DataFrame(sink.attr_store, columns=COLUMNS), pd.DataFrame(sink.rel_store, columns=COLUMNS)
