"""OAEI knowledge graph track dataset class."""
import logging
import typing
from collections import namedtuple
from typing import Dict, Literal, Tuple

import pandas as pd
from joblib import Memory
from rdflib import Graph
from tqdm import tqdm

from .base import BASE_DATASET_MODULE, RDFBasedEADataset
from .typing import EA_SIDES
from .utils import load_from_rdf

OAEI_MODULE = BASE_DATASET_MODULE.module("oaei")

TASK_NAME = Literal[
    "starwars-swg",
    "starwars-swtor",
    "marvelcinematicuniverse-marvel",
    "memoryalpha-memorybeta",
    "memoryalpha-stexpanded",
]
OAEITaskFiles = namedtuple(
    "OAEITaskFiles",
    [
        "kg1",
        "kg2",
        "ref",
    ],
)

logger = logging.getLogger(__name__)
cachedir = OAEI_MODULE.base
memory = Memory(cachedir, verbose=0)


class OAEI(RDFBasedEADataset):
    """The  OAEI (Ontology Alignment Evaluation Initiative) Knowledge Graph Track tasks contain graphs created from fandom wikis.

    Five integration tasks are available:
        - starwars-swg
        - starwars-swtor
        - marvelcinematicuniverse-marvel
        - memoryalpha-memorybeta
        - memoryalpha-stexpanded

    More information can be found at the `website <http://oaei.ontologymatching.org/2019/knowledgegraph/index.html>`_.
    """

    _STARWARS = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/starwars-swg/component/source/"
    _TOR = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/starwars-swtor/component/target/"
    _SWG = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/starwars-swg/component/target/"
    _MARVEL = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/marvelcinematicuniverse-marvel/component/target/"
    _MCU = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/marvelcinematicuniverse-marvel/component/source/"
    _MEMORY_ALPHA = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/memoryalpha-stexpanded/component/source/"
    _STEXPANDED = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/memoryalpha-stexpanded/component/target/"
    _MEMORY_BETA = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/memoryalpha-memorybeta/component/target/"
    _SW_SWG = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/starwars-swg/component/reference.xml"
    _SW_TOR = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/starwars-swtor/component/reference.xml"
    _MCU_MARVEL = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/marvelcinematicuniverse-marvel/component/reference.xml"
    _MEMORY_ALPHA_BETA = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/memoryalpha-memorybeta/component/reference.xml"
    _MEMORY_ALPHA_STEXPANDED = "http://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph/v3/suite/memoryalpha-stexpanded/component/reference.xml"
    _TASKS = {
        "starwars-swg": OAEITaskFiles(kg1=_STARWARS, kg2=_SWG, ref=_SW_SWG),
        "starwars-swtor": OAEITaskFiles(kg1=_STARWARS, kg2=_TOR, ref=_SW_TOR),
        "marvelcinematicuniverse-marvel": OAEITaskFiles(
            kg1=_MCU, kg2=_MARVEL, ref=_MCU_MARVEL
        ),
        "memoryalpha-memorybeta": OAEITaskFiles(
            kg1=_MEMORY_ALPHA, kg2=_MEMORY_BETA, ref=_MEMORY_ALPHA_BETA
        ),
        "memoryalpha-stexpanded": OAEITaskFiles(
            kg1=_MEMORY_ALPHA, kg2=_STEXPANDED, ref=_MEMORY_ALPHA_STEXPANDED
        ),
    }

    def __init__(self, task: str = "starwars-swg"):
        """Initialize a OAEI Knowledge Graph Track task.

        Parameters
        ----------
        task : str
            Name of the task.
            Has to be one of {starwars-swg,starwars-swtor,marvelcinematicuniverse-marvel,memoryalpha-memorybeta, memoryalpha-stexpanded}
        """
        if not task in typing.get_args(TASK_NAME):
            raise ValueError(f"Task has to be one of {TASK_NAME}, but got{task}")
        self.task = task
        task_files = OAEI._TASKS[self.task]
        file_format = "xml"
        super().__init__(
            left_file=task_files.kg1,
            right_file=task_files.kg2,
            links_file=task_files.ref,
            left_format=file_format,
            right_format=file_format,
        )

    def _load_entity_links(self, ref_path: str) -> pd.DataFrame:
        logger.info("Parsing ref graph")
        ref = Graph()
        ref.parse(ref_path, format="xml")
        pairs: Dict[str, Tuple[str, str]] = {}
        for stmt in tqdm(ref, desc="Gathering links"):
            s, p, o = stmt
            if (
                "http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity"
                in str(p)
            ):
                # key is BNode id, value is entity id tuple
                tup: Tuple[str, str] = ("", "")
                if str(s) in pairs:
                    tup = pairs[str(s)]
                if "alignmententity1" in str(p):
                    tup = (str(o), tup[1])
                else:
                    tup = (tup[0], str(o))
                pairs[str(s)] = tup
        links = [{t[0], t[1]} for _, t in pairs.items()]
        return pd.DataFrame(links, columns=EA_SIDES)
