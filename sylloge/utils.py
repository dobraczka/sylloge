import logging
from typing import Type

logger = logging.getLogger(__name__)


def fix_dataclass_init_docs(cls: Type) -> Type:
    """Fix the ``__init__`` documentation for a :class:`dataclasses.dataclass`.
    :param cls: The class whose docstring needs fixing
    :returns: The class that was passed so this function can be used as a decorator

    .. seealso:: https://github.com/agronholm/sphinx-autodoc-typehints/issues/123
    """
    cls.__init__.__qualname__ = f"{cls.__name__}.__init__"
    return cls
