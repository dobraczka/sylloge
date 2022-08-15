from importlib.metadata import version  # pragma: no cover

from .open_ea_loader import OpenEA

__all__ = ["OpenEA"]
__version__ = version(__package__)
