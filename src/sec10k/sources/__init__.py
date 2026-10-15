from .base import FilingSource
from .edgar import EdgarSource

__all__ = ["EdgarSource", "FilingSource", "SecApiSource"]


def __getattr__(name: str):
    if name == "SecApiSource":  # lazy: sec_api is an optional dependency
        from .secapi import SecApiSource

        return SecApiSource
    raise AttributeError(name)
