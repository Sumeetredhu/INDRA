"""Persistence layer.

Every protocol in :mod:`indra.core.contracts` has two implementations here — a real backend and an
in-process one — plus :func:`~indra.storage.factory.build_stores`, which picks between them.
Nothing outside this package should import a concrete store; go through the factory.
"""

from indra.storage.factory import StoreBundle, build_stores

__all__ = ["StoreBundle", "build_stores"]
