"""INDRA core: configuration, domain models, protocols, logging, ids, and the event vocabulary.

Nothing in ``indra.core`` imports from an agent package. The dependency arrow always points
inward — agents depend on core, core depends on nobody.
"""

from indra.core.config import Settings, get_settings
from indra.core.exceptions import IndraError
from indra.core.ids import correlation_context, get_correlation_id, new_id
from indra.core.logging import configure_logging, get_logger

__all__ = [
    "IndraError",
    "Settings",
    "configure_logging",
    "correlation_context",
    "get_correlation_id",
    "get_logger",
    "get_settings",
    "new_id",
]
