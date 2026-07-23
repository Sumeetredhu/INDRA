"""INDRA agents.

Each subpackage is an independent service behind a ``Protocol`` from
:mod:`indra.core.contracts`. Agents never import each other — cross-agent work goes through the
orchestrator's ``bind()`` setters or through the event bus (``docs/DECISIONS.md`` D8).
"""

__all__: list[str] = []
