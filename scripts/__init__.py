"""Operator-facing scripts for INDRA: corpus generation, seeding, verification, environment check.

Nothing in this package is imported by the running service. These are entry points a human (or CI)
runs from the repository root::

    python -m scripts.setup              # dependency + environment check
    python -m scripts.generate_demo_data # build the synthetic plant corpus
    python -m scripts.seed_demo_data     # generate -> ingest -> verify -> summarise
    python -m scripts.run_demo_check     # execute every beat of the 3-minute demo script

The corpus these scripts build is *deterministic*: fixed seed, fixed dates, no ``datetime.now()``
anywhere in generated content, and byte-stable output files. That is what makes the demo
reproducible from a cold clone and what makes content-addressed ingestion (``docs/DECISIONS.md``
D6) idempotent across rehearsals.
"""

from __future__ import annotations

__all__: list[str] = []
