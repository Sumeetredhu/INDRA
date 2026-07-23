"""What a technician carries into a dead zone.

A bundle is a *budgeted, prioritised, checksummed* projection of the plant brain onto a phone. Four
properties drive every decision in this module:

**Priority is criticality-first.** ``settings.offline_priority_order`` (``A``, ``B``, ``C``) is the
primary sort key, refined by open-alert severity and by how much evidence an asset actually has.
A technician walking into a dead zone must carry the pumps that stop production, not the first
twenty tags in alphabetical order.

**Nothing is dropped silently.** When ``settings.offline_bundle_mb`` is exhausted, every asset that
did not fit is named in :attr:`~indra.core.models.OfflineBundle.excluded_tags`, with the reason
recorded in :attr:`BuiltBundle.dropped`. A truncated bundle that *looks* complete is how a
technician ends up standing in front of a pump believing INDRA has nothing on it. Packing is
best-effort rather than all-or-nothing: an oversized asset is skipped and the next one is still
tried, so one enormous manual cannot starve the rest of the shift.

**It is deterministic.** Same graph, same budget, same bytes — same
:attr:`~indra.core.models.OfflineBundle.checksum` and same ``bundle_id``. Every collection is sorted
by a total order before packing, and the checksum covers content only, never wall-clock time. That
is what makes "did my bundle change?" answerable on the device.

**Semantic search comes with it.** The bundle carries a compact embedding index — chunk vectors
L2-normalised and quantised to ``int8`` (a 4× compaction over ``float32``) — so
:mod:`indra.agents.mobile_agent.local_model` can answer by cosine similarity with no network and no
vector database. Chunks whose embedding could not be produced are still packed; they remain
reachable through the index's lexical path.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final, Iterable, Literal, Sequence

import numpy as np

from indra.core.config import Settings
from indra.core.deps import AgentDeps
from indra.core.exceptions import IndraError, OfflineSyncError
from indra.core.ids import content_hash, content_id
from indra.core.logging import get_logger
from indra.core.models import (
    Alert,
    Chunk,
    DocumentMeta,
    Equipment,
    OfflineBundle,
    Procedure,
    Severity,
    SourceRef,
)

logger = get_logger(__name__)


# ======================================================================================
# Shape constants
# ======================================================================================
#
# These describe what a *bundle* looks like, not how INDRA is deployed, which is why they live here
# rather than in ``indra.core.config`` (which is read-only to this agent). Every one is named and
# justified so the packing heuristic stays auditable.

#: Documents carried per asset. Beyond a dozen the marginal document is an old work order the
#: technician will never open, and it costs index space that a procedure would use better.
_MAX_DOCUMENTS_PER_TAG: Final[int] = 12

#: Index chunks carried per asset. Forty passages is roughly a full OEM manual section plus the last
#: year of work orders — enough for the extractive answerer to be useful, small enough to stay compact.
_MAX_CHUNKS_PER_TAG: Final[int] = 40

#: Procedures carried per asset. SOPs are small and disproportionately useful in the field.
_MAX_PROCEDURES_PER_TAG: Final[int] = 8

#: How many alerts to pull from the metadata store in one sweep before grouping them by tag.
_ALERT_FETCH_LIMIT: Final[int] = 500

#: Chunk text is trimmed to this many characters in the index. A passage longer than this is being
#: carried for retrieval, not for reading; the full document is a separate, larger artefact.
_MAX_INDEX_TEXT_CHARS: Final[int] = 700

#: Assets fetched concurrently. Bounded so a 400-tag plant does not open 400 simultaneous graph reads.
_TAG_FETCH_CONCURRENCY: Final[int] = 8

#: Bytes charged for the JSON envelope around one index entry (keys, quotes, separators). Measured
#: against the serialised payload; it keeps the accounting honest rather than optimistic.
_INDEX_ENTRY_OVERHEAD_BYTES: Final[int] = 160

#: Below this much remaining budget nothing further can fit, so the remaining assets are marked
#: excluded without paying for the graph reads that would prove it.
_MIN_USEFUL_BUDGET_BYTES: Final[int] = 4096

#: int8 quantisation scale. A unit-norm component lies in [-1, 1]; 127 uses the full signed range.
_QUANT_SCALE: Final[float] = 127.0

#: Priority weights. Criticality dominates by design — that is the charter's rule — with alerts able
#: to lift a B asset above a quiet A asset only when the alert is severe.
_W_CRITICALITY: Final[float] = 0.60
_W_ALERT_SEVERITY: Final[float] = 0.25
_W_ALERT_COUNT: Final[float] = 0.10
_W_EVIDENCE: Final[float] = 0.05

#: Alert count contribution saturates here; the tenth open alert says nothing the third did not.
_ALERT_COUNT_SATURATION: Final[int] = 3

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/.]*")

DropReason = Literal["budget_exhausted", "asset_too_large", "no_content"]


# ======================================================================================
# Compact embedding index
# ======================================================================================


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One searchable passage inside an offline bundle."""

    chunk_id: str
    document_id: str
    document_title: str
    equipment_tag: str | None
    page: int | None
    text: str
    #: True when a real embedding was packed for this entry. False entries are lexical-only.
    has_vector: bool = True

    def to_source(self, *, relevance: float) -> SourceRef:
        """Project into a citation so an offline answer is as grounded as an online one."""
        return SourceRef(
            document_id=self.document_id,
            document_title=self.document_title,
            chunk_id=self.chunk_id,
            page=self.page,
            snippet=self.text[:600],
            relevance=max(0.0, min(1.0, relevance)),
            retrieved_via="direct",
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A scored index entry, with the method that found it."""

    entry: IndexEntry
    score: float
    method: Literal["semantic", "lexical"]


class LocalIndex:
    """A quantised, self-contained embedding index that fits on a phone.

    Vectors are stored as one ``int8`` matrix (rows already L2-normalised before quantisation), so
    cosine similarity is a single matrix-vector product with no dependency beyond numpy. Entries
    that never received an embedding keep a zero row: they score 0 semantically and remain reachable
    through :meth:`lexical_search`, which is better than dropping evidence the technician has.
    """

    def __init__(self, entries: Sequence[IndexEntry], codes: np.ndarray, *, dimensions: int) -> None:
        if codes.shape[0] != len(entries):
            raise OfflineSyncError(
                "Offline index is misaligned: the vector matrix and the entry list differ in length. "
                "Rebuild the bundle.",
                context={"entries": len(entries), "rows": int(codes.shape[0])},
            )
        self._entries: tuple[IndexEntry, ...] = tuple(entries)
        self._codes: np.ndarray = codes.astype(np.int8, copy=False)
        self._dimensions = int(dimensions)
        self._matrix: np.ndarray | None = None
        self._document_frequency: dict[str, int] | None = None

    # -- introspection -----------------------------------------------------------------
    @property
    def entries(self) -> tuple[IndexEntry, ...]:
        return self._entries

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def is_empty(self) -> bool:
        return not self._entries

    @property
    def vector_count(self) -> int:
        """How many entries carry a real embedding (as opposed to lexical-only)."""
        return sum(1 for entry in self._entries if entry.has_vector)

    @property
    def nbytes(self) -> int:
        """Serialised size of the index, used by the budget accounting."""
        text_bytes = sum(len(entry.text.encode("utf-8")) + _INDEX_ENTRY_OVERHEAD_BYTES for entry in self._entries)
        return int(self._codes.nbytes) + text_bytes

    @classmethod
    def empty(cls, *, dimensions: int) -> LocalIndex:
        return cls([], np.zeros((0, max(1, dimensions)), dtype=np.int8), dimensions=dimensions)

    # -- search ------------------------------------------------------------------------
    def search(self, vector: Sequence[float], *, top_k: int = 8) -> list[SearchHit]:
        """Cosine search over the packed vectors.

        Scores are the raw cosine clamped at zero. A negative cosine between two text embeddings
        carries no usable meaning, and mapping the range to ``[0, 1]`` the way the online vector
        store does would give a zero row (an entry with no embedding) a misleading 0.5.
        """
        if self.is_empty or top_k <= 0:
            return []
        matrix = self._float_matrix()
        query = np.asarray(vector, dtype=np.float32)
        if query.ndim != 1 or query.size == 0:
            return []
        if query.shape[0] != matrix.shape[1]:
            # Dimension drift means the bundle was built by a different embedding provider than the
            # one answering now. Say so instead of ranking noise.
            logger.warning(
                "offline query embedding does not match the bundle index; falling back to lexical search",
                extra={"query_dim": int(query.shape[0]), "index_dim": int(matrix.shape[1])},
            )
            return []
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        scores = matrix @ (query / norm)
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(self._entries))]
        hits = [
            SearchHit(entry=self._entries[int(i)], score=round(max(0.0, float(scores[int(i)])), 6), method="semantic")
            for i in order
        ]
        return [hit for hit in hits if hit.score > 0.0]

    def lexical_search(self, query: str, *, top_k: int = 8) -> list[SearchHit]:
        """Rare-term-weighted overlap search. The floor under every offline answer.

        No embeddings, no model, no network — just inverse document frequency over the packed text,
        which is exactly what still works when everything else in the chain is missing.
        """
        if self.is_empty or top_k <= 0:
            return []
        terms = _tokenise(query)
        if not terms:
            return []
        frequency = self._frequencies()
        total = len(self._entries)
        scored: list[tuple[float, int]] = []
        for position, entry in enumerate(self._entries):
            entry_terms = _tokenise(entry.text)
            if not entry_terms:
                continue
            counts = _counts(entry_terms)
            score = 0.0
            for term in set(terms):
                occurrences = counts.get(term, 0)
                if not occurrences:
                    continue
                idf = math.log(1.0 + total / (1.0 + frequency.get(term, 0)))
                score += idf * (1.0 + math.log(occurrences))
            if score > 0.0:
                scored.append((score / (1.0 + math.log(1 + len(entry_terms))), position))
        if not scored:
            return []
        peak = max(score for score, _ in scored) or 1.0
        scored.sort(key=lambda item: (-item[0], self._entries[item[1]].chunk_id))
        return [
            SearchHit(entry=self._entries[position], score=round(score / peak, 6), method="lexical")
            for score, position in scored[:top_k]
        ]

    # -- serialisation -----------------------------------------------------------------
    def to_payload(self) -> dict[str, Any]:
        """JSON-safe representation, ready to ship to the device."""
        return {
            "dimensions": self._dimensions,
            "count": len(self._entries),
            "codes_base64": base64.b64encode(self._codes.tobytes()).decode("ascii"),
            "entries": [
                {
                    "chunk_id": entry.chunk_id,
                    "document_id": entry.document_id,
                    "document_title": entry.document_title,
                    "equipment_tag": entry.equipment_tag,
                    "page": entry.page,
                    "text": entry.text,
                    "has_vector": entry.has_vector,
                }
                for entry in self._entries
            ],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LocalIndex:
        """Rebuild an index produced by :meth:`to_payload`.

        Raises:
            OfflineSyncError: the payload is malformed or the matrix does not match the entry count.
        """
        try:
            dimensions = int(payload["dimensions"])
            raw = base64.b64decode(str(payload.get("codes_base64", "")), validate=True)
            rows = [
                IndexEntry(
                    chunk_id=str(item["chunk_id"]),
                    document_id=str(item["document_id"]),
                    document_title=str(item.get("document_title", "")),
                    equipment_tag=(str(item["equipment_tag"]) if item.get("equipment_tag") else None),
                    page=(int(item["page"]) if item.get("page") is not None else None),
                    text=str(item.get("text", "")),
                    has_vector=bool(item.get("has_vector", True)),
                )
                for item in payload.get("entries", [])
            ]
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            raise OfflineSyncError(
                "Offline index payload is malformed. Re-download the bundle from the server.",
                cause=exc,
            ) from exc
        expected = len(rows) * dimensions
        if len(raw) != expected:
            raise OfflineSyncError(
                "Offline index payload is truncated: the vector matrix does not match the entry count. "
                "Re-download the bundle.",
                context={"expected_bytes": expected, "actual_bytes": len(raw)},
            )
        codes = np.frombuffer(raw, dtype=np.int8).reshape(len(rows), dimensions) if rows else np.zeros(
            (0, max(1, dimensions)), dtype=np.int8
        )
        return cls(rows, codes, dimensions=dimensions)

    # -- internals ---------------------------------------------------------------------
    def _float_matrix(self) -> np.ndarray:
        """De-quantise once and cache. Rows are re-normalised after the int8 round trip."""
        if self._matrix is None:
            matrix = self._codes.astype(np.float32) / _QUANT_SCALE
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self._matrix = matrix / norms
        return self._matrix

    def _frequencies(self) -> dict[str, int]:
        if self._document_frequency is None:
            frequency: dict[str, int] = {}
            for entry in self._entries:
                for term in set(_tokenise(entry.text)):
                    frequency[term] = frequency.get(term, 0) + 1
            self._document_frequency = frequency
        return self._document_frequency


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokens, keeping plant tags (``P-101``) intact as single terms."""
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _counts(terms: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for term in terms:
        counts[term] = counts.get(term, 0) + 1
    return counts


def quantise(vector: Sequence[float], *, dimensions: int) -> np.ndarray:
    """L2-normalise ``vector``, pad or truncate to ``dimensions``, and quantise to ``int8``."""
    row = np.zeros(dimensions, dtype=np.float32)
    values = np.asarray(list(vector)[:dimensions], dtype=np.float32)
    if values.size:
        row[: values.size] = values
    norm = float(np.linalg.norm(row))
    if norm > 0.0:
        row = row / norm
    return np.clip(np.rint(row * _QUANT_SCALE), -_QUANT_SCALE, _QUANT_SCALE).astype(np.int8)


# ======================================================================================
# Packing
# ======================================================================================


@dataclass(frozen=True, slots=True)
class DroppedAsset:
    """An asset that did not make it into the bundle, and why."""

    equipment_tag: str
    reason: DropReason
    required_bytes: int
    remaining_bytes: int
    priority: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "equipment_tag": self.equipment_tag,
            "reason": self.reason,
            "required_bytes": self.required_bytes,
            "remaining_bytes": self.remaining_bytes,
            "priority": round(self.priority, 4),
        }


@dataclass(frozen=True, slots=True)
class AssetPacket:
    """Everything the bundle carries for one asset, already sorted and costed."""

    equipment: Equipment
    priority: float
    documents: tuple[DocumentMeta, ...]
    alerts: tuple[Alert, ...]
    procedures: tuple[Procedure, ...]
    chunks: tuple[Chunk, ...]

    @property
    def is_empty(self) -> bool:
        return not (self.documents or self.alerts or self.procedures or self.chunks)


@dataclass(frozen=True, slots=True)
class BuiltBundle:
    """The packing result: the contract model plus the artefacts it has no field for.

    :class:`~indra.core.models.OfflineBundle` carries tags, documents, alerts and procedures. The
    equipment profiles and the embedding index are just as much part of what ships to the device, so
    they travel here and are serialised together by :meth:`payload`.
    """

    bundle: OfflineBundle
    index: LocalIndex
    equipment: tuple[Equipment, ...]
    dropped: tuple[DroppedAsset, ...]
    build_ms: float = 0.0

    @property
    def excluded_tags(self) -> list[str]:
        return list(self.bundle.excluded_tags)

    def payload(self) -> dict[str, Any]:
        """The complete device payload: bundle, equipment profiles, and the embedding index."""
        return {
            "bundle": self.bundle.model_dump(mode="json"),
            "equipment": [item.model_dump(mode="json") for item in self.equipment],
            "index": self.index.to_payload(),
            "dropped": [item.as_dict() for item in self.dropped],
        }

    def summary(self) -> dict[str, Any]:
        """Compact description for logs, health checks, and the API response header."""
        return {
            "bundle_id": self.bundle.bundle_id,
            "checksum": self.bundle.checksum,
            "equipment_tags": len(self.bundle.equipment_tags),
            "documents": len(self.bundle.documents),
            "alerts": len(self.bundle.alerts),
            "procedures": len(self.bundle.procedures),
            "index_entries": len(self.index.entries),
            "index_vectors": self.index.vector_count,
            "excluded_tags": len(self.bundle.excluded_tags),
            "size_bytes": self.bundle.size_bytes,
            "budget_bytes": self.bundle.budget_bytes,
            "build_ms": round(self.build_ms, 2),
        }


def priority_score(
    equipment: Equipment,
    *,
    alerts: Sequence[Alert],
    order: Sequence[str],
    evidence: int = 0,
) -> float:
    """Score an asset's claim on scarce offline bytes, in ``[0, 1]``.

    Criticality is the dominant term because that is the charter's rule: ``A`` equipment stops
    production or endangers life. Open-alert severity and volume refine within a class, and a small
    evidence term breaks ties towards assets INDRA actually knows something about — carrying a tag
    with no documents helps nobody.
    """
    positions = {value.upper(): index for index, value in enumerate(order)}
    span = max(1, len(positions) - 1) if len(positions) > 1 else 1
    position = positions.get(equipment.criticality.value.upper(), len(positions))
    criticality = 1.0 - min(1.0, position / span)

    if alerts:
        worst = max(alert.severity.rank for alert in alerts)
        severity = worst / Severity.CRITICAL.rank
        volume = min(1.0, len(alerts) / _ALERT_COUNT_SATURATION)
    else:
        severity = 0.0
        volume = 0.0

    density = min(1.0, evidence / _MAX_CHUNKS_PER_TAG) if evidence else 0.0
    score = (
        _W_CRITICALITY * criticality
        + _W_ALERT_SEVERITY * severity
        + _W_ALERT_COUNT * volume
        + _W_EVIDENCE * density
    )
    return round(min(1.0, max(0.0, score)), 6)


class OfflineBundleBuilder:
    """Builds a prioritised, budgeted, checksummed bundle for field use.

    One instance per agent; :meth:`build` holds no state between calls, so concurrent builds are
    safe (they simply both pay for their own graph reads).
    """

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps
        self._settings: Settings = deps.settings

    @property
    def default_budget_bytes(self) -> int:
        return max(0, self._settings.offline_bundle_mb) * 1024 * 1024

    async def build(
        self,
        *,
        budget_bytes: int | None = None,
        equipment_tags: Sequence[str] | None = None,
    ) -> BuiltBundle:
        """Pack the highest-priority slice of the plant that fits inside the budget.

        Args:
            budget_bytes: Override for ``settings.offline_bundle_mb``. Useful for a phone with
                little free storage, and for tests.
            equipment_tags: Restrict the bundle to these assets. ``None`` considers the whole fleet.

        Raises:
            OfflineSyncError: the equipment registry could not be read, so no honest bundle can be
                built. Every other store failure degrades the bundle and is logged, never raised.
        """
        started = time.perf_counter()
        budget = self.default_budget_bytes if budget_bytes is None else max(0, int(budget_bytes))

        fleet = await self._fleet(equipment_tags)
        alerts_by_tag = await self._alerts_by_tag()
        ranked = self._rank(fleet, alerts_by_tag)

        packed: list[AssetPacket] = []
        dropped: list[DroppedAsset] = []
        seen_documents: set[str] = set()
        used = 0

        for batch in _batched(ranked, _TAG_FETCH_CONCURRENCY):
            if used >= budget or (budget - used) < _MIN_USEFUL_BUDGET_BYTES:
                dropped.extend(
                    DroppedAsset(
                        equipment_tag=equipment.tag,
                        reason="budget_exhausted",
                        required_bytes=0,
                        remaining_bytes=max(0, budget - used),
                        priority=priority,
                    )
                    for equipment, priority in _flatten(ranked, after=batch)
                )
                break

            packets = await asyncio.gather(
                *(
                    self._collect(equipment, priority, alerts_by_tag.get(equipment.tag, []))
                    for equipment, priority in batch
                )
            )
            for packet in packets:
                cost = _packet_cost(packet, seen_documents)
                if packet.is_empty:
                    dropped.append(
                        DroppedAsset(
                            equipment_tag=packet.equipment.tag,
                            reason="no_content",
                            required_bytes=0,
                            remaining_bytes=max(0, budget - used),
                            priority=packet.priority,
                        )
                    )
                    continue
                if used + cost > budget:
                    # Skip and keep going: one oversized asset must not starve every smaller,
                    # lower-priority asset behind it.
                    dropped.append(
                        DroppedAsset(
                            equipment_tag=packet.equipment.tag,
                            reason="asset_too_large" if cost > budget else "budget_exhausted",
                            required_bytes=cost,
                            remaining_bytes=max(0, budget - used),
                            priority=packet.priority,
                        )
                    )
                    continue
                used += cost
                packed.append(packet)
                seen_documents.update(document.document_id for document in packet.documents)

        index = await self._build_index(packed)
        bundle = _assemble(packed, dropped, index=index, budget_bytes=budget)
        built = BuiltBundle(
            bundle=bundle,
            index=index,
            equipment=tuple(packet.equipment for packet in packed),
            dropped=tuple(dropped),
            build_ms=(time.perf_counter() - started) * 1000.0,
        )
        logger.info("offline bundle built", extra=built.summary())
        if dropped:
            logger.warning(
                "offline bundle could not carry the whole fleet",
                extra={
                    "excluded_tags": bundle.excluded_tags[:20],
                    "excluded_count": len(bundle.excluded_tags),
                    "budget_bytes": budget,
                    "size_bytes": bundle.size_bytes,
                },
            )
        return built

    # -- collection --------------------------------------------------------------------
    async def _fleet(self, equipment_tags: Sequence[str] | None) -> list[Equipment]:
        try:
            fleet = list(await self._deps.graph.list_equipment())
        except IndraError as exc:
            raise OfflineSyncError(
                "Cannot build an offline bundle: the equipment registry is unreachable. Bring the "
                "graph store up, or run with INDRA_STORAGE_BACKEND=memory, then retry.",
                context={"error": exc.message},
                cause=exc,
            ) from exc
        except Exception as exc:  # defensive: store contract violation
            raise OfflineSyncError(
                "Cannot build an offline bundle: the equipment registry raised an unexpected error.",
                cause=exc,
            ) from exc
        if equipment_tags:
            wanted = {tag.strip().upper() for tag in equipment_tags if tag and tag.strip()}
            fleet = [item for item in fleet if item.tag in wanted]
        return fleet

    async def _alerts_by_tag(self) -> dict[str, list[Alert]]:
        try:
            alerts = list(
                await self._deps.metadata.list_alerts(unresolved_only=True, limit=_ALERT_FETCH_LIMIT)
            )
        except IndraError as exc:
            logger.warning(
                "alert store unavailable; the bundle will carry no alerts",
                extra={"error": exc.message},
            )
            return {}
        except Exception as exc:  # defensive
            logger.warning("alert store raised an untyped error", extra={"error": str(exc)})
            return {}
        grouped: dict[str, list[Alert]] = {}
        for alert in alerts:
            grouped.setdefault(alert.equipment_tag.strip().upper(), []).append(alert)
        for bucket in grouped.values():
            bucket.sort(key=lambda alert: (-alert.severity.rank, alert.alert_id))
        return grouped

    def _rank(
        self, fleet: Sequence[Equipment], alerts_by_tag: dict[str, list[Alert]]
    ) -> list[tuple[Equipment, float]]:
        """Order the fleet by priority, breaking every tie on the tag so builds are reproducible."""
        ranked = [
            (
                equipment,
                priority_score(
                    equipment,
                    alerts=alerts_by_tag.get(equipment.tag, []),
                    order=self._settings.offline_priority_order,
                ),
            )
            for equipment in fleet
        ]
        ranked.sort(key=lambda item: (-item[1], item[0].tag))
        return ranked

    async def _collect(
        self, equipment: Equipment, priority: float, alerts: Sequence[Alert]
    ) -> AssetPacket:
        """Gather one asset's payload. Every store read is wrapped; a failure thins the packet."""
        documents, procedures, chunks = await asyncio.gather(
            self._documents(equipment.tag),
            self._procedures(equipment.tag),
            self._chunks(equipment.tag),
        )
        refined = priority_score(
            equipment,
            alerts=alerts,
            order=self._settings.offline_priority_order,
            evidence=len(chunks),
        )
        return AssetPacket(
            equipment=equipment,
            priority=max(priority, refined),
            documents=tuple(documents),
            alerts=tuple(alerts),
            procedures=tuple(procedures),
            chunks=tuple(chunks),
        )

    async def _documents(self, tag: str) -> list[DocumentMeta]:
        try:
            documents = list(await self._deps.graph.documents_for_tag(tag, limit=_MAX_DOCUMENTS_PER_TAG * 2))
        except IndraError as exc:
            logger.warning("document lookup failed while bundling",
                           extra={"equipment_tag": tag, "error": exc.message})
            return []
        except Exception as exc:  # defensive
            logger.warning("document lookup raised an untyped error while bundling",
                           extra={"equipment_tag": tag, "error": str(exc)})
            return []
        documents.sort(key=lambda meta: (-(meta.document_date or date.min).toordinal(), meta.document_id))
        return documents[:_MAX_DOCUMENTS_PER_TAG]

    async def _procedures(self, tag: str) -> list[Procedure]:
        try:
            procedures = list(await self._deps.graph.procedures_for(tag))
        except IndraError as exc:
            logger.warning("procedure lookup failed while bundling",
                           extra={"equipment_tag": tag, "error": exc.message})
            return []
        except Exception as exc:  # defensive
            logger.warning("procedure lookup raised an untyped error while bundling",
                           extra={"equipment_tag": tag, "error": str(exc)})
            return []
        procedures.sort(key=lambda procedure: procedure.procedure_id)
        return procedures[:_MAX_PROCEDURES_PER_TAG]

    async def _chunks(self, tag: str) -> list[Chunk]:
        """Pull the passages the graph associates with this asset, newest-scored first."""
        key = f"Equipment:{tag.strip().upper()}"
        try:
            ranked = await self._deps.graph.chunks_for_entities([key], limit=_MAX_CHUNKS_PER_TAG)
        except IndraError as exc:
            logger.warning("chunk lookup failed while bundling",
                           extra={"equipment_tag": tag, "error": exc.message})
            return []
        except Exception as exc:  # defensive
            logger.warning("chunk lookup raised an untyped error while bundling",
                           extra={"equipment_tag": tag, "error": str(exc)})
            return []
        if not ranked:
            return []
        chunk_ids = [chunk_id for chunk_id, _ in ranked][:_MAX_CHUNKS_PER_TAG]
        try:
            chunks = list(await self._deps.vectors.get_chunks(chunk_ids))
        except IndraError as exc:
            logger.warning("chunk fetch failed while bundling",
                           extra={"equipment_tag": tag, "error": exc.message})
            return []
        except Exception as exc:  # defensive
            logger.warning("chunk fetch raised an untyped error while bundling",
                           extra={"equipment_tag": tag, "error": str(exc)})
            return []
        chunks.sort(key=lambda chunk: chunk.chunk_id)
        return chunks

    # -- index -------------------------------------------------------------------------
    async def _build_index(self, packed: Sequence[AssetPacket]) -> LocalIndex:
        """Assemble the compact embedding index over every packed chunk.

        Chunks that already carry an embedding are used as-is; the rest are embedded in one batched
        router call. If embedding is impossible (no provider, no network, no key), the entries are
        still packed with a zero row so the lexical path keeps working.
        """
        dimensions = max(1, int(self._settings.embedding_dimensions))
        titles = {
            document.document_id: document.title
            for packet in packed
            for document in packet.documents
        }
        rows: list[tuple[IndexEntry, list[float] | None]] = []
        seen: set[str] = set()
        for packet in packed:
            for chunk in packet.chunks:
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                rows.append(
                    (
                        IndexEntry(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            document_title=titles.get(chunk.document_id, chunk.document_id),
                            equipment_tag=packet.equipment.tag,
                            page=chunk.page,
                            text=chunk.text[:_MAX_INDEX_TEXT_CHARS],
                            has_vector=bool(chunk.embedding),
                        ),
                        list(chunk.embedding) if chunk.embedding else None,
                    )
                )
        if not rows:
            return LocalIndex.empty(dimensions=dimensions)

        rows.sort(key=lambda item: item[0].chunk_id)
        await self._fill_missing_embeddings(rows)
        return await asyncio.to_thread(_pack_index, rows, dimensions)

    async def _fill_missing_embeddings(
        self, rows: list[tuple[IndexEntry, list[float] | None]]
    ) -> None:
        """Embed the chunks the vector store did not hand back with a vector. Best effort."""
        missing = [index for index, (_, vector) in enumerate(rows) if not vector]
        if not missing:
            return
        texts = [rows[index][0].text for index in missing]
        try:
            vectors = await self._deps.llm.embed(texts, task="document")
        except IndraError as exc:
            logger.warning(
                "offline index will be lexical-only for some passages; embedding failed",
                extra={"missing": len(missing), "error": exc.message},
            )
            return
        except Exception as exc:  # defensive: router contract violation
            logger.warning(
                "embedding raised an untyped error while building the offline index",
                extra={"missing": len(missing), "error": str(exc)},
            )
            return
        if len(vectors) != len(missing):
            logger.warning(
                "embedding returned the wrong number of vectors; keeping those passages lexical-only",
                extra={"requested": len(missing), "returned": len(vectors)},
            )
            return
        for position, vector in zip(missing, vectors):
            entry, _ = rows[position]
            rows[position] = (
                IndexEntry(
                    chunk_id=entry.chunk_id,
                    document_id=entry.document_id,
                    document_title=entry.document_title,
                    equipment_tag=entry.equipment_tag,
                    page=entry.page,
                    text=entry.text,
                    has_vector=True,
                ),
                list(vector),
            )


def _pack_index(
    rows: Sequence[tuple[IndexEntry, list[float] | None]], dimensions: int
) -> LocalIndex:
    """Quantise every vector into one int8 matrix. Pure numpy; always called in a worker thread."""
    codes = np.zeros((len(rows), dimensions), dtype=np.int8)
    entries: list[IndexEntry] = []
    for position, (entry, vector) in enumerate(rows):
        entries.append(entry)
        if vector:
            codes[position] = quantise(vector, dimensions=dimensions)
    return LocalIndex(entries, codes, dimensions=dimensions)


# ======================================================================================
# Assembly, costing, checksum
# ======================================================================================


def _assemble(
    packed: Sequence[AssetPacket],
    dropped: Sequence[DroppedAsset],
    *,
    index: LocalIndex,
    budget_bytes: int,
) -> OfflineBundle:
    """Fold the packed assets into the contract model, deterministically."""
    tags = sorted({packet.equipment.tag for packet in packed})
    documents: dict[str, DocumentMeta] = {}
    alerts: dict[str, Alert] = {}
    procedures: dict[str, Procedure] = {}
    for packet in packed:
        for document in packet.documents:
            documents.setdefault(document.document_id, document)
        for alert in packet.alerts:
            alerts.setdefault(alert.alert_id, alert)
        for procedure in packet.procedures:
            procedures.setdefault(procedure.procedure_id, procedure)

    ordered_documents = [documents[key] for key in sorted(documents)]
    ordered_alerts = sorted(alerts.values(), key=lambda alert: (-alert.severity.rank, alert.alert_id))
    ordered_procedures = [procedures[key] for key in sorted(procedures)]
    excluded = sorted({item.equipment_tag for item in dropped})

    size = (
        _json_bytes([meta.model_dump(mode="json") for meta in ordered_documents])
        + _json_bytes([alert.model_dump(mode="json") for alert in ordered_alerts])
        + _json_bytes([procedure.model_dump(mode="json") for procedure in ordered_procedures])
        + index.nbytes
    )

    checksum = _checksum(
        tags=tags,
        document_ids=[meta.document_id for meta in ordered_documents],
        alert_ids=[alert.alert_id for alert in ordered_alerts],
        procedure_ids=[procedure.procedure_id for procedure in ordered_procedures],
        chunk_ids=[entry.chunk_id for entry in index.entries],
        excluded=excluded,
        budget_bytes=budget_bytes,
    )
    return OfflineBundle(
        bundle_id=content_id(checksum, kind="session"),
        equipment_tags=tags,
        size_bytes=size,
        budget_bytes=budget_bytes,
        documents=ordered_documents,
        alerts=ordered_alerts,
        procedures=ordered_procedures,
        excluded_tags=excluded,
        checksum=checksum,
    )


def _checksum(
    *,
    tags: Sequence[str],
    document_ids: Sequence[str],
    alert_ids: Sequence[str],
    procedure_ids: Sequence[str],
    chunk_ids: Sequence[str],
    excluded: Sequence[str],
    budget_bytes: int,
) -> str:
    """SHA-256 over the bundle's *content identity*.

    Deliberately excludes ``built_at`` and ``bundle_id``: two builds of an unchanged plant must
    produce the same checksum, which is what lets a device answer "do I need to re-download?"
    without transferring the bundle.
    """
    canonical = json.dumps(
        {
            "budget_bytes": budget_bytes,
            "equipment_tags": list(tags),
            "documents": list(document_ids),
            "alerts": list(alert_ids),
            "procedures": list(procedure_ids),
            "index": list(chunk_ids),
            "excluded": list(excluded),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return content_hash(canonical)


def _packet_cost(packet: AssetPacket, seen_documents: set[str]) -> int:
    """Marginal bytes this asset adds. Documents already packed for another asset are free."""
    cost = _json_bytes(packet.equipment.model_dump(mode="json"))
    for document in packet.documents:
        if document.document_id not in seen_documents:
            cost += _json_bytes(document.model_dump(mode="json"))
    for alert in packet.alerts:
        cost += _json_bytes(alert.model_dump(mode="json"))
    for procedure in packet.procedures:
        cost += _json_bytes(procedure.model_dump(mode="json"))
    for chunk in packet.chunks:
        cost += len(chunk.text[:_MAX_INDEX_TEXT_CHARS].encode("utf-8")) + _INDEX_ENTRY_OVERHEAD_BYTES
    return cost


def _json_bytes(payload: Any) -> int:
    """Serialised UTF-8 size of ``payload``. The budget is measured in what actually ships."""
    return len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8"))


def _batched(
    ranked: Sequence[tuple[Equipment, float]], size: int
) -> Iterable[Sequence[tuple[Equipment, float]]]:
    for start in range(0, len(ranked), size):
        yield ranked[start : start + size]


def _flatten(
    ranked: Sequence[tuple[Equipment, float]], *, after: Sequence[tuple[Equipment, float]]
) -> Iterable[tuple[Equipment, float]]:
    """Yield every ranked asset from the start of ``after`` onwards — the unvisited remainder."""
    if not after:
        return []
    first = after[0][0].tag
    for position, (equipment, priority) in enumerate(ranked):
        if equipment.tag == first:
            return list(ranked[position:])
    return list(after)


__all__ = [
    "AssetPacket",
    "BuiltBundle",
    "DroppedAsset",
    "IndexEntry",
    "LocalIndex",
    "OfflineBundleBuilder",
    "SearchHit",
    "priority_score",
    "quantise",
]
