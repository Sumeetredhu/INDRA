"""Entity resolution and cross-document linking.

This module is the reason INDRA can answer a question no single document answers. The same
``P-101`` written in a work order, an inspection PDF, a shift log, an OEM manual and a P&ID must
become **one node with five source documents**. Everything else — traversal, fusion, the graph
preview — is downstream of getting that right.

Three mechanisms, in order of authority:

1. **Exact merge on :attr:`~indra.core.models.ExtractedEntity.key`.** The key is
   ``"<Type>:<CANONICAL OR SURFACE FORM>"``, so two mentions that already agree collapse for free.
2. **The plant tag grammar.** Equipment tags are the canonical join key across the whole plant, so
   they get a dedicated normaliser (``P 101``, ``p-101``, ``P–101`` → ``P-101``) before the key is
   computed, plus a ``rapidfuzz`` pass against the live equipment registry for OCR damage that the
   grammar alone cannot repair (D5). The fuzzy pass never corrects silently: the match score is
   folded into the merge confidence and the rejected candidates are kept as ``alternatives``.
3. **The alias table.** Every surface form ever seen for a key is remembered, so a later document
   writing ``Pump 101`` links to the node ``P-101`` without re-deriving anything.

**Provenance is unioned, never replaced.** When two mentions merge, the survivor carries the union
of their document ids, chunk ids, pages and character offsets. Dropping a source to keep the model
tidy would silently delete the evidence an operator is entitled to see.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final, Iterable, Mapping, Sequence

from rapidfuzz import fuzz, process

from indra.core.config import Settings
from indra.core.logging import get_logger
from indra.core.models import (
    Confidence,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
)

logger = get_logger(__name__)


# ======================================================================================
# The shared plant tag grammar
# ======================================================================================

#: Tag prefixes that are unambiguous plant equipment even without a separator (``PSV214``).
#: ISA-5.1 instrument letters plus the equipment classes this corpus actually contains.
TAG_PREFIXES: Final[frozenset[str]] = frozenset(
    {
        # rotating and static equipment
        "P", "V", "T", "TK", "E", "HX", "HE", "C", "K", "B", "R", "D", "F", "S", "M", "AG", "MX",
        # valves and actuated devices
        "PSV", "PRV", "MOV", "XV", "CV", "FCV", "PCV", "TCV", "LCV", "HV", "BV", "GV",
        # instruments (ISA letter combinations seen on P&IDs)
        "FT", "PT", "TT", "LT", "AT", "VT", "ZT", "FIC", "PIC", "TIC", "LIC", "AIC",
        "FI", "PI", "TI", "LI", "AI", "FE", "PE", "TE", "LE", "PSH", "PSL", "TSH", "TSL",
        "LSH", "LSL", "FSH", "FSL", "PDT", "PDI", "PAH", "PAL", "TAH", "TAL",
        # documents that behave like tags in maintenance systems
        "WO", "PM", "MR", "SR", "CR",
    }
)

#: Uppercase tokens that look like a tag prefix but never are. Without this, ``OISD-STD-118``
#: and ``ISO-9001`` become equipment and pollute every traversal.
TAG_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "ISO", "IEC", "API", "ASME", "ASTM", "ANSI", "NFPA", "OSHA", "DGMS", "PESO", "OISD",
        "STD", "REV", "SEC", "NO", "PAGE", "FIG", "TAB", "TABLE", "IS", "BS", "EN", "DIN",
        "SOP", "RCA", "HAZOP", "SIL", "IP", "NPS", "ANSI", "PPE", "MSDS", "SDS", "QA", "QC",
        "AM", "PM", "IST", "UTC", "GMT", "USD", "INR", "KG", "MM", "CM", "KW", "HP", "RPM",
        "BAR", "PSI", "DEG", "VOL", "PART", "ITEM", "LINE", "CLASS", "TYPE", "MODEL",
    }
)

#: Loose scanner: finds anything shaped like ``<LETTERS><sep?><DIGITS><SUFFIX?>`` in free text.
#: Only a plain hyphen appears in the class because :func:`normalise_unicode` folds every exotic
#: dash to ``-`` before this pattern is applied.
_TAG_SCAN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{1,4})[ \t]*([-_/]?)[ \t]*(\d{2,5})([A-Za-z]{0,2})(?![A-Za-z0-9])"
)

#: Strict grammar for a fully-formed tag, applied after normalisation.
_TAG_STRICT_RE: Final[re.Pattern[str]] = re.compile(r"^([A-Z]{1,4})-(\d{2,5})([A-Z]{0,2})$")

#: Unicode dash variants OCR and word processors emit in place of a plain hyphen.
_DASHES: Final[str] = "‐‑‒–—―−"

#: An alias shorter than this matches half the corpus. ``P-1`` is not a useful alias; ``P-101`` is.
_MIN_ALIAS_LENGTH: Final[int] = 3

#: Longest multi-word alias considered when scanning free text ("main feed water pump" = 4).
_MAX_ALIAS_TOKENS: Final[int] = 4

#: A merge corroborated by many documents may close at most this fraction of the remaining gap
#: to 1.0. Corroboration strengthens a claim; it cannot manufacture certainty that no single
#: source had. Deliberately capped so a hundred copies of one bad OCR read never reach "high".
_CORROBORATION_CEILING: Final[float] = 0.5

#: Cap on the merged ``evidence_text`` of a deduplicated relationship.
_EVIDENCE_MAX_CHARS: Final[int] = 400

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9./-]*")


def normalise_unicode(text: str) -> str:
    """NFKC-normalise and fold exotic dashes to ``-``.

    OCR and copy-paste routinely produce ``P–101`` (en dash) or full-width digits. Folding them here
    means every downstream comparison is on the same alphabet.
    """
    folded = unicodedata.normalize("NFKC", text)
    for dash in _DASHES:
        folded = folded.replace(dash, "-")
    return folded


def canonical_tag(raw: str) -> str | None:
    """Return the canonical plant tag for ``raw``, or ``None`` if it is not tag-shaped.

    The grammar is ``<1-4 LETTERS>-<2-5 DIGITS><0-2 LETTERS>``. Separators are optional on input
    and always present on output, so ``p 101``, ``P101`` and ``P–101`` all yield ``P-101``.

    A bare prefix that is a known stopword (``ISO 9001``) is rejected outright — see
    :data:`TAG_STOPWORDS`.
    """
    if not raw:
        return None
    candidate = normalise_unicode(raw).strip().upper()
    candidate = re.sub(r"[\s_/]+", "-", candidate)
    candidate = re.sub(r"-{2,}", "-", candidate).strip("-")

    match = _TAG_STRICT_RE.match(candidate)
    if match is None:
        # Try the un-separated form, e.g. "P101" or "PSV214".
        loose = re.match(r"^([A-Z]{1,4})(\d{2,5})([A-Z]{0,2})$", candidate)
        if loose is None:
            return None
        match = _TAG_STRICT_RE.match(f"{loose.group(1)}-{loose.group(2)}{loose.group(3)}")
        if match is None:  # pragma: no cover - the reconstruction always matches
            return None

    prefix, digits, suffix = match.group(1), match.group(2), match.group(3)
    if prefix in TAG_STOPWORDS:
        return None
    return f"{prefix}-{digits}{suffix}"


def find_tag_candidates(text: str) -> list[str]:
    """Scan free text for plant tags, in order of first appearance, deduplicated.

    A match is accepted when either the prefix is a known equipment/instrument prefix
    (:data:`TAG_PREFIXES`) **or** the writer used an explicit separator, which is a strong signal
    that the token is a tag rather than a stray measurement. Stopword prefixes are always rejected.
    """
    found: list[str] = []
    seen: set[str] = set()
    for match in _TAG_SCAN_RE.finditer(normalise_unicode(text)):
        prefix = match.group(1).upper()
        separator = match.group(2)
        if prefix in TAG_STOPWORDS:
            continue
        if prefix not in TAG_PREFIXES and not separator:
            continue
        tag = canonical_tag(f"{prefix}-{match.group(3)}{match.group(4).upper()}")
        if tag and tag not in seen:
            seen.add(tag)
            found.append(tag)
    return found


def entity_key(entity_type: EntityType, name: str) -> str:
    """Build the universal node key the whole graph merges on.

    Mirrors :attr:`indra.core.models.ExtractedEntity.key` exactly, so a key computed here and a key
    computed by the model are the same string.
    """
    return f"{entity_type.value}:{name.strip().upper()}"


def display_name(key: str) -> str:
    """Strip the ``<Type>:`` prefix from a node key for human-facing rendering."""
    return key.split(":", 1)[1] if ":" in key else key


def key_type(key: str) -> EntityType | None:
    """Return the :class:`EntityType` encoded in a node key, or ``None`` if it is unrecognised."""
    label = key.split(":", 1)[0] if ":" in key else ""
    try:
        return EntityType(label)
    except ValueError:
        return None


# ======================================================================================
# Registry and aliases
# ======================================================================================


class EquipmentRegistry:
    """The live set of plant tags, used to repair OCR damage a grammar cannot fix.

    ``P-l0l`` is grammatically valid nonsense: the grammar happily produces ``P-L0L``… which is not
    a tag at all because ``L0L`` is not digits. Fuzzy matching against the tags that *actually
    exist* is what turns it back into ``P-101``. The cutoff is ``settings.pid_tag_fuzzy_threshold``
    (D5) and the runner-up candidates are always returned, never discarded.
    """

    __slots__ = ("_settings", "_tags")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tags: set[str] = set()

    def __len__(self) -> int:
        return len(self._tags)

    @property
    def tags(self) -> frozenset[str]:
        """Every tag currently known to the plant."""
        return frozenset(self._tags)

    def add(self, tag: str) -> bool:
        """Register a canonical tag. Returns ``True`` if it was new."""
        canonical = canonical_tag(tag)
        if canonical is None or canonical in self._tags:
            return False
        self._tags.add(canonical)
        return True

    def extend(self, tags: Iterable[str]) -> int:
        """Register many tags. Returns the count of newly-added ones."""
        return sum(1 for tag in tags if self.add(tag))

    def resolve(self, raw: str) -> tuple[str | None, float, list[str]]:
        """Resolve ``raw`` to a registry tag.

        Returns:
            ``(tag_or_None, confidence, alternatives)``.

            * A canonical tag already in the registry → confidence ``1.0``.
            * A canonical tag not yet in the registry → confidence ``0.9``: the grammar is
              satisfied, so it is almost certainly a real tag from a document we have not indexed
              the registry entry for yet. Admitting it is better than discarding a real asset.
            * A fuzzy match above the cutoff → confidence ``score / 100``, with the other
              candidates returned so nothing is corrected silently.
            * No match → ``(None, 0.0, alternatives)``.
        """
        canonical = canonical_tag(raw)
        if canonical is not None:
            if canonical in self._tags:
                return canonical, 1.0, []
            return canonical, 0.9, []

        if not self._tags:
            return None, 0.0, []

        probe = re.sub(r"[\s_]+", "-", normalise_unicode(raw).strip().upper())
        cutoff = float(self._settings.pid_tag_fuzzy_threshold)
        matches = process.extract(probe, sorted(self._tags), scorer=fuzz.WRatio, limit=4)
        accepted = [(tag, score) for tag, score, _ in matches if score >= cutoff]
        if not accepted:
            return None, 0.0, [tag for tag, _, _ in matches[:3]]

        best_tag, best_score = accepted[0]
        alternatives = [tag for tag, _ in accepted[1:]]
        return best_tag, round(best_score / 100.0, 4), alternatives


class AliasTable:
    """Every surface form ever seen for a node key, both directions.

    The forward map answers "what node is this text talking about?" during query-time entity
    resolution. The reverse map is what lets the graph preview label a node ``P-101 (Boiler Feed
    Pump)`` instead of a bare key.
    """

    __slots__ = ("_forward", "_reverse")

    def __init__(self) -> None:
        self._forward: dict[str, str] = {}
        self._reverse: dict[str, set[str]] = {}

    def __len__(self) -> int:
        return len(self._forward)

    @staticmethod
    def normalise(surface: str) -> str:
        """Fold a surface form to its lookup key: unicode-normalised, uppercased, single-spaced."""
        folded = normalise_unicode(surface).strip().upper()
        return re.sub(r"\s+", " ", folded)

    def add(self, surface: str, key: str) -> bool:
        """Record ``surface`` as an alias of ``key``. Returns ``True`` if this was new.

        A surface form already bound to a *different* key is not rebound — first binding wins, and
        the collision is logged. Silently re-pointing an alias would make retrieval non-reproducible
        between runs, which is worse than an occasional missed link.
        """
        normalised = self.normalise(surface)
        if len(normalised) < _MIN_ALIAS_LENGTH:
            return False
        existing = self._forward.get(normalised)
        if existing is not None:
            if existing != key:
                logger.debug(
                    "alias collision: keeping first binding",
                    extra={"alias": normalised, "bound_to": existing, "rejected": key},
                )
            return False
        self._forward[normalised] = key
        self._reverse.setdefault(key, set()).add(normalised)
        return True

    def lookup(self, surface: str) -> str | None:
        """Return the node key bound to ``surface``, if any."""
        return self._forward.get(self.normalise(surface))

    def aliases_for(self, key: str) -> frozenset[str]:
        """Every surface form bound to ``key``."""
        return frozenset(self._reverse.get(key, ()))

    def scan(self, text: str) -> list[str]:
        """Return node keys whose alias appears in ``text``, longest alias first.

        Implemented as n-gram lookup rather than a scan over every alias: the cost is proportional
        to the length of the text, not to the size of the plant.
        """
        tokens = _WORD_RE.findall(normalise_unicode(text).upper())
        hits: dict[str, int] = {}
        for size in range(min(_MAX_ALIAS_TOKENS, len(tokens)), 0, -1):
            for start in range(0, len(tokens) - size + 1):
                phrase = " ".join(tokens[start : start + size])
                key = self._forward.get(phrase)
                if key is not None and key not in hits:
                    hits[key] = size
        return [key for key, _ in sorted(hits.items(), key=lambda item: (-item[1], item[0]))]


# ======================================================================================
# Resolution results
# ======================================================================================


@dataclass(frozen=True, slots=True)
class EntityMention:
    """One occurrence of an entity in one place. Provenance, atomised.

    Every field here survives a merge. Together they are what ``SourceRef`` is built from when the
    Copilot cites the claim.
    """

    surface: str
    document_id: str | None
    chunk_id: str | None
    page: int | None
    char_start: int | None
    char_end: int | None
    confidence: Confidence
    entity_id: str

    def location(self) -> str:
        """Compact human rendering, e.g. ``doc_ab12/chk_0003 p.4``."""
        parts = [p for p in (self.document_id, self.chunk_id) if p]
        stem = "/".join(parts) if parts else "unattributed"
        return f"{stem} p.{self.page}" if self.page else stem


@dataclass(slots=True)
class ResolvedEntity:
    """One graph node, assembled from every mention that resolved to it."""

    key: str
    type: EntityType
    canonical_name: str
    display: str
    mentions: list[EntityMention] = field(default_factory=list)
    aliases: set[str] = field(default_factory=set)
    alternatives: list[str] = field(default_factory=list)
    attributes: dict[str, object] = field(default_factory=dict)
    #: Worst per-mention resolution score seen (1.0 exact, ``fuzz/100`` for a repaired tag).
    match_score: float = 1.0
    merge_confidence: Confidence = field(
        default_factory=lambda: Confidence(value=0.0, rationale="not yet computed", method="aggregate")
    )

    @property
    def document_ids(self) -> list[str]:
        """Distinct source documents, sorted. This list is the cross-document linking claim."""
        return sorted({m.document_id for m in self.mentions if m.document_id})

    @property
    def chunk_ids(self) -> list[str]:
        """Distinct source chunks, sorted."""
        return sorted({m.chunk_id for m in self.mentions if m.chunk_id})

    def compute_confidence(self) -> Confidence:
        """Score the merge and store it on the entity.

        Three factors, in this order:

        1. :meth:`Confidence.aggregate` over the mentions — weakest-link dominated, so one bad OCR
           read holds the node down even if nine clean ones agree.
        2. **Corroboration lift.** Each additional *distinct document* closes half the remaining
           distance to the ceiling. Two documents agreeing is meaningfully stronger than one; the
           tenth adds almost nothing, which matches how an engineer would actually update.
        3. **Match-score penalty.** A node assembled through a fuzzy tag repair is multiplied by the
           worst repair score, so ``P-l0l → P-101`` at 88 % never presents as certain.
        """
        if not self.mentions:
            self.merge_confidence = Confidence(
                value=0.0, rationale="No mentions supported this entity", method="aggregate"
            )
            return self.merge_confidence

        base = Confidence.aggregate([m.confidence for m in self.mentions])
        documents = len(self.document_ids)
        lift = (1.0 - 0.5 ** max(0, documents - 1)) * _CORROBORATION_CEILING
        corroborated = base.value + (1.0 - base.value) * lift
        value = max(0.0, min(1.0, corroborated * self.match_score))

        if self.match_score < 1.0:
            rationale = (
                f"{len(self.mentions)} mention(s) across {documents} document(s); "
                f"tag repaired by fuzzy match at {self.match_score:.2f} — verify before acting"
            )
        elif documents > 1:
            rationale = (
                f"{len(self.mentions)} mention(s) corroborated across {documents} documents "
                f"({', '.join(self.document_ids[:3])}{'…' if documents > 3 else ''})"
            )
        else:
            rationale = f"{len(self.mentions)} mention(s) in a single document; no corroboration"

        self.merge_confidence = Confidence(value=round(value, 4), rationale=rationale, method="aggregate")
        return self.merge_confidence

    def to_entity(self) -> ExtractedEntity:
        """Project into the :class:`ExtractedEntity` the graph store writes.

        Provenance rides in ``attributes`` because the model has no repeated-source field; the
        graph node therefore carries the full union of sources, which is what makes
        "five documents, one node" a checkable claim rather than a slogan.
        """
        strongest = max(self.mentions, key=lambda m: m.confidence.value, default=None)
        attributes: dict[str, object] = dict(self.attributes)
        attributes.update(
            {
                "source_documents": self.document_ids,
                "source_chunks": self.chunk_ids,
                "aliases": sorted(self.aliases),
                "mention_count": len(self.mentions),
                "mention_locations": [m.location() for m in self.mentions[:12]],
                "merge_confidence": self.merge_confidence.value,
                "merge_rationale": self.merge_confidence.rationale,
                "match_score": self.match_score,
                "key": self.key,
            }
        )
        return ExtractedEntity(
            entity_id=strongest.entity_id if strongest else self.key,
            type=self.type,
            name=self.display,
            canonical_name=self.canonical_name,
            normalized_value=self.canonical_name if self.type is EntityType.EQUIPMENT else None,
            confidence=self.merge_confidence,
            document_id=strongest.document_id if strongest else None,
            chunk_id=strongest.chunk_id if strongest else None,
            page=strongest.page if strongest else None,
            char_start=strongest.char_start if strongest else None,
            char_end=strongest.char_end if strongest else None,
            alternatives=list(self.alternatives),
            attributes=attributes,
        )


@dataclass(slots=True)
class LinkResult:
    """What one :meth:`EntityResolver.resolve_batch` call produced."""

    resolved: dict[str, ResolvedEntity] = field(default_factory=dict)
    #: Original ``ExtractedEntity.key`` → canonical key. Relationships are rewritten through this.
    key_map: dict[str, str] = field(default_factory=dict)
    merges: int = 0
    repaired_tags: int = 0
    new_tags: list[str] = field(default_factory=list)

    @property
    def entities(self) -> list[ExtractedEntity]:
        """The merged entities, ready for ``GraphStore.upsert_entities``."""
        return [resolved.to_entity() for resolved in self.resolved.values()]

    def summary(self) -> dict[str, int]:
        cross_document = sum(1 for r in self.resolved.values() if len(r.document_ids) > 1)
        return {
            "nodes": len(self.resolved),
            "merges": self.merges,
            "repaired_tags": self.repaired_tags,
            "new_tags": len(self.new_tags),
            "cross_document_nodes": cross_document,
        }


# ======================================================================================
# The resolver
# ======================================================================================


class EntityResolver:
    """Merges extracted entities into graph nodes and keeps the alias table current.

    Stateful and long-lived: it is owned by the agent, seeded from the graph at startup, and grows
    as documents arrive. That state is what makes cross-*document* linking work — resolution of the
    fifth document benefits from everything learned in the first four.
    """

    __slots__ = ("_aliases", "_registry", "_settings")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._registry = EquipmentRegistry(settings)
        self._aliases = AliasTable()

    @property
    def registry(self) -> EquipmentRegistry:
        return self._registry

    @property
    def aliases(self) -> AliasTable:
        return self._aliases

    def seed_registry(self, tags: Iterable[str]) -> int:
        """Load known plant tags (from ``GraphStore.list_equipment``) into the registry."""
        added = self._registry.extend(tags)
        for tag in self._registry.tags:
            self._aliases.add(tag, entity_key(EntityType.EQUIPMENT, tag))
        return added

    # ---------------------------------------------------------------- single-entity resolution

    def canonicalise(self, entity: ExtractedEntity) -> tuple[str, str, float, list[str]]:
        """Resolve one entity to ``(key, canonical_name, match_score, alternatives)``.

        Equipment goes through the tag grammar and the registry. Everything else is resolved
        through the alias table first (so a known surface form links to the node it already named)
        and falls back to its own normalised surface form.
        """
        surface = (entity.canonical_name or entity.name).strip()

        if entity.type is EntityType.EQUIPMENT:
            tag, score, alternatives = self._registry.resolve(surface)
            if tag is None:
                # Not a tag at all — a named asset like "Boiler Feed Pump". Fall through to the
                # alias table so it can still merge with whatever node first claimed that name.
                bound = self._aliases.lookup(surface)
                if bound is not None:
                    return bound, display_name(bound), 1.0, list(entity.alternatives)
                normalised = AliasTable.normalise(surface)
                return entity_key(EntityType.EQUIPMENT, normalised), normalised, 1.0, list(entity.alternatives)
            merged_alternatives = list(dict.fromkeys([*entity.alternatives, *alternatives]))
            return entity_key(EntityType.EQUIPMENT, tag), tag, score, merged_alternatives

        bound = self._aliases.lookup(surface)
        if bound is not None and key_type(bound) is entity.type:
            return bound, display_name(bound), 1.0, list(entity.alternatives)

        normalised = AliasTable.normalise(surface)
        return entity_key(entity.type, normalised), normalised, 1.0, list(entity.alternatives)

    # ---------------------------------------------------------------- batch resolution

    def resolve_batch(self, entities: Sequence[ExtractedEntity]) -> LinkResult:
        """Merge a batch of extracted entities into graph nodes.

        Args:
            entities: Raw extractions from one document (or several).

        Returns:
            A :class:`LinkResult` whose ``entities`` are ready to upsert and whose ``key_map``
            must be applied to the document's relationships before they are written.
        """
        result = LinkResult()
        if not entities:
            return result

        for entity in entities:
            key, canonical, score, alternatives = self.canonicalise(entity)
            result.key_map[entity.key] = key

            existing = result.resolved.get(key)
            if existing is None:
                existing = ResolvedEntity(
                    key=key,
                    type=entity.type,
                    canonical_name=canonical,
                    display=canonical,
                    match_score=score,
                    alternatives=list(alternatives),
                )
                result.resolved[key] = existing
            else:
                result.merges += 1
                existing.match_score = min(existing.match_score, score)
                for alternative in alternatives:
                    if alternative not in existing.alternatives:
                        existing.alternatives.append(alternative)

            if score < 1.0:
                result.repaired_tags += 1

            existing.mentions.append(
                EntityMention(
                    surface=entity.name,
                    document_id=entity.document_id,
                    chunk_id=entity.chunk_id,
                    page=entity.page,
                    char_start=entity.char_start,
                    char_end=entity.char_end,
                    confidence=entity.confidence,
                    entity_id=entity.entity_id,
                )
            )
            existing.aliases.add(AliasTable.normalise(entity.name))
            existing.aliases.add(AliasTable.normalise(canonical))
            self._merge_attributes(existing, entity)

            if entity.type is EntityType.EQUIPMENT and self._registry.add(canonical):
                result.new_tags.append(canonical)

        for resolved in result.resolved.values():
            resolved.compute_confidence()
            for alias in resolved.aliases:
                self._aliases.add(alias, resolved.key)

        logger.info("entities resolved", extra=result.summary())
        return result

    @staticmethod
    def _merge_attributes(target: ResolvedEntity, entity: ExtractedEntity) -> None:
        """Union attribute dictionaries, recording rather than hiding disagreements.

        First value wins for a given attribute; a conflicting later value is appended to
        ``attribute_conflicts`` so a reviewer can see that two documents disagree about, say, the
        manufacturer, instead of one silently overwriting the other.
        """
        for name, value in entity.attributes.items():
            if name not in target.attributes:
                target.attributes[name] = value
                continue
            if target.attributes[name] == value:
                continue
            conflicts = target.attributes.setdefault("attribute_conflicts", {})
            if isinstance(conflicts, dict):
                bucket = conflicts.setdefault(name, [])
                if isinstance(bucket, list) and value not in bucket:
                    bucket.append(value)

    # ---------------------------------------------------------------- relationships

    def remap_relationships(
        self,
        relationships: Sequence[ExtractedRelationship],
        key_map: Mapping[str, str],
    ) -> list[ExtractedRelationship]:
        """Rewrite relationship endpoints through ``key_map`` and deduplicate.

        Without this step a merge is half-done: the nodes collapse but the edges still point at the
        pre-merge keys, so ``P-101``'s connectivity ends up split across two dangling identities.

        Self-loops created by a merge (both endpoints resolved to the same node) are dropped —
        ``P-101 CONNECTED_TO P-101`` is an artefact, not a fact. Duplicates are collapsed keeping
        the highest confidence, with evidence text unioned so provenance survives here too.
        """
        merged: dict[tuple[str, str, str], ExtractedRelationship] = {}
        dropped_self_loops = 0

        for relationship in relationships:
            source = key_map.get(relationship.source_key, relationship.source_key)
            target = key_map.get(relationship.target_key, relationship.target_key)
            if source == target:
                dropped_self_loops += 1
                continue

            identity = (relationship.type.value, source, target)
            candidate = relationship.model_copy(update={"source_key": source, "target_key": target})
            existing = merged.get(identity)
            if existing is None:
                merged[identity] = candidate
                continue

            evidence = existing.evidence_text
            if candidate.evidence_text and candidate.evidence_text not in evidence:
                joined = f"{evidence} | {candidate.evidence_text}" if evidence else candidate.evidence_text
                evidence = joined[:_EVIDENCE_MAX_CHARS]
            properties = {**candidate.properties, **existing.properties}
            winner = existing if existing.confidence.value >= candidate.confidence.value else candidate
            merged[identity] = winner.model_copy(
                update={
                    "source_key": source,
                    "target_key": target,
                    "evidence_text": evidence,
                    "properties": properties,
                }
            )

        if dropped_self_loops:
            logger.debug(
                "dropped self-loops created by entity merge",
                extra={"dropped": dropped_self_loops, "kept": len(merged)},
            )
        return list(merged.values())

    # ---------------------------------------------------------------- query-time resolution

    def resolve_text(self, text: str) -> list[str]:
        """Extract and resolve the entity keys a free-text string is talking about.

        Two passes, unioned in a stable order:

        1. The tag grammar, then the registry — so ``"why did P 101 trip"`` finds ``P-101`` even
           though the writer omitted the hyphen.
        2. The alias table — so ``"the boiler feed pump"`` finds the same node once any document
           has taught INDRA that name.

        Returns:
            Node keys, tags first (they are the strongest signal), then alias hits.
        """
        keys: list[str] = []
        seen: set[str] = set()

        for tag in find_tag_candidates(text):
            resolved, score, _ = self._registry.resolve(tag)
            chosen = resolved or tag
            key = entity_key(EntityType.EQUIPMENT, chosen)
            if key not in seen:
                seen.add(key)
                keys.append(key)
                if score < 1.0:
                    logger.debug(
                        "query tag resolved by fuzzy match",
                        extra={"raw": tag, "resolved": chosen, "score": score},
                    )

        for key in self._aliases.scan(text):
            if key not in seen:
                seen.add(key)
                keys.append(key)

        return keys


__all__ = [
    "TAG_PREFIXES",
    "TAG_STOPWORDS",
    "AliasTable",
    "EntityMention",
    "EntityResolver",
    "EquipmentRegistry",
    "LinkResult",
    "ResolvedEntity",
    "canonical_tag",
    "display_name",
    "entity_key",
    "find_tag_candidates",
    "key_type",
    "normalise_unicode",
]
