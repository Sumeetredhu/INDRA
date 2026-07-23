"""Regulatory requirement catalogue: load, validate, and index structured obligations.

The Compliance Agent must be useful *before* anyone uploads a regulation PDF, so the shipped
``regulations/*.yaml`` files carry a seeded requirement set — one file per entry in
``settings.regulations``, named after the slug of that entry. Uploaded regulation documents are
parsed by :mod:`.parser` into exactly the same shape and merged into the same catalogue.

Two models live here:

* :class:`indra.core.models.RegulatoryRequirement` — the shared domain object. Never redefined.
* :class:`RequirementSpec` — an agent-local *wrapper* that carries the extra facts gap detection
  needs in order to be mechanically checkable: which evidence fields must be present, who owns the
  remediation, how severe a breach is, and which document revision is current. These are checking
  metadata, not vocabulary, which is why they live in the agent rather than in ``core``.

Requirement ids are content-addressed (``regulation|clause|obligation``) rather than random, so the
same requirement carries the same id across processes, restarts, and machines. A compliance finding
that changes id every run cannot be tracked to closure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable, Iterator, Literal, Mapping, Sequence

import yaml
from pydantic import Field, ValidationError, field_validator

from indra.core.config import Settings, get_settings
from indra.core.exceptions import ComplianceError
from indra.core.ids import content_id
from indra.core.logging import get_logger
from indra.core.models import (
    DocumentType,
    Equipment,
    IndraModel,
    RegulatoryRequirement,
    Severity,
    SourceRef,
)

logger = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Implementation constants
#
# Product tunables live in ``indra.core.config``. The values below are structural facts about the
# on-disk catalogue format, not knobs an operator would ever turn.
# --------------------------------------------------------------------------------------

#: Directory holding the seeded requirement files, resolved relative to this package.
SEED_REGULATION_DIR: Final[Path] = Path(__file__).resolve().parent / "regulations"

#: Pseudo-tag used for obligations that bind the installation rather than a single asset
#: (``applies_to_types: ["*"]``). Gap detection evaluates these once against this scope.
PLANT_SCOPE_TAG: Final[str] = "PLANT"

#: ``applies_to_types`` entry meaning "every asset in scope".
WILDCARD_TYPE: Final[str] = "*"

#: Prefix for the synthetic document id attached to a seeded requirement's :class:`SourceRef`.
SEED_DOCUMENT_PREFIX: Final[str] = "seed"


def slugify(value: str) -> str:
    """Normalise a name to ``snake_case`` for filenames and equality comparison.

    ``"OISD-STD-118"`` → ``"oisd_std_118"``, ``"Factory Act 1948"`` → ``"factory_act_1948"``.
    """
    out: list[str] = []
    previous_sep = False
    for char in value.strip().lower():
        if char.isalnum():
            out.append(char)
            previous_sep = False
        elif not previous_sep:
            out.append("_")
            previous_sep = True
    return "".join(out).strip("_")


def normalise_field_key(value: str) -> str:
    """Normalise an evidence field name so ``"Test Pressure (bar)"`` matches ``test_pressure``."""
    return slugify(value)


def requirement_key(regulation: str, clause: str, obligation: str) -> str:
    """Stable, content-addressed requirement id.

    The same clause always produces the same id, which is what lets a gap be tracked from detection
    to closure across restarts.
    """
    return content_id(f"{slugify(regulation)}|{slugify(clause)}|{slugify(obligation)}", kind="entity")


# ======================================================================================
# On-disk schema
# ======================================================================================


class RawRequirement(IndraModel):
    """One requirement as written in a ``regulations/*.yaml`` file.

    ``extra="forbid"`` is inherited from :class:`IndraModel` on purpose: a typo in a key silently
    dropping a frequency would turn a real obligation into an unchecked one.
    """

    clause: str = Field(min_length=1)
    obligation: str = Field(min_length=1)
    text: str = Field(min_length=1)
    frequency_days: int | None = Field(default=None, ge=1)
    applies_to_types: list[str] = Field(default_factory=list)
    applies_to_tags: list[str] = Field(default_factory=list)
    evidence_types: list[DocumentType] = Field(default_factory=list)
    required_evidence_fields: list[str] = Field(default_factory=list)
    penalty: str | None = None
    owner_role: str = "Plant Compliance Officer"
    severity: Severity = Severity.HIGH
    grace_days: int = Field(default=0, ge=0)
    current_revision: str | None = None
    superseded_revisions: list[str] = Field(default_factory=list)
    remediation: str = ""
    remediation_minutes: int | None = Field(default=None, ge=1)

    @field_validator("text", "obligation", "remediation", "penalty")
    @classmethod
    def _collapse_whitespace(cls, value: str | None) -> str | None:
        return " ".join(value.split()) if value else value

    @field_validator("required_evidence_fields")
    @classmethod
    def _normalise_fields(cls, value: list[str]) -> list[str]:
        return [normalise_field_key(item) for item in value if item.strip()]

    @field_validator("applies_to_types")
    @classmethod
    def _normalise_types(cls, value: list[str]) -> list[str]:
        return [item if item == WILDCARD_TYPE else slugify(item) for item in value if item.strip()]

    @field_validator("applies_to_tags")
    @classmethod
    def _normalise_tags(cls, value: list[str]) -> list[str]:
        return [item.strip().upper() for item in value if item.strip()]


class RegulationFile(IndraModel):
    """A whole ``regulations/*.yaml`` file."""

    regulation: str = Field(min_length=1)
    title: str = ""
    authority: str = ""
    version: str = ""
    disclaimer: str = ""
    requirements: list[RawRequirement] = Field(min_length=1)

    @field_validator("disclaimer", "title")
    @classmethod
    def _collapse(cls, value: str) -> str:
        return " ".join(value.split())


# ======================================================================================
# The checkable requirement
# ======================================================================================


class RequirementSpec(IndraModel):
    """A :class:`RegulatoryRequirement` plus everything needed to check it mechanically.

    Wrapping rather than subclassing keeps ``ComplianceGap.requirement`` exactly the shared domain
    model while giving gap detection the checking metadata it needs.
    """

    requirement: RegulatoryRequirement
    regulation_title: str = ""
    authority: str = ""
    regulation_version: str = ""
    disclaimer: str = ""
    provenance: Literal["seed", "parsed"] = "seed"
    required_evidence_fields: list[str] = Field(default_factory=list)
    owner_role: str = "Plant Compliance Officer"
    breach_severity: Severity = Severity.HIGH
    grace_days: int = Field(default=0, ge=0)
    current_revision: str | None = None
    superseded_revisions: list[str] = Field(default_factory=list)
    remediation: str = ""
    remediation_minutes: int | None = Field(default=None, ge=1)

    # -- convenience projections ---------------------------------------------------
    @property
    def requirement_id(self) -> str:
        return self.requirement.requirement_id

    @property
    def regulation(self) -> str:
        return self.requirement.regulation

    @property
    def clause(self) -> str:
        return self.requirement.clause

    @property
    def obligation(self) -> str:
        return self.requirement.obligation

    @property
    def frequency_days(self) -> int | None:
        return self.requirement.frequency_days

    @property
    def evidence_types(self) -> list[DocumentType]:
        return list(self.requirement.evidence_types)

    @property
    def is_plant_wide(self) -> bool:
        """True when the obligation binds the installation rather than a single asset."""
        return WILDCARD_TYPE in self.requirement.applies_to_types

    @property
    def citation(self) -> str:
        return f"{self.regulation} {self.clause}"

    def applies_to(self, equipment: Equipment) -> bool:
        """Deterministic applicability test — no fuzzy matching, no model in the loop.

        An asset is bound by a requirement when its tag is named explicitly, when the requirement is
        plant-wide, or when one of the requirement's equipment-type patterns generalises the asset's
        own type (``pressure_vessel`` binds ``horizontal_pressure_vessel``, but ``fire_water_pump``
        does not bind every ``pump``).
        """
        tag = equipment.tag.strip().upper()
        if tag in {t.upper() for t in self.requirement.applies_to_tags}:
            return True
        if self.is_plant_wide:
            return tag == PLANT_SCOPE_TAG
        if tag == PLANT_SCOPE_TAG:
            return False
        return type_matches(equipment.equipment_type, self.requirement.applies_to_types)


def type_matches(equipment_type: str, patterns: Sequence[str]) -> bool:
    """Return True when any pattern generalises ``equipment_type``.

    Matching is token-subset in one direction only: every token of the pattern must appear in the
    asset's type. That makes ``pump`` match ``centrifugal_pump`` while stopping ``fire_water_pump``
    from claiming every pump on the plant — an over-broad match manufactures false gaps, and a false
    gap in front of a regulator is worse than a missed one.
    """
    actual = slugify(equipment_type)
    if not actual or actual == "unknown":
        return False
    actual_tokens = set(actual.split("_"))
    for raw in patterns:
        pattern = raw if raw == WILDCARD_TYPE else slugify(raw)
        if not pattern:
            continue
        if pattern == WILDCARD_TYPE or pattern == actual:
            return True
        if set(pattern.split("_")) <= actual_tokens:
            return True
    return False


def plant_scope_equipment() -> Equipment:
    """The synthetic asset that carries plant-wide obligations."""
    return Equipment(
        tag=PLANT_SCOPE_TAG,
        name="Installation-wide scope",
        equipment_type="installation",
        location="Whole installation",
    )


# ======================================================================================
# Catalogue
# ======================================================================================


class RequirementCatalogue:
    """An indexed, deduplicated set of :class:`RequirementSpec`.

    Deduplication is by ``requirement_id``; a spec parsed from an uploaded regulation document
    replaces the seeded spec for the same clause, because the uploaded document is the authority.
    """

    __slots__ = ("_specs", "_by_regulation")

    def __init__(self, specs: Iterable[RequirementSpec] = ()) -> None:
        self._specs: dict[str, RequirementSpec] = {}
        self._by_regulation: dict[str, list[str]] = {}
        self.add(specs)

    # -- mutation ------------------------------------------------------------------
    def add(self, specs: Iterable[RequirementSpec]) -> int:
        """Merge ``specs`` in. Returns the number of new or replaced entries."""
        written = 0
        for spec in specs:
            key = spec.requirement_id
            existing = self._specs.get(key)
            if existing is not None and existing.provenance == "parsed" and spec.provenance == "seed":
                continue  # a parsed requirement outranks the seeded baseline
            self._specs[key] = spec
            written += 1
        self._reindex()
        return written

    def _reindex(self) -> None:
        index: dict[str, list[str]] = {}
        for key, spec in self._specs.items():
            index.setdefault(slugify(spec.regulation), []).append(key)
        self._by_regulation = index

    # -- reads ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[RequirementSpec]:
        return iter(self.all())

    def all(self) -> list[RequirementSpec]:
        """Every spec, ordered deterministically by regulation then clause."""
        return sorted(self._specs.values(), key=lambda s: (s.regulation, s.clause, s.obligation))

    def regulations(self) -> list[str]:
        """Distinct regulation names present in the catalogue."""
        return sorted({spec.regulation for spec in self._specs.values()})

    def get(self, requirement_id: str) -> RequirementSpec | None:
        return self._specs.get(requirement_id)

    def for_regulations(self, names: Sequence[str] | None) -> list[RequirementSpec]:
        """Filter by regulation name or title. ``None`` means every regulation."""
        if not names:
            return self.all()
        wanted = {slugify(name) for name in names}
        return [
            spec for spec in self.all()
            if slugify(spec.regulation) in wanted or slugify(spec.regulation_title) in wanted
        ]

    def for_equipment(
        self,
        equipment: Equipment,
        *,
        regulations: Sequence[str] | None = None,
    ) -> list[RequirementSpec]:
        """Every requirement that binds ``equipment``, ordered deterministically."""
        return [spec for spec in self.for_regulations(regulations) if spec.applies_to(equipment)]

    def plant_wide(self, *, regulations: Sequence[str] | None = None) -> list[RequirementSpec]:
        return [spec for spec in self.for_regulations(regulations) if spec.is_plant_wide]

    def requirements(self, *, regulations: Sequence[str] | None = None) -> list[RegulatoryRequirement]:
        """Project to the shared domain model — what the API and the orchestrator hand out."""
        return [spec.requirement for spec in self.for_regulations(regulations)]

    def index_by_equipment_type(self) -> Mapping[str, list[RequirementSpec]]:
        """Requirement index keyed by equipment-type pattern, for diagnostics and the API."""
        index: dict[str, list[RequirementSpec]] = {}
        for spec in self.all():
            for pattern in spec.requirement.applies_to_types or [WILDCARD_TYPE]:
                index.setdefault(pattern, []).append(spec)
        return index

    def describe(self) -> dict[str, int]:
        """Counts per regulation, for ``/health`` and the ops panel."""
        return {spec_regulation: len(keys) for spec_regulation, keys in sorted(self._by_regulation.items())}


# ======================================================================================
# Loading
# ======================================================================================


def spec_from_raw(
    raw: RawRequirement,
    *,
    regulation: str,
    file: RegulationFile,
    provenance: Literal["seed", "parsed"] = "seed",
    source: SourceRef | None = None,
) -> RequirementSpec:
    """Build a :class:`RequirementSpec` from one parsed YAML entry."""
    requirement_id = requirement_key(regulation, raw.clause, raw.obligation)
    citation_source = source or SourceRef(
        document_id=f"{SEED_DOCUMENT_PREFIX}:{slugify(regulation)}",
        document_title=file.title or regulation,
        document_type=DocumentType.REGULATION,
        snippet=raw.text[:600],
        relevance=1.0,
        extraction_confidence=1.0,
        retrieved_via="direct",
    )
    requirement = RegulatoryRequirement(
        requirement_id=requirement_id,
        regulation=regulation,
        clause=raw.clause,
        text=raw.text,
        obligation=raw.obligation,
        frequency_days=raw.frequency_days,
        applies_to_types=list(raw.applies_to_types),
        applies_to_tags=list(raw.applies_to_tags),
        evidence_types=list(raw.evidence_types),
        penalty=raw.penalty,
        source=citation_source,
    )
    return RequirementSpec(
        requirement=requirement,
        regulation_title=file.title or regulation,
        authority=file.authority,
        regulation_version=file.version,
        disclaimer=file.disclaimer,
        provenance=provenance,
        required_evidence_fields=list(raw.required_evidence_fields),
        owner_role=raw.owner_role,
        breach_severity=raw.severity,
        grace_days=raw.grace_days,
        current_revision=raw.current_revision,
        superseded_revisions=list(raw.superseded_revisions),
        remediation=raw.remediation,
        remediation_minutes=raw.remediation_minutes,
    )


def load_regulation_file(path: Path, *, regulation: str | None = None) -> list[RequirementSpec]:
    """Read and validate one ``regulations/*.yaml``.

    Raises:
        ComplianceError: the file is unreadable, is not valid YAML, or does not satisfy the
            catalogue schema. The message names the file and the offending key.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ComplianceError(
            f"Cannot read regulation catalogue {path.name}: {exc}. "
            f"Check that {path} exists and is readable by the service account.",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ComplianceError(
            f"Regulation catalogue {path.name} is not valid YAML: {exc}. Fix the syntax and restart.",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(raw, dict):
        raise ComplianceError(
            f"Regulation catalogue {path.name} must be a YAML mapping with 'regulation' and "
            "'requirements' keys.",
            context={"path": str(path)},
        )
    try:
        file_model = RegulationFile.model_validate(raw)
    except ValidationError as exc:
        raise ComplianceError(
            f"Regulation catalogue {path.name} failed schema validation: {exc.errors()[0].get('loc')} "
            f"-> {exc.errors()[0].get('msg')}. Every requirement needs clause, obligation and text.",
            context={"path": str(path), "errors": exc.error_count()},
            cause=exc,
        ) from exc

    canonical = regulation or file_model.regulation
    if slugify(file_model.regulation) != slugify(canonical):
        logger.warning(
            "regulation file declares a different name than settings.regulations; using the settings name",
            extra={"path": str(path), "declared": file_model.regulation, "configured": canonical},
        )
    return [spec_from_raw(raw_req, regulation=canonical, file=file_model) for raw_req in file_model.requirements]


def load_seed_catalogue(
    settings: Settings | None = None,
    *,
    directory: Path | None = None,
    strict: bool = False,
) -> RequirementCatalogue:
    """Load the seeded catalogue for every regulation named in ``settings.regulations``.

    Blocking filesystem work — call it through :func:`asyncio.to_thread`.

    A regulation with no file, or with a broken file, degrades that one regulation and logs loudly;
    it never prevents startup (CLAUDE.md rule 6). ``strict=True`` re-raises instead, which is what
    the catalogue's own unit test uses to prove the shipped files are valid.
    """
    cfg = settings or get_settings()
    base = directory or SEED_REGULATION_DIR
    specs: list[RequirementSpec] = []
    missing: list[str] = []

    if not base.is_dir():
        message = (
            f"Seed regulation directory {base} is missing. Compliance gap detection will report "
            "nothing until it is restored or a regulation document is uploaded."
        )
        if strict:
            raise ComplianceError(message, context={"directory": str(base)})
        logger.error("seed regulation directory missing", extra={"directory": str(base)})
        return RequirementCatalogue()

    for name in cfg.regulations:
        path = base / f"{slugify(name)}.yaml"
        if not path.exists():
            missing.append(name)
            continue
        try:
            specs.extend(load_regulation_file(path, regulation=name))
        except ComplianceError as exc:
            if strict:
                raise
            logger.error(
                "regulation catalogue rejected; that regulation will not be checked",
                extra={"regulation": name, "path": str(path), "detail": exc.message},
            )

    if missing:
        message = f"No seeded requirement file for: {', '.join(missing)}"
        if strict:
            raise ComplianceError(
                f"{message}. Expected {[f'{slugify(n)}.yaml' for n in missing]} in {base}.",
                context={"missing": missing, "directory": str(base)},
            )
        logger.warning(
            "regulations configured without a seeded requirement file",
            extra={"missing": missing, "directory": str(base)},
        )

    catalogue = RequirementCatalogue(specs)
    logger.info(
        "loaded seeded regulatory requirements",
        extra={"requirements": len(catalogue), "regulations": catalogue.describe()},
    )
    return catalogue


__all__ = [
    "PLANT_SCOPE_TAG",
    "SEED_DOCUMENT_PREFIX",
    "SEED_REGULATION_DIR",
    "WILDCARD_TYPE",
    "RawRequirement",
    "RegulationFile",
    "RequirementCatalogue",
    "RequirementSpec",
    "load_regulation_file",
    "load_seed_catalogue",
    "normalise_field_key",
    "plant_scope_equipment",
    "requirement_key",
    "slugify",
    "spec_from_raw",
    "type_matches",
]
