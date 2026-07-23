"""The canonical fact sheet for INDRA's synthetic plant corpus.

Every number, name and date that appears in more than one generated document is defined **once**,
here. The generators import these constants; nothing in ``generate_demo_data.py`` types a threshold
or a date literal. That is not tidiness for its own sake — the entire demo is a claim about
*reasoning across documents*, and that claim collapses the moment the 78% in the work order and the
78% the inspection compares against the 2022 failure drift apart.

The load-bearing links, in the order the demo walks them:

============================================  ===================================================
Fact                                          Documents that must agree
============================================  ===================================================
Bearing replacement at 85% wear (OEM limit)   OEM manual, SOP, work order, inspection, alert logic
Bearing wear measured at 78%                  Work order (typed + handwritten), maintenance log
"wear pattern similar to 2022 failure"        Inspection report -> incident report -> RCA
2022 seizure: 14 h downtime, Rs 25,00,000     Incident report, RCA, maintenance log
Root cause: lubrication failure (LP-101A)     RCA, incident report, maintenance log
Alarm bypassed twice, night of 2024-06-14     Shift log, work order (references the shift log)
Rajesh Kumar, 23 years, retires March 2027    Retirement e-mail, work order, maintenance log
P-101 -> V-201 / E-301 process connectivity   P&ID drawing, OEM manual, incident report
Monthly pressure-vessel inspection (41(b))    Factory Act extract, maintenance log (V-201 rows)
============================================  ===================================================

**Reference date.** The corpus is anchored at :data:`REFERENCE_DATE` (2024-06-30), not at
wall-clock now. Every lookback window in the corpus is tuned to that anchor: the shift log is 16
days old (``shift_log_lookback_days`` = 30), the work order 12 days (``maintenance_lookback_days``
= 90), the quarterly inspection 107 days (``inspection_lookback_days`` = 180). Consumers that
window on evidence recency must anchor on this date — it is published in ``manifest.json`` as
``reference_date`` — or on the newest document date, never on ``datetime.now()``.

**Two retirements, deliberately.** ``retirement_horizon_days`` is 720. Rajesh Kumar's March 2027
retirement is inside that horizon for a demo run in 2026 but outside it for one anchored at the
2024 reference date, so the corpus also carries a second, nearer superannuation (S. Ramaswamy,
2024-11-30) in the same HR notice. The knowledge-cliff and ``expertise_loss`` signals therefore
fire under either anchoring, which is what stops the demo from depending on the calendar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Final, Literal, Mapping, Sequence

# ======================================================================================
# Plant identity
# ======================================================================================

PLANT_NAME: Final[str] = "Bharat Vindhya Petrochemicals Ltd."
PLANT_SHORT: Final[str] = "BVPL"
PLANT_UNIT: Final[str] = "Unit-2 Utilities Block"
PLANT_LOCATION: Final[str] = "Raigad, Maharashtra, India"
PLANT_ADDRESS: Final[str] = f"{PLANT_NAME} — {PLANT_UNIT}, {PLANT_LOCATION}"

#: Every relative window in the corpus is measured from this date. Never ``date.today()``.
REFERENCE_DATE: Final[date] = date(2024, 6, 30)

#: Fixed timestamp stamped into container metadata (PDF creation date, xlsx zip entries) so that
#: regenerating the corpus produces byte-identical files and D6 idempotency holds across rehearsals.
FIXED_TIMESTAMP: Final[tuple[int, int, int, int, int, int]] = (2024, 6, 30, 0, 0, 0)


# ======================================================================================
# Typed fact records
# ======================================================================================


@dataclass(frozen=True, slots=True)
class EquipmentFact:
    """One physical asset, as the corpus describes it."""

    tag: str
    name: str
    equipment_type: str
    criticality: Literal["A", "B", "C"]
    location: str
    unit: str
    installed_on: date
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    specifications: Mapping[str, str] = field(default_factory=dict)
    oem_thresholds: Mapping[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True, slots=True)
class PersonFact:
    """One plant person. ``retirement_on`` is what makes the knowledge cliff computable."""

    name: str
    employee_id: str
    role: str
    department: str
    years_experience: float
    expertise_tags: tuple[str, ...]
    retirement_on: date | None = None
    contact: str | None = None
    #: Documents in this corpus that record this person's knowledge. The denominator of the
    #: knowledge-cliff score: a 23-year expert with two documents is the cliff.
    documented_contributions: int = 0


@dataclass(frozen=True, slots=True)
class ReadingFact:
    """A single condition-monitoring measurement."""

    equipment_tag: str
    parameter: str
    value: float
    unit: str
    measured_on: date
    source_document: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class MaintenanceRowFact:
    """One row of the maintenance-history spreadsheet."""

    record_id: str
    equipment_tag: str
    record_type: Literal["work_order", "inspection", "preventive", "calibration", "breakdown"]
    performed_on: date
    performed_by: str
    findings: str
    action_taken: str
    downtime_hours: float
    cost_inr: float
    status: Literal["open", "closed", "deferred"]
    reference_document: str


@dataclass(frozen=True, slots=True)
class ProcedureStepFact:
    """One numbered step of the bearing-replacement SOP."""

    number: int
    text: str
    minutes: int
    hold_point: bool = False


@dataclass(frozen=True, slots=True)
class DocumentSpec:
    """What one generated file is, for the manifest and for the demo checks."""

    filename: str
    title: str
    document_type: str
    document_date: date
    role: Literal["corpus", "asset"]
    equipment_tags: tuple[str, ...]
    people: tuple[str, ...] = ()
    key_facts: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class CrossLink:
    """A claim that only resolves when two or more documents are read together.

    ``scripts/run_demo_check.py`` asserts these; they are the demo's thesis in machine-checkable
    form. If one of these stops resolving, the demo has silently broken.
    """

    link_id: str
    claim: str
    documents: tuple[str, ...]
    probe_terms: tuple[str, ...]


# ======================================================================================
# Equipment registry
# ======================================================================================

OEM_BEARING_WEAR_LIMIT_PCT: Final[float] = 85.0
"""Sulzer's stated bearing replacement limit. The single most-cited number in the corpus."""

MEASURED_BEARING_WEAR_PCT: Final[float] = 78.0
"""Wear measured under WO-2024-0342 — typed in the work order and written in the margin by hand."""

OEM_VIBRATION_LIMIT_MM_S: Final[float] = 7.1
OEM_BEARING_TEMP_LIMIT_C: Final[float] = 95.0
OEM_LUBE_PRESSURE_MIN_BAR: Final[float] = 1.4

P101: Final[EquipmentFact] = EquipmentFact(
    tag="P-101",
    name="Boiler Feed Water Pump P-101",
    equipment_type="centrifugal_pump",
    criticality="A",
    location="Unit-2 Utilities, Pump Bay B, Elev. +3.5 m",
    unit="Unit-2 Utilities",
    installed_on=date(2016, 4, 12),
    manufacturer="Sulzer",
    model="CP 150-400",
    serial_number="SLZ-9174-2016",
    specifications={
        "Service": "Boiler feed water, deaerator to pre-heater",
        "Rated flow": "148 m3/h",
        "Rated head": "395 m",
        "Rated speed": "2965 rpm",
        "Driver": "250 kW, 6.6 kV squirrel-cage induction motor",
        "Casing": "Radially split, 12% Cr steel",
        "Impeller stages": "4",
        "DE bearing": "SKF 6316 C3 deep-groove ball bearing",
        "NDE bearing": "SKF 7314 BECBM angular-contact ball bearing",
        "Lubrication": "Forced feed, ISO VG 46 turbine oil, 18 l/min at 2.1 bar",
        "Suction source": "V-201 deaerator storage vessel",
        "Discharge destination": "E-301 feed water pre-heater",
        "Mechanical seal": "Cartridge, API Plan 11 flush",
    },
    oem_thresholds={
        "bearing_wear_pct": OEM_BEARING_WEAR_LIMIT_PCT,
        "vibration_mm_s_rms": OEM_VIBRATION_LIMIT_MM_S,
        "bearing_temp_c": OEM_BEARING_TEMP_LIMIT_C,
        "lube_oil_pressure_bar_min": OEM_LUBE_PRESSURE_MIN_BAR,
    },
    notes="Duty pump of the P-101 / P-102 duty-standby pair. Loss of both stops Unit-2 steam raising.",
)

P102: Final[EquipmentFact] = EquipmentFact(
    tag="P-102",
    name="Boiler Feed Water Pump P-102 (standby)",
    equipment_type="centrifugal_pump",
    criticality="A",
    location="Unit-2 Utilities, Pump Bay B, Elev. +3.5 m",
    unit="Unit-2 Utilities",
    installed_on=date(2016, 4, 12),
    manufacturer="Sulzer",
    model="CP 150-400",
    serial_number="SLZ-9175-2016",
    specifications={"Service": "Boiler feed water, standby to P-101", "Rated flow": "148 m3/h"},
    oem_thresholds={
        "bearing_wear_pct": OEM_BEARING_WEAR_LIMIT_PCT,
        "vibration_mm_s_rms": OEM_VIBRATION_LIMIT_MM_S,
    },
    notes="Identical machine to P-101. Suffered the same NDE bearing failure mode in November 2023.",
)

P105: Final[EquipmentFact] = EquipmentFact(
    tag="P-105",
    name="Boiler Feed Water Pump P-105 (Unit-3)",
    equipment_type="centrifugal_pump",
    criticality="B",
    location="Unit-3 Utilities, Pump Bay A",
    unit="Unit-3 Utilities",
    installed_on=date(2018, 9, 3),
    manufacturer="Sulzer",
    model="CP 150-400",
    serial_number="SLZ-11208-2018",
    specifications={"Service": "Boiler feed water, Unit-3"},
    oem_thresholds={"bearing_wear_pct": OEM_BEARING_WEAR_LIMIT_PCT},
    notes="Third machine of the same class; NDE bearing wear trending high since January 2024.",
)

V201: Final[EquipmentFact] = EquipmentFact(
    tag="V-201",
    name="Deaerator Storage Vessel V-201",
    equipment_type="pressure_vessel",
    criticality="A",
    location="Unit-2 Utilities, Elev. +14.0 m",
    unit="Unit-2 Utilities",
    installed_on=date(2016, 2, 28),
    manufacturer="Godrej Process Equipment",
    model="DA-24-8.5",
    serial_number="GPE-DA-4471",
    specifications={
        "Type": "Horizontal storage vessel with spray-tray deaerating head",
        "Design pressure": "8.5 barg",
        "Design temperature": "180 degC",
        "Capacity": "24 m3",
        "Material of construction": "SA-516 Gr.70, 12 mm shell",
        "Relief device": "PSV-2011, set 8.5 barg",
        "Statutory class": "Pressure vessel under Factory Act 1948 Section 41(b)",
    },
    oem_thresholds={"operating_pressure_bar_max": 8.5, "dissolved_o2_ppb_max": 7.0},
    notes="Suction source for P-101 and P-102. Statutory monthly inspection applies.",
)

E301: Final[EquipmentFact] = EquipmentFact(
    tag="E-301",
    name="Feed Water Pre-Heater E-301",
    equipment_type="heat_exchanger",
    criticality="B",
    location="Unit-2 Utilities, Elev. +6.0 m",
    unit="Unit-2 Utilities",
    installed_on=date(2016, 3, 20),
    manufacturer="Alfa Laval India",
    model="TEMA BEM 500-4000",
    serial_number="AL-BEM-2231",
    specifications={
        "Type": "Shell-and-tube, single shell pass, two tube passes",
        "Duty": "1.8 MW",
        "Shell design pressure": "12 barg",
        "Tube design pressure": "45 barg",
        "Tube bundle": "196 tubes, 19.05 mm OD, SS-304L",
        "Shell side": "LP steam, 3.5 barg",
        "Tube side": "Boiler feed water from P-101 discharge",
    },
    oem_thresholds={"tube_side_dp_bar_max": 1.8},
    notes="Receives P-101 discharge. Fouling raises P-101 discharge pressure and bearing load.",
)

LP101A: Final[EquipmentFact] = EquipmentFact(
    tag="LP-101A",
    name="Auxiliary Lube Oil Pump LP-101A",
    equipment_type="gear_pump",
    criticality="B",
    location="Unit-2 Utilities, P-101 skid",
    unit="Unit-2 Utilities",
    installed_on=date(2016, 4, 12),
    manufacturer="Rotodel",
    model="RGP-18",
    serial_number="RD-18-3390",
    specifications={
        "Service": "Forced-feed lubrication for P-101 DE and NDE bearings",
        "Rated flow": "18 l/min",
        "Rated discharge pressure": "2.1 bar",
    },
    oem_thresholds={"discharge_pressure_bar_min": OEM_LUBE_PRESSURE_MIN_BAR},
    notes="Root cause of the August 2022 P-101 bearing seizure. Overhauled 2022-08-24.",
)

EQUIPMENT: Final[tuple[EquipmentFact, ...]] = (P101, P102, P105, V201, E301, LP101A)

#: Instrument and valve tags that appear on the P&ID. Kept here so the drawing, the OEM manual and
#: the shift log all name the same vibration transmitter.
INSTRUMENT_TAGS: Final[Mapping[str, str]] = {
    "VT-1011": "P-101 NDE bearing vibration transmitter (alarm 7.1 mm/s, trip 9.5 mm/s)",
    "PI-1015": "P-101 discharge pressure indicator",
    "TI-2011": "V-201 outlet temperature indicator",
    "PI-1016": "P-101 lube oil header pressure indicator (low alarm 1.4 bar)",
}
VALVE_TAGS: Final[Mapping[str, str]] = {
    "HV-1012": "P-101 suction isolation valve",
    "CV-1013": "P-101 discharge non-return valve",
    "HV-3014": "E-301 tube-side inlet isolation valve",
}


# ======================================================================================
# People
# ======================================================================================

RAJESH_KUMAR: Final[PersonFact] = PersonFact(
    name="Rajesh Kumar",
    employee_id="EMP-1187",
    role="Senior Technician — Rotating Equipment",
    department="Mechanical Maintenance",
    years_experience=23.0,
    expertise_tags=("P-101", "P-102", "LP-101A", "V-201"),
    retirement_on=date(2027, 3, 31),
    contact="r.kumar@bvpl.example",
    documented_contributions=3,
)

S_RAMASWAMY: Final[PersonFact] = PersonFact(
    name="S. Ramaswamy",
    employee_id="EMP-0904",
    role="Foreman — Boiler & Utilities Operations",
    department="Operations",
    years_experience=31.0,
    expertise_tags=("V-201", "P-101", "E-301"),
    retirement_on=date(2024, 11, 30),
    contact="s.ramaswamy@bvpl.example",
    documented_contributions=1,
)

PRIYA_SHARMA: Final[PersonFact] = PersonFact(
    name="Priya Sharma",
    employee_id="EMP-2246",
    role="Reliability Engineer",
    department="Asset Reliability",
    years_experience=8.0,
    expertise_tags=("P-101", "P-102", "P-105", "E-301"),
    contact="p.sharma@bvpl.example",
    documented_contributions=4,
)

ANIL_DESHMUKH: Final[PersonFact] = PersonFact(
    name="Anil Deshmukh",
    employee_id="EMP-1533",
    role="Shift Supervisor — B Shift",
    department="Operations",
    years_experience=15.0,
    expertise_tags=("P-101", "V-201"),
    contact="a.deshmukh@bvpl.example",
    documented_contributions=2,
)

SURESH_PATIL: Final[PersonFact] = PersonFact(
    name="Suresh Patil",
    employee_id="EMP-3120",
    role="Panel Operator — Night Shift",
    department="Operations",
    years_experience=6.0,
    expertise_tags=("P-101",),
    contact="s.patil@bvpl.example",
    documented_contributions=1,
)

MEERA_IYER: Final[PersonFact] = PersonFact(
    name="Meera Iyer",
    employee_id="EMP-0771",
    role="Head — Human Resources",
    department="Human Resources",
    years_experience=19.0,
    expertise_tags=(),
    contact="m.iyer@bvpl.example",
)

D_KRISHNAN: Final[PersonFact] = PersonFact(
    name="D. Krishnan",
    employee_id="EMP-0615",
    role="Maintenance Manager",
    department="Mechanical Maintenance",
    years_experience=21.0,
    expertise_tags=("P-101", "E-301"),
    contact="d.krishnan@bvpl.example",
    documented_contributions=2,
)

PEOPLE: Final[tuple[PersonFact, ...]] = (
    RAJESH_KUMAR, S_RAMASWAMY, PRIYA_SHARMA, ANIL_DESHMUKH, SURESH_PATIL, MEERA_IYER, D_KRISHNAN,
)


# ======================================================================================
# The 2022 failure — the event every other document points back to
# ======================================================================================

INCIDENT_ID: Final[str] = "INC-2022-0820"
INCIDENT_DATE: Final[date] = date(2022, 8, 20)
INCIDENT_FAILURE_MODE: Final[str] = "NDE bearing seizure"
INCIDENT_DOWNTIME_HOURS: Final[float] = 14.0
INCIDENT_COST_INR: Final[float] = 2_500_000.0
INCIDENT_COST_TEXT: Final[str] = "Rs 25,00,000 (Rs 25 lakh)"
"""Rendered as ``Rs`` rather than the rupee sign: the PDF base-14 fonts are WinAnsi-encoded and
U+20B9 does not survive text extraction (it comes back as ``n``), which would silently corrupt the
one number the business case rests on."""

INCIDENT_PRECURSORS: Final[tuple[str, ...]] = (
    "NDE bearing temperature rose from 68 degC to 91 degC over the six days preceding the seizure.",
    "Overall vibration at the NDE bearing trended from 4.2 mm/s to 8.9 mm/s RMS in the same window.",
    "Lube oil header pressure decayed from 2.1 bar to 1.2 bar, below the 1.4 bar low-low limit.",
    "Two VT-1011 high-vibration alarms were acknowledged on the night shift without corrective action.",
    "Bearing wear was recorded at 84% of the OEM replacement limit at the June 2022 inspection.",
)

RCA_ID: Final[str] = "RCA-2022-0825"
RCA_DATE: Final[date] = date(2022, 8, 25)
RCA_ROOT_CAUSE: Final[str] = (
    "Lubrication failure. Internal wear in auxiliary lube oil pump LP-101A allowed the oil header "
    "pressure to decay below 1.4 bar, starving the P-101 non-drive-end bearing of oil film until "
    "the rolling elements welded to the outer race."
)
RCA_CONTRIBUTING_FACTORS: Final[tuple[str, ...]] = (
    "LP-101A had no condition-based maintenance task; it was run to failure by default.",
    "Lube oil analysis had not been performed since November 2021 (18-month interval against a "
    "3-month standard).",
    "The VT-1011 high-vibration alarm was acknowledged twice on the night shift of 19 August 2022 "
    "without a corrective work order being raised.",
    "Bearing wear at 84% at the June 2022 inspection did not trigger the OEM 85% replacement rule "
    "because the trend was not plotted against the limit.",
)
RCA_FIVE_WHYS: Final[tuple[tuple[str, str], ...]] = (
    ("Why did P-101 stop?", "The NDE bearing seized and the shaft locked."),
    ("Why did the bearing seize?", "It ran without an oil film and the rolling elements welded to the race."),
    ("Why was there no oil film?", "Lube oil header pressure fell to 1.2 bar, below the 1.4 bar minimum."),
    ("Why did header pressure fall?", "LP-101A internal clearances had opened up from wear; volumetric "
                                      "efficiency dropped to roughly 55%."),
    ("Why was that wear not detected?", "LP-101A carried no condition-monitoring task and lube oil "
                                        "analysis had lapsed for 18 months."),
)
RCA_CORRECTIVE_ACTIONS: Final[tuple[tuple[str, str, date], ...]] = (
    ("Overhaul LP-101A and restore rated discharge pressure of 2.1 bar", "Rajesh Kumar", date(2022, 8, 24)),
    ("Replace P-101 NDE and DE bearings per SOP-MECH-014", "Rajesh Kumar", date(2022, 8, 26)),
    ("Reinstate quarterly lube oil analysis for P-101 and P-102", "Priya Sharma", date(2022, 9, 15)),
    ("Add lube oil header low-pressure alarm PI-1016 at 1.4 bar to the DCS alarm list", "D. Krishnan",
     date(2022, 9, 30)),
    ("Plot bearing wear against the 85% OEM limit on every inspection report", "Priya Sharma",
     date(2022, 10, 14)),
)


# ======================================================================================
# The 2024 condition picture — what the demo is actually looking at
# ======================================================================================

INSPECTION_ID: Final[str] = "INSP-2024-0315"
INSPECTION_DATE: Final[date] = date(2024, 3, 15)
INSPECTION_SIMILARITY_NOTE: Final[str] = (
    "Wear pattern similar to 2022 failure: identical scoring on the NDE outer race at the load zone, "
    "and the same 1x/2x running-speed spectral signature that preceded the August 2022 seizure."
)
INSPECTION_FINDINGS: Final[tuple[str, ...]] = (
    "NDE bearing wear measured at 71% of the OEM replacement limit by shell-thickness comparison.",
    "Overall vibration 5.8 mm/s RMS at the NDE bearing housing, against a 7.1 mm/s alarm limit.",
    "NDE bearing housing temperature 74 degC at steady 148 m3/h duty.",
    "Lube oil header pressure 1.9 bar, within limits but 0.2 bar below the commissioning value.",
    "Oil sample: ISO 4406 code 18/16/13, iron content 22 ppm, up from 9 ppm in September 2023.",
    INSPECTION_SIMILARITY_NOTE,
)

SHIFT_LOG_DATE: Final[date] = date(2024, 6, 14)
SHIFT_LOG_SHIFT: Final[str] = "B Shift (Night) 22:00 - 06:00"
ALARM_BYPASS_NOTE: Final[str] = (
    "Operator bypassed P-101 alarm twice during night shift"
)
SHIFT_LOG_ENTRIES: Final[tuple[tuple[str, str], ...]] = (
    ("22:05", "Shift handover taken from A Shift. Unit-2 steam header stable at 41 barg. "
              "P-101 running, P-102 on standby."),
    ("23:40", "V-201 deaerator level 62%, dissolved oxygen 5 ppb. Normal."),
    ("01:20", "P-101 discharge pressure PI-1015 reading 42.1 barg, 0.6 barg above the shift average. "
              "E-301 tube-side fouling suspected."),
    ("02:14", "VT-1011 high vibration alarm on P-101 (6.9 mm/s). Alarm bypassed by panel operator "
              "S. Patil to stop repeat annunciation. No work order raised."),
    ("03:47", "VT-1011 high vibration alarm on P-101 again. Alarm bypassed a second time. "
              "Operator bypassed P-101 alarm twice during night shift; supervisor informed at 04:10."),
    ("04:10", "Supervisor A. Deshmukh informed. Instructed operator to log the bypasses and to raise "
              "a maintenance notification at day-shift handover."),
    ("05:30", "P-101 NDE bearing housing temperature 81 degC, up 7 degC on the shift. Noted for "
              "mechanical maintenance attention."),
    ("06:00", "Handover to C Shift. Open item: P-101 vibration and twice-bypassed VT-1011 alarm."),
)

WORK_ORDER_ID: Final[str] = "WO-2024-0342"
WORK_ORDER_DATE: Final[date] = date(2024, 6, 18)
WORK_ORDER_PRIORITY: Final[str] = "P2 — Investigate within 7 days"
WORK_ORDER_STATUS: Final[str] = "OPEN"
WORK_ORDER_HANDWRITTEN_NOTE: Final[str] = "bearing wear 78%"
"""Rendered as a scanned margin annotation in a script hand. This is the genuine trigger for the
low-OCR-confidence uncertainty flag: the number the whole diagnosis hinges on is handwritten."""

WORK_ORDER_FINDINGS: Final[tuple[str, ...]] = (
    f"NDE bearing wear measured at {MEASURED_BEARING_WEAR_PCT:.0f}% of the OEM replacement limit of "
    f"{OEM_BEARING_WEAR_LIMIT_PCT:.0f}% (Sulzer CP 150-400 IOM, Section 7.4).",
    "Overall vibration 6.9 mm/s RMS at the NDE bearing housing (VT-1011), against a 7.1 mm/s alarm "
    "limit — 97% of the alarm setting.",
    "NDE bearing housing temperature 81 degC, up from 74 degC at the March 2024 inspection.",
    "Lube oil header pressure 1.8 bar; LP-101A discharge steady but 0.3 bar below commissioning.",
    "Shift log of 14 June 2024 records that the VT-1011 high-vibration alarm was bypassed twice "
    "during the night shift.",
    "Vibration spectrum shows 1x and 2x running-speed components consistent with the March 2024 "
    "inspection finding of a wear pattern similar to the 2022 failure.",
)
WORK_ORDER_NO_REPLACEMENT_SENTENCE: Final[str] = (
    "No bearing replacement work order has been raised and no bearing replacement is scheduled. "
    "This work order covers investigation only; a replacement work order against SOP-MECH-014 is "
    "recommended but not yet approved."
)
"""The literal sentence the ``threshold_without_workorder`` rule needs. An open *investigation*
work order is not scheduled *corrective* maintenance, and the corpus says so in as many words so
that the distinction is mechanically checkable rather than a matter of interpretation."""

WORK_ORDER_RECOMMENDATIONS: Final[tuple[str, ...]] = (
    "Raise a bearing replacement work order against SOP-MECH-014 before wear reaches the 85% OEM "
    "limit; at the observed rate of approximately 7 percentage points per quarter the limit is "
    "reached in September 2024.",
    "Restore VT-1011 to service and remove the alarm bypass; brief night shift on the 2022 "
    "precedent.",
    "Draw a lube oil sample and compare iron content against the March 2024 result of 22 ppm.",
    "Inspect LP-101A discharge pressure trend — this was the root cause of the 2022 seizure.",
)


# ======================================================================================
# Retirement / knowledge cliff
# ======================================================================================

RETIREMENT_EMAIL_DATE: Final[date] = date(2024, 6, 20)
RETIREMENT_EMAIL_SUBJECT: Final[str] = (
    "Superannuation schedule FY2024-27 — Mechanical Maintenance and Utilities Operations"
)
RETIREMENT_HEADLINE: Final[str] = (
    f"{RAJESH_KUMAR.name} ({RAJESH_KUMAR.employee_id}), Senior Technician — Rotating Equipment, "
    f"completes {RAJESH_KUMAR.years_experience:.0f} years of service and retires in March 2027."
)
RETIREMENT_KNOWLEDGE_RISK: Final[tuple[str, ...]] = (
    f"{RAJESH_KUMAR.name} is the only technician at {PLANT_SHORT} certified on the Sulzer CP-series "
    "bearing replacement procedure (SOP-MECH-014).",
    "He personally executed the August 2022 P-101 bearing replacement and the LP-101A overhaul; "
    "neither job has a written method statement beyond the generic SOP.",
    "His judgement on when a Sulzer CP bearing 'sounds wrong' before the vibration alarm picks it "
    "up is not documented anywhere in the maintenance system.",
    f"{S_RAMASWAMY.name} ({S_RAMASWAMY.employee_id}), Foreman — Boiler & Utilities Operations, with "
    f"{S_RAMASWAMY.years_experience:.0f} years of service, superannuates on 30 November 2024 and holds "
    "the equivalent operating knowledge for V-201.",
    "Handover plan and knowledge-capture sessions are to be scheduled by Asset Reliability.",
)


# ======================================================================================
# SOP-MECH-014 — bearing replacement
# ======================================================================================

SOP_ID: Final[str] = "SOP-MECH-014"
SOP_TITLE: Final[str] = "Bearing Replacement — Horizontal Centrifugal Pumps (Sulzer CP Series)"
SOP_REVISION: Final[str] = "Rev 3"
SOP_EFFECTIVE_DATE: Final[date] = date(2023, 2, 1)
SOP_TOTAL_MINUTES: Final[int] = 240

SOP_STEPS: Final[tuple[ProcedureStepFact, ...]] = (
    ProcedureStepFact(1, "Raise and validate the work permit. Confirm electrical isolation of the "
                         "250 kW driver at the 6.6 kV switchboard and apply lock-out/tag-out.", 20, True),
    ProcedureStepFact(2, "Isolate suction valve HV-1012 and discharge non-return valve CV-1013. "
                         "Drain the casing to the oily water sewer and confirm zero pressure at PI-1015.", 20),
    ProcedureStepFact(3, "Disconnect the lube oil supply and return lines from LP-101A at the bearing "
                         "housings. Blank the open ends.", 10),
    ProcedureStepFact(4, "Record coupling alignment readings (rim and face) before disturbing the "
                         "coupling. These are the reference for step 11.", 15, True),
    ProcedureStepFact(5, "Remove the coupling spacer and the NDE bearing housing end cover. Photograph "
                         "the bearing in situ before extraction.", 20),
    ProcedureStepFact(6, "Extract the NDE angular-contact bearing (SKF 7314 BECBM) with a hydraulic "
                         "puller. Do not apply load through the rolling elements.", 30),
    ProcedureStepFact(7, "Measure and record the shaft journal diameter and the housing bore. Reject "
                         "the shaft if the journal is more than 0.03 mm below nominal.", 15, True),
    ProcedureStepFact(8, "Inspect the removed bearing and record the wear pattern against the OEM "
                         f"replacement criteria in Section 7.4 ({OEM_BEARING_WEAR_LIMIT_PCT:.0f}% limit). "
                         "Retain the bearing for failure analysis if wear exceeds the limit.", 15),
    ProcedureStepFact(9, "Induction-heat the replacement bearing to 110 degC maximum and fit it to the "
                         "shaft. Never flame-heat and never drive a bearing on cold.", 25, True),
    ProcedureStepFact(10, "Refit the bearing housing, renew the lip seals, and reconnect the lube oil "
                          "lines. Confirm free rotation by hand.", 20),
    ProcedureStepFact(11, "Re-align the pump to driver within 0.05 mm rim and 0.03 mm face, using the "
                          "step 4 readings as the reference.", 30, True),
    ProcedureStepFact(12, "Restore lubrication, remove isolation, and run for 30 minutes. Record "
                          "vibration at the NDE and DE bearings; acceptance is below 4.5 mm/s RMS.", 20, True),
)

SOP_TOOLS: Final[tuple[str, ...]] = (
    "Hydraulic bearing puller, 20 t",
    "Induction bearing heater, 110 degC controlled",
    "Dial indicator alignment set (rim and face)",
    "Micrometer set, 75-100 mm, calibrated",
    "Torque wrench, 40-200 Nm, calibrated",
    "Vibration data collector with 1x/2x spectral capability",
)
SOP_SAFETY_NOTES: Final[tuple[str, ...]] = (
    "Hot work permit is NOT valid for bearing fitting. Flame heating destroys the bearing "
    "metallurgy and is prohibited.",
    "The casing holds boiler feed water at up to 165 degC. Confirm drain-down and cool-down before "
    "breaking any joint.",
    "Lock-out/tag-out at the 6.6 kV switchboard must be verified by a second person and recorded.",
    "Lube oil is a slip hazard; bund the work area before disconnecting oil lines.",
)


# ======================================================================================
# Regulation — Factory Act 1948, Section 41(b)
# ======================================================================================

REGULATION_NAME: Final[str] = "Factory Act 1948"
REGULATION_CLAUSE: Final[str] = "Section 41(b)"
REGULATION_OBLIGATION: Final[str] = "monthly pressure vessel inspection"
REGULATION_FREQUENCY_DAYS: Final[int] = 30
REGULATION_APPLIES_TO_TYPES: Final[tuple[str, ...]] = ("pressure_vessel",)
REGULATION_APPLIES_TO_TAGS: Final[tuple[str, ...]] = ("V-201",)
REGULATION_EVIDENCE_TYPES: Final[tuple[str, ...]] = ("inspection_report",)
REGULATION_PENALTY: Final[str] = (
    "Section 92 — imprisonment for a term which may extend to two years, or a fine which may extend "
    "to Rs 1,00,000, or both; continuing contravention attracts a further fine of Rs 1,000 per day."
)
REGULATION_TEXT: Final[str] = (
    "41(b). In every factory, every pressure vessel or plant used in any process shall be examined "
    "by a competent person at intervals not exceeding one month, and a record of every such "
    "examination, signed by the competent person and stating the condition of the vessel and any "
    "defect found, shall be maintained in the prescribed register and produced on demand to the "
    "Inspector."
)
#: The last statutory examination of V-201 that the corpus records. As of ``REFERENCE_DATE`` this
#: is 143 days old against a 30-day obligation — the Factory Act 41(b) gap the demo surfaces.
LAST_V201_STATUTORY_INSPECTION: Final[date] = date(2024, 2, 8)


# ======================================================================================
# Condition-monitoring history (feeds the spreadsheet and the trend narrative)
# ======================================================================================

#: Bearing wear resets to zero at the 2022-08-26 replacement, which is why the 2024 trend restarts.
#: The 2022 run reached 84% and seized; the 2024 run stands at 78% — the same place on the curve.
READINGS: Final[tuple[ReadingFact, ...]] = (
    ReadingFact("P-101", "bearing_wear_pct", 62.0, "%", date(2022, 3, 10), "Maintenance_Log"),
    ReadingFact("P-101", "vibration_mm_s_rms", 4.2, "mm/s", date(2022, 3, 10), "Maintenance_Log"),
    ReadingFact("P-101", "bearing_wear_pct", 84.0, "%", date(2022, 6, 15), "Maintenance_Log",
                "Within 1 point of the OEM limit; no replacement raised."),
    ReadingFact("P-101", "vibration_mm_s_rms", 5.4, "mm/s", date(2022, 6, 15), "Maintenance_Log"),
    ReadingFact("P-101", "bearing_temp_c", 68.0, "degC", date(2022, 8, 14), "Incident_2022_0820"),
    ReadingFact("P-101", "vibration_mm_s_rms", 8.9, "mm/s", date(2022, 8, 19), "Incident_2022_0820",
                "Two alarms acknowledged without action on the night shift."),
    ReadingFact("P-101", "lube_oil_pressure_bar", 1.2, "bar", date(2022, 8, 19), "Incident_2022_0820",
                "Below the 1.4 bar low-low limit."),
    ReadingFact("P-101", "bearing_temp_c", 91.0, "degC", date(2022, 8, 20), "Incident_2022_0820"),
    ReadingFact("P-101", "bearing_wear_pct", 0.0, "%", date(2022, 8, 26), "Maintenance_Log",
                "New NDE and DE bearings fitted under SOP-MECH-014."),
    ReadingFact("P-101", "bearing_wear_pct", 34.0, "%", date(2023, 3, 14), "Maintenance_Log"),
    ReadingFact("P-101", "vibration_mm_s_rms", 3.6, "mm/s", date(2023, 3, 14), "Maintenance_Log"),
    ReadingFact("P-101", "bearing_wear_pct", 52.0, "%", date(2023, 9, 19), "Maintenance_Log"),
    ReadingFact("P-101", "vibration_mm_s_rms", 4.4, "mm/s", date(2023, 9, 19), "Maintenance_Log"),
    ReadingFact("P-101", "bearing_wear_pct", 71.0, "%", INSPECTION_DATE, "Inspection_2024_0315",
                INSPECTION_SIMILARITY_NOTE),
    ReadingFact("P-101", "vibration_mm_s_rms", 5.8, "mm/s", INSPECTION_DATE, "Inspection_2024_0315"),
    ReadingFact("P-101", "bearing_temp_c", 74.0, "degC", INSPECTION_DATE, "Inspection_2024_0315"),
    ReadingFact("P-101", "vibration_mm_s_rms", 6.9, "mm/s", SHIFT_LOG_DATE, "ShiftLog_2024_0614",
                "VT-1011 alarm bypassed twice."),
    ReadingFact("P-101", "bearing_temp_c", 81.0, "degC", SHIFT_LOG_DATE, "ShiftLog_2024_0614"),
    ReadingFact("P-101", "bearing_wear_pct", MEASURED_BEARING_WEAR_PCT, "%", WORK_ORDER_DATE,
                "WO_2024_0342", "Handwritten margin annotation on the work order."),
    ReadingFact("P-101", "vibration_mm_s_rms", 6.9, "mm/s", WORK_ORDER_DATE, "WO_2024_0342"),
    ReadingFact("P-101", "lube_oil_pressure_bar", 1.8, "bar", WORK_ORDER_DATE, "WO_2024_0342"),
    ReadingFact("P-102", "bearing_wear_pct", 88.0, "%", date(2023, 11, 9), "Maintenance_Log",
                "Exceeded the OEM limit; NDE bearing seizure followed the same day."),
    ReadingFact("P-105", "bearing_wear_pct", 69.0, "%", date(2024, 1, 22), "Maintenance_Log",
                "Same failure mode developing on the third machine of the class."),
    ReadingFact("V-201", "shell_thickness_mm", 11.6, "mm", LAST_V201_STATUTORY_INSPECTION,
                "Maintenance_Log", "Against 12.0 mm nominal; corrosion allowance intact."),
)


MAINTENANCE_ROWS: Final[tuple[MaintenanceRowFact, ...]] = (
    MaintenanceRowFact("MR-2022-0301", "P-101", "inspection", date(2022, 3, 10), "Rajesh Kumar",
                       "Bearing wear 62%. Vibration 4.2 mm/s. Lube oil header 2.0 bar.",
                       "Continue monitoring. Next inspection in 90 days.", 0.0, 12000.0, "closed",
                       "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2022-0615", "P-101", "inspection", date(2022, 6, 15), "Rajesh Kumar",
                       "Bearing wear 84%, one point below the 85% OEM replacement limit. "
                       "Vibration 5.4 mm/s.",
                       "Flagged for review. No replacement work order raised.", 0.0, 12000.0, "closed",
                       "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2022-0820", "P-101", "breakdown", INCIDENT_DATE, "Rajesh Kumar",
                       "NDE bearing seizure. Shaft locked. Unit-2 steam raising lost for 14 hours.",
                       "Emergency shutdown, P-102 started, bearing and shaft assessment.",
                       INCIDENT_DOWNTIME_HOURS, INCIDENT_COST_INR, "closed",
                       "Incident_2022_0820_P-101.pdf"),
    MaintenanceRowFact("MR-2022-0824", "LP-101A", "work_order", date(2022, 8, 24), "Rajesh Kumar",
                       "Auxiliary lube oil pump internal wear; volumetric efficiency approximately 55%.",
                       "Overhauled; rotor and idler renewed; discharge restored to 2.1 bar.",
                       0.0, 145000.0, "closed", "RCA_2022_0825_P-101.pdf"),
    MaintenanceRowFact("MR-2022-0826", "P-101", "work_order", date(2022, 8, 26), "Rajesh Kumar",
                       "NDE and DE bearings replaced following the seizure.",
                       f"Executed per {SOP_ID} {SOP_REVISION}. Post-job vibration 3.1 mm/s.",
                       0.0, 380000.0, "closed", "SOP_Bearing_Replacement.pdf"),
    MaintenanceRowFact("MR-2023-0314", "P-101", "inspection", date(2023, 3, 14), "Rajesh Kumar",
                       "Bearing wear 34% on the new bearing set. Vibration 3.6 mm/s.",
                       "Normal. Quarterly oil analysis reinstated.", 0.0, 12000.0, "closed",
                       "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2023-0512", "V-201", "inspection", date(2023, 5, 12), "S. Ramaswamy",
                       "Statutory pressure vessel examination. Shell thickness 11.7 mm. No defects.",
                       "Register signed by competent person. Next examination due 2023-06-11.",
                       0.0, 45000.0, "closed", "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2023-0618", "E-301", "preventive", date(2023, 6, 18), "Priya Sharma",
                       "Tube-side differential pressure 1.4 bar, fouling on the water side.",
                       "Tube bundle chemically cleaned. dP restored to 0.7 bar.", 6.0, 210000.0,
                       "closed", "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2023-0814", "V-201", "inspection", date(2023, 8, 14), "S. Ramaswamy",
                       "Statutory pressure vessel examination. No defects. Relief valve PSV-2011 "
                       "test certificate current.",
                       "Register signed. Next examination due 2023-09-13.", 0.0, 45000.0, "closed",
                       "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2023-0919", "P-101", "inspection", date(2023, 9, 19), "Rajesh Kumar",
                       "Bearing wear 52%. Vibration 4.4 mm/s. Oil iron content 9 ppm.",
                       "Normal wear progression. Continue quarterly monitoring.", 0.0, 12000.0,
                       "closed", "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2023-1109", "P-102", "breakdown", date(2023, 11, 9), "Rajesh Kumar",
                       "NDE bearing seizure on the standby pump during a monthly test run. "
                       "Bearing wear had reached 88%, above the 85% OEM limit. Same failure mode as "
                       "the P-101 event of August 2022.",
                       "Bearings replaced per SOP-MECH-014. Lube oil header pressure found at 1.5 bar.",
                       9.5, 1450000.0, "closed", "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2023-1120", "V-201", "inspection", date(2023, 11, 20), "S. Ramaswamy",
                       "Statutory pressure vessel examination. Minor pitting at the manway seat.",
                       "Register signed. Pitting monitored. Next examination due 2023-12-20.",
                       0.0, 45000.0, "closed", "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2024-0122", "P-105", "inspection", date(2024, 1, 22), "Priya Sharma",
                       "Bearing wear 69% on the Unit-3 machine of the same class. Vibration 5.1 mm/s.",
                       "Added to the fleet bearing-wear watch list.", 0.0, 12000.0, "closed",
                       "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2024-0208", "V-201", "inspection", LAST_V201_STATUTORY_INSPECTION,
                       "S. Ramaswamy",
                       "Statutory pressure vessel examination. Shell thickness 11.6 mm against 12.0 mm "
                       "nominal. Manway seat pitting unchanged.",
                       "Register signed by competent person. Next examination due 2024-03-09.",
                       0.0, 45000.0, "closed", "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2024-0315", "P-101", "inspection", INSPECTION_DATE, "Priya Sharma",
                       "Bearing wear 71%. Vibration 5.8 mm/s. Oil iron content 22 ppm, up from 9 ppm. "
                       + INSPECTION_SIMILARITY_NOTE,
                       "Recommended replacement planning before the 85% limit is reached.",
                       0.0, 12000.0, "closed", "Inspection_2024_0315_P-101.pdf"),
    MaintenanceRowFact("MR-2024-0405", "E-301", "preventive", date(2024, 4, 5), "Priya Sharma",
                       "Tube-side dP 1.2 bar. Fouling returning.",
                       "Partial clean. dP restored to 0.9 bar.", 4.0, 160000.0, "closed",
                       "Maintenance_Log_2022_2024.xlsx"),
    MaintenanceRowFact("MR-2024-0342", "P-101", "work_order", WORK_ORDER_DATE, "Rajesh Kumar",
                       f"Vibration anomaly investigation. NDE bearing wear measured at "
                       f"{MEASURED_BEARING_WEAR_PCT:.0f}% of the {OEM_BEARING_WEAR_LIMIT_PCT:.0f}% OEM "
                       "limit. Vibration 6.9 mm/s against a 7.1 mm/s alarm. VT-1011 alarm bypassed "
                       "twice on the night shift of 14 June 2024.",
                       "Investigation open. No bearing replacement scheduled.", 0.0, 0.0, "open",
                       "WO_2024_0342_P-101.pdf"),
)


# ======================================================================================
# Document manifest — what gets generated, and what each file is for
# ======================================================================================

DOC_OEM_MANUAL: Final[str] = "P-101_OEM_Manual.pdf"
DOC_WORK_ORDER: Final[str] = "WO_2024_0342_P-101.pdf"
DOC_INSPECTION: Final[str] = "Inspection_2024_0315_P-101.pdf"
DOC_SHIFT_LOG: Final[str] = "ShiftLog_2024_0614.pdf"
DOC_INCIDENT: Final[str] = "Incident_2022_0820_P-101.pdf"
DOC_RCA: Final[str] = "RCA_2022_0825_P-101.pdf"
DOC_PID: Final[str] = "P-101_P&ID.png"
DOC_PID_SCANNED: Final[str] = "P-101_P&ID_scanned.png"
DOC_EMAIL: Final[str] = "Email_Retirement_Rajesh.pdf"
DOC_SOP: Final[str] = "SOP_Bearing_Replacement.pdf"
DOC_REGULATION: Final[str] = "Factory_Act_Section41b.pdf"
DOC_MAINTENANCE_LOG: Final[str] = "Maintenance_Log_2022_2024.xlsx"
ASSET_NAMEPLATE: Final[str] = "photos/P-101_nameplate.jpg"

DOCUMENTS: Final[tuple[DocumentSpec, ...]] = (
    DocumentSpec(
        filename=DOC_OEM_MANUAL,
        title="Sulzer CP 150-400 Installation, Operation and Maintenance Manual — P-101",
        document_type="oem_manual",
        document_date=date(2016, 1, 15),
        role="corpus",
        equipment_tags=("P-101", "V-201", "E-301", "LP-101A"),
        key_facts=(
            f"Bearing replacement is mandatory at {OEM_BEARING_WEAR_LIMIT_PCT:.0f}% wear.",
            f"Vibration alarm {OEM_VIBRATION_LIMIT_MM_S} mm/s RMS, trip 9.5 mm/s.",
            f"Lube oil header minimum {OEM_LUBE_PRESSURE_MIN_BAR} bar.",
        ),
        description="The authority for every threshold the rest of the corpus is measured against.",
    ),
    DocumentSpec(
        filename=DOC_WORK_ORDER,
        title=f"Work Order {WORK_ORDER_ID} — P-101 Vibration Anomaly Investigation",
        document_type="work_order",
        document_date=WORK_ORDER_DATE,
        role="corpus",
        equipment_tags=("P-101", "LP-101A", "E-301"),
        people=(RAJESH_KUMAR.name, ANIL_DESHMUKH.name),
        key_facts=(
            f"Bearing wear measured at {MEASURED_BEARING_WEAR_PCT:.0f}%.",
            "Handwritten margin annotation 'bearing wear 78%' — low OCR confidence trigger.",
            "States explicitly that no bearing replacement is scheduled.",
        ),
        description="The open work order the demo's first diagnostic question lands on.",
    ),
    DocumentSpec(
        filename=DOC_INSPECTION,
        title=f"Quarterly Mechanical Inspection {INSPECTION_ID} — P-101",
        document_type="inspection_report",
        document_date=INSPECTION_DATE,
        role="corpus",
        equipment_tags=("P-101",),
        people=(PRIYA_SHARMA.name,),
        key_facts=(
            "Bearing wear 71%, vibration 5.8 mm/s.",
            "'Wear pattern similar to 2022 failure' — the bridge to the incident report.",
        ),
        description="The single sentence that connects the 2024 condition to the 2022 failure.",
    ),
    DocumentSpec(
        filename=DOC_SHIFT_LOG,
        title=f"Shift Log — {SHIFT_LOG_SHIFT}, {SHIFT_LOG_DATE.isoformat()}",
        document_type="shift_log",
        document_date=SHIFT_LOG_DATE,
        role="corpus",
        equipment_tags=("P-101", "V-201", "E-301"),
        people=(ANIL_DESHMUKH.name, SURESH_PATIL.name),
        key_facts=(ALARM_BYPASS_NOTE, "Two VT-1011 bypasses logged at 02:14 and 03:47."),
        description="The operational signal nobody would have connected to the bearing.",
    ),
    DocumentSpec(
        filename=DOC_INCIDENT,
        title=f"Incident Report {INCIDENT_ID} — P-101 NDE Bearing Seizure",
        document_type="incident_report",
        document_date=INCIDENT_DATE,
        role="corpus",
        equipment_tags=("P-101", "P-102", "V-201", "LP-101A"),
        people=(RAJESH_KUMAR.name, ANIL_DESHMUKH.name, D_KRISHNAN.name),
        key_facts=(
            f"{INCIDENT_DOWNTIME_HOURS:.0f} hours downtime.",
            f"{INCIDENT_COST_TEXT} total cost.",
            "Precursor list that the 2024 condition now matches.",
        ),
        description="The business case: what happens if the pattern repeats.",
    ),
    DocumentSpec(
        filename=DOC_RCA,
        title=f"Root Cause Analysis {RCA_ID} — P-101 NDE Bearing Seizure",
        document_type="root_cause_analysis",
        document_date=RCA_DATE,
        role="corpus",
        equipment_tags=("P-101", "LP-101A"),
        people=(PRIYA_SHARMA.name, RAJESH_KUMAR.name, D_KRISHNAN.name),
        key_facts=("Root cause: lubrication failure via LP-101A wear.", "Five-why chain and five "
                   "corrective actions with owners."),
        description="Names the mechanism, so the Copilot can explain *why*, not just *what*.",
    ),
    DocumentSpec(
        filename=DOC_PID,
        title="P&ID — Unit-2 Boiler Feed Water System (P-101 / V-201 / E-301)",
        document_type="pid_drawing",
        document_date=date(2016, 1, 15),
        role="corpus",
        equipment_tags=("P-101", "P-102", "V-201", "E-301", "LP-101A"),
        key_facts=("V-201 -> P-101 -> E-301 process connectivity.",
                   "Valves HV-1012, CV-1013 and instruments VT-1011, PI-1015 rendered as ISO symbols."),
        description="A real raster engineering drawing, drawn to be genuinely machine-parsable.",
    ),
    DocumentSpec(
        filename=DOC_PID_SCANNED,
        title="P&ID — Unit-2 Boiler Feed Water System (scanned copy)",
        document_type="pid_drawing",
        document_date=date(2016, 1, 15),
        role="corpus",
        equipment_tags=("P-101", "P-102", "V-201", "E-301", "LP-101A"),
        key_facts=("Rotated, noisy, contrast-reduced, JPEG-degraded copy of the clean drawing.",
                   "Exercises the OCR tag-correction path (P-l0l -> P-101) for real."),
        description="The same drawing after a photocopier and a flatbed scanner have had their way.",
    ),
    DocumentSpec(
        filename=DOC_EMAIL,
        title=f"E-mail — {RETIREMENT_EMAIL_SUBJECT}",
        document_type="email",
        document_date=RETIREMENT_EMAIL_DATE,
        role="corpus",
        equipment_tags=("P-101", "P-102", "V-201", "LP-101A"),
        people=(MEERA_IYER.name, RAJESH_KUMAR.name, S_RAMASWAMY.name, D_KRISHNAN.name),
        key_facts=(RETIREMENT_HEADLINE,
                   "S. Ramaswamy (31 years) superannuates 2024-11-30 — the nearer cliff."),
        description="The knowledge cliff, hiding in an HR e-mail nobody would think to search.",
    ),
    DocumentSpec(
        filename=DOC_SOP,
        title=f"{SOP_ID} {SOP_REVISION} — {SOP_TITLE}",
        document_type="sop",
        document_date=SOP_EFFECTIVE_DATE,
        role="corpus",
        equipment_tags=("P-101", "P-102", "P-105"),
        key_facts=(f"{SOP_TOTAL_MINUTES} minutes total across 12 steps.",
                   "Five hold points; flame heating explicitly prohibited."),
        description="The procedural answer, with real steps and a real time budget.",
    ),
    DocumentSpec(
        filename=DOC_REGULATION,
        title=f"{REGULATION_NAME} — {REGULATION_CLAUSE} Extract and Compliance Note",
        document_type="regulation",
        document_date=date(2023, 4, 1),
        role="corpus",
        equipment_tags=("V-201",),
        key_facts=(f"{REGULATION_CLAUSE}: {REGULATION_OBLIGATION}, every "
                   f"{REGULATION_FREQUENCY_DAYS} days.",
                   f"Last V-201 examination on record: {LAST_V201_STATUTORY_INSPECTION.isoformat()}."),
        description="The obligation the audit measures V-201 against.",
    ),
    DocumentSpec(
        filename=DOC_MAINTENANCE_LOG,
        title="Maintenance and Condition-Monitoring Log 2022-2024 — Unit-2 Utilities",
        document_type="spreadsheet",
        document_date=WORK_ORDER_DATE,
        role="corpus",
        equipment_tags=("P-101", "P-102", "P-105", "V-201", "E-301", "LP-101A"),
        people=(RAJESH_KUMAR.name, PRIYA_SHARMA.name, S_RAMASWAMY.name),
        key_facts=("17 maintenance records and 24 condition readings across three years.",
                   "P-102 bearing seizure 2023-11-09 gives the fleet-pattern rule a second asset."),
        description="Structured history: the trend line behind every claim the prose makes.",
    ),
    DocumentSpec(
        filename=ASSET_NAMEPLATE,
        title="Equipment nameplate photograph — P-101",
        document_type="unknown",
        document_date=REFERENCE_DATE,
        role="asset",
        equipment_tags=("P-101",),
        key_facts=("Synthetic field photo for the photo-to-query demo beat.",),
        description="Not part of the ingest corpus; input for the mobile agent's AR moment.",
    ),
)

CORPUS_FILENAMES: Final[tuple[str, ...]] = tuple(
    doc.filename for doc in DOCUMENTS if doc.role == "corpus"
)


# ======================================================================================
# Cross-document links — the demo's thesis, in machine-checkable form
# ======================================================================================

CROSS_DOCUMENT_LINKS: Final[tuple[CrossLink, ...]] = (
    CrossLink(
        link_id="threshold_vs_measurement",
        claim=(f"P-101 bearing wear is at {MEASURED_BEARING_WEAR_PCT:.0f}% against an OEM "
               f"replacement limit of {OEM_BEARING_WEAR_LIMIT_PCT:.0f}% — a number that exists in "
               "neither document alone."),
        documents=(DOC_OEM_MANUAL, DOC_WORK_ORDER),
        probe_terms=("85", "78", "bearing wear"),
    ),
    CrossLink(
        link_id="pattern_matches_2022",
        claim=("The March 2024 inspection calls the wear pattern similar to the 2022 failure; the "
               "2022 incident report says that failure cost 14 hours and Rs 25 lakh."),
        documents=(DOC_INSPECTION, DOC_INCIDENT),
        probe_terms=("wear pattern similar to 2022 failure", "14", "25,00,000"),
    ),
    CrossLink(
        link_id="root_cause_is_lubrication",
        claim="The 2022 seizure was caused by lubrication failure originating in LP-101A.",
        documents=(DOC_INCIDENT, DOC_RCA),
        probe_terms=("lubrication failure", "LP-101A"),
    ),
    CrossLink(
        link_id="alarm_bypass_precedes_wear",
        claim=("An operator bypassed the P-101 vibration alarm twice on the night shift of "
               "14 June 2024, while an open work order records bearing wear at 78%."),
        documents=(DOC_SHIFT_LOG, DOC_WORK_ORDER),
        probe_terms=("bypassed", "VT-1011", "78"),
    ),
    CrossLink(
        link_id="expert_retirement",
        claim=(f"{RAJESH_KUMAR.name}, the only technician certified on the bearing replacement "
               "procedure, retires in March 2027 after 23 years."),
        documents=(DOC_EMAIL, DOC_WORK_ORDER, DOC_SOP),
        probe_terms=("Rajesh Kumar", "23", "2027"),
    ),
    CrossLink(
        link_id="process_connectivity",
        claim="P-101 draws from V-201 and discharges to E-301; the drawing and the manual agree.",
        documents=(DOC_PID, DOC_OEM_MANUAL),
        probe_terms=("V-201", "E-301", "P-101"),
    ),
    CrossLink(
        link_id="fleet_pattern",
        claim=("The same NDE bearing failure mode has now occurred on two Sulzer CP 150-400 machines "
               "(P-101 in 2022, P-102 in 2023) with a third trending (P-105)."),
        documents=(DOC_MAINTENANCE_LOG, DOC_INCIDENT),
        probe_terms=("P-102", "bearing seizure", "P-105"),
    ),
    CrossLink(
        link_id="statutory_gap",
        claim=(f"{REGULATION_CLAUSE} requires {REGULATION_OBLIGATION}; the last recorded V-201 "
               f"examination is {LAST_V201_STATUTORY_INSPECTION.isoformat()}."),
        documents=(DOC_REGULATION, DOC_MAINTENANCE_LOG),
        probe_terms=("41(b)", "V-201", "monthly"),
    ),
)


# ======================================================================================
# Derived helpers
# ======================================================================================


def days_since(when: date, *, anchor: date = REFERENCE_DATE) -> int:
    """Whole days between ``when`` and the corpus anchor. Never uses the wall clock."""
    return (anchor - when).days


def equipment_by_tag(tag: str) -> EquipmentFact | None:
    """Look up an :class:`EquipmentFact` by plant tag, case-insensitively."""
    wanted = tag.strip().upper()
    for item in EQUIPMENT:
        if item.tag.upper() == wanted:
            return item
    return None


def person_by_name(name: str) -> PersonFact | None:
    """Look up a :class:`PersonFact` by display name, case-insensitively."""
    wanted = name.strip().casefold()
    for person in PEOPLE:
        if person.name.casefold() == wanted:
            return person
    return None


def document_spec(filename: str) -> DocumentSpec | None:
    """Look up a :class:`DocumentSpec` by generated filename."""
    for doc in DOCUMENTS:
        if doc.filename == filename:
            return doc
    return None


def readings_for(tag: str, parameter: str) -> tuple[ReadingFact, ...]:
    """Every reading of one parameter for one asset, in chronological order."""
    wanted = tag.strip().upper()
    matches = [r for r in READINGS if r.equipment_tag.upper() == wanted and r.parameter == parameter]
    return tuple(sorted(matches, key=lambda r: r.measured_on))


def maintenance_for(tag: str) -> tuple[MaintenanceRowFact, ...]:
    """Every maintenance row for one asset, in chronological order."""
    wanted = tag.strip().upper()
    matches = [row for row in MAINTENANCE_ROWS if row.equipment_tag.upper() == wanted]
    return tuple(sorted(matches, key=lambda r: r.performed_on))


def consistency_report() -> tuple[str, ...]:
    """Self-check the fact sheet. Returns a tuple of problems; empty means consistent.

    Cheap insurance against a future edit that moves one number and not its siblings. Called by
    :mod:`scripts.generate_demo_data` before a single byte is written.
    """
    problems: list[str] = []

    if MEASURED_BEARING_WEAR_PCT >= OEM_BEARING_WEAR_LIMIT_PCT:
        problems.append(
            f"Measured wear {MEASURED_BEARING_WEAR_PCT} must sit below the OEM limit "
            f"{OEM_BEARING_WEAR_LIMIT_PCT}; the whole 'approaching the limit' narrative depends on it."
        )
    if P101.oem_thresholds.get("bearing_wear_pct") != OEM_BEARING_WEAR_LIMIT_PCT:
        problems.append("P-101 oem_thresholds disagree with OEM_BEARING_WEAR_LIMIT_PCT.")

    wear = readings_for("P-101", "bearing_wear_pct")
    latest = [r for r in wear if r.measured_on == WORK_ORDER_DATE]
    if not latest or latest[0].value != MEASURED_BEARING_WEAR_PCT:
        problems.append("READINGS has no P-101 bearing_wear_pct entry matching the work-order value.")

    post_replacement = [r for r in wear if r.measured_on > date(2022, 8, 26)]
    for earlier, later in zip(post_replacement, post_replacement[1:], strict=False):
        if later.value < earlier.value:
            problems.append(
                f"Bearing wear must increase monotonically after the 2022 replacement; "
                f"{later.measured_on} ({later.value}) is below {earlier.measured_on} ({earlier.value})."
            )

    if INSPECTION_SIMILARITY_NOTE.split(":")[0] != "Wear pattern similar to 2022 failure":
        problems.append("The inspection similarity note no longer opens with the charter's phrase.")
    if str(int(INCIDENT_DOWNTIME_HOURS)) not in "14":
        problems.append("Incident downtime must be 14 hours.")
    if INCIDENT_COST_INR != 2_500_000.0:
        problems.append("Incident cost must be Rs 25,00,000.")
    if "lubrication failure" not in RCA_ROOT_CAUSE.casefold():
        problems.append("The RCA root cause must name lubrication failure.")
    if RAJESH_KUMAR.retirement_on != date(2027, 3, 31) or RAJESH_KUMAR.years_experience != 23.0:
        problems.append("Rajesh Kumar must have 23 years of service and a March 2027 retirement.")
    if SHIFT_LOG_DATE != date(2024, 6, 14):
        problems.append("The alarm bypass must fall on the night shift of 2024-06-14.")

    bypass_mentions = sum(1 for _, text in SHIFT_LOG_ENTRIES if "bypass" in text.casefold())
    if bypass_mentions < 2:
        problems.append("The shift log must record two distinct alarm bypasses.")

    fleet = {row.equipment_tag for row in MAINTENANCE_ROWS
             if row.record_type == "breakdown" and "bearing seizure" in row.findings.casefold()}
    if len(fleet) < 2:
        problems.append(
            "At least two assets must share the bearing-seizure failure mode or the fleet_pattern "
            "rule has nothing to fire on."
        )

    total_minutes = sum(step.minutes for step in SOP_STEPS)
    if total_minutes != SOP_TOTAL_MINUTES:
        problems.append(
            f"SOP step minutes sum to {total_minutes}, not the advertised {SOP_TOTAL_MINUTES}."
        )

    overdue = days_since(LAST_V201_STATUTORY_INSPECTION) - REGULATION_FREQUENCY_DAYS
    if overdue <= 0:
        problems.append(
            "The last V-201 statutory examination must be more than one month before the reference "
            "date, or there is no Factory Act 41(b) gap to find."
        )

    filenames = [doc.filename for doc in DOCUMENTS]
    if len(filenames) != len(set(filenames)):
        problems.append("Duplicate filename in DOCUMENTS.")

    known = set(filenames)
    for link in CROSS_DOCUMENT_LINKS:
        missing = [name for name in link.documents if name not in known]
        if missing:
            problems.append(f"Cross-link {link.link_id} references unknown documents: {missing}.")

    return tuple(problems)


def summary_lines() -> tuple[str, ...]:
    """Human-readable one-liners describing the corpus, used by the generator's report."""
    return (
        f"Plant                 {PLANT_ADDRESS}",
        f"Reference date        {REFERENCE_DATE.isoformat()} (all lookbacks anchor here)",
        f"Equipment             {', '.join(item.tag for item in EQUIPMENT)}",
        f"People                {', '.join(person.name for person in PEOPLE)}",
        f"OEM bearing limit     {OEM_BEARING_WEAR_LIMIT_PCT:.0f}%",
        f"Measured wear         {MEASURED_BEARING_WEAR_PCT:.0f}% ({WORK_ORDER_ID}, "
        f"{days_since(WORK_ORDER_DATE)} days before the reference date)",
        f"2022 failure          {INCIDENT_ID}, {INCIDENT_DOWNTIME_HOURS:.0f} h, {INCIDENT_COST_TEXT}",
        f"Statutory gap         {REGULATION_CLAUSE} on V-201, "
        f"{days_since(LAST_V201_STATUTORY_INSPECTION) - REGULATION_FREQUENCY_DAYS} days overdue",
    )


def as_manifest_facts() -> dict[str, object]:
    """The fact sheet, flattened for ``manifest.json``.

    Downstream scripts and tests read this rather than re-deriving constants, so there is exactly
    one place a number can be wrong.
    """
    return {
        "plant": {
            "name": PLANT_NAME,
            "short_name": PLANT_SHORT,
            "unit": PLANT_UNIT,
            "location": PLANT_LOCATION,
        },
        "reference_date": REFERENCE_DATE.isoformat(),
        "thresholds": {
            "oem_bearing_wear_pct": OEM_BEARING_WEAR_LIMIT_PCT,
            "measured_bearing_wear_pct": MEASURED_BEARING_WEAR_PCT,
            "oem_vibration_mm_s_rms": OEM_VIBRATION_LIMIT_MM_S,
            "oem_bearing_temp_c": OEM_BEARING_TEMP_LIMIT_C,
            "oem_lube_pressure_bar_min": OEM_LUBE_PRESSURE_MIN_BAR,
        },
        "incident_2022": {
            "incident_id": INCIDENT_ID,
            "occurred_on": INCIDENT_DATE.isoformat(),
            "failure_mode": INCIDENT_FAILURE_MODE,
            "downtime_hours": INCIDENT_DOWNTIME_HOURS,
            "cost_inr": INCIDENT_COST_INR,
            "cost_text": INCIDENT_COST_TEXT,
            "root_cause": RCA_ROOT_CAUSE,
            "rca_id": RCA_ID,
        },
        "work_order_2024": {
            "work_order_id": WORK_ORDER_ID,
            "raised_on": WORK_ORDER_DATE.isoformat(),
            "status": WORK_ORDER_STATUS,
            "assigned_to": RAJESH_KUMAR.name,
            "measured_bearing_wear_pct": MEASURED_BEARING_WEAR_PCT,
            "handwritten_note": WORK_ORDER_HANDWRITTEN_NOTE,
        },
        "alarm_bypass": {
            "occurred_on": SHIFT_LOG_DATE.isoformat(),
            "shift": SHIFT_LOG_SHIFT,
            "note": ALARM_BYPASS_NOTE,
            "instrument": "VT-1011",
            "count": 2,
        },
        "equipment": [
            {
                "tag": item.tag,
                "name": item.name,
                "equipment_type": item.equipment_type,
                "criticality": item.criticality,
                "manufacturer": item.manufacturer,
                "model": item.model,
                "installed_on": item.installed_on.isoformat(),
                "oem_thresholds": dict(item.oem_thresholds),
            }
            for item in EQUIPMENT
        ],
        "people": [
            {
                "name": person.name,
                "employee_id": person.employee_id,
                "role": person.role,
                "years_experience": person.years_experience,
                "retirement_on": person.retirement_on.isoformat() if person.retirement_on else None,
                "expertise_tags": list(person.expertise_tags),
                "documented_contributions": person.documented_contributions,
            }
            for person in PEOPLE
        ],
        "regulation": {
            "regulation": REGULATION_NAME,
            "clause": REGULATION_CLAUSE,
            "obligation": REGULATION_OBLIGATION,
            "frequency_days": REGULATION_FREQUENCY_DAYS,
            "applies_to_tags": list(REGULATION_APPLIES_TO_TAGS),
            "last_evidence_date": LAST_V201_STATUTORY_INSPECTION.isoformat(),
            "days_overdue": days_since(LAST_V201_STATUTORY_INSPECTION) - REGULATION_FREQUENCY_DAYS,
            "penalty": REGULATION_PENALTY,
        },
        "procedure": {
            "procedure_id": SOP_ID,
            "title": SOP_TITLE,
            "revision": SOP_REVISION,
            "estimated_minutes": SOP_TOTAL_MINUTES,
            "step_count": len(SOP_STEPS),
        },
        "cross_document_links": [
            {
                "link_id": link.link_id,
                "claim": link.claim,
                "documents": list(link.documents),
                "probe_terms": list(link.probe_terms),
            }
            for link in CROSS_DOCUMENT_LINKS
        ],
    }


__all__ = [
    "ALARM_BYPASS_NOTE", "ASSET_NAMEPLATE", "CORPUS_FILENAMES", "CROSS_DOCUMENT_LINKS",
    "DOCUMENTS", "DOC_EMAIL", "DOC_INCIDENT", "DOC_INSPECTION", "DOC_MAINTENANCE_LOG",
    "DOC_OEM_MANUAL", "DOC_PID", "DOC_PID_SCANNED", "DOC_RCA", "DOC_REGULATION", "DOC_SHIFT_LOG",
    "DOC_SOP", "DOC_WORK_ORDER", "EQUIPMENT", "CrossLink", "DocumentSpec", "EquipmentFact",
    "FIXED_TIMESTAMP", "INCIDENT_COST_INR", "INCIDENT_COST_TEXT", "INCIDENT_DATE",
    "INCIDENT_DOWNTIME_HOURS", "INCIDENT_FAILURE_MODE", "INCIDENT_ID", "INCIDENT_PRECURSORS",
    "INSPECTION_DATE", "INSPECTION_FINDINGS", "INSPECTION_ID", "INSPECTION_SIMILARITY_NOTE",
    "INSTRUMENT_TAGS", "LAST_V201_STATUTORY_INSPECTION", "MAINTENANCE_ROWS",
    "MEASURED_BEARING_WEAR_PCT", "MaintenanceRowFact", "OEM_BEARING_TEMP_LIMIT_C",
    "OEM_BEARING_WEAR_LIMIT_PCT", "OEM_LUBE_PRESSURE_MIN_BAR", "OEM_VIBRATION_LIMIT_MM_S",
    "PEOPLE", "PLANT_ADDRESS", "PLANT_LOCATION", "PLANT_NAME", "PLANT_SHORT", "PLANT_UNIT",
    "PersonFact", "ProcedureStepFact", "READINGS", "REFERENCE_DATE", "REGULATION_APPLIES_TO_TAGS",
    "REGULATION_APPLIES_TO_TYPES", "REGULATION_CLAUSE", "REGULATION_EVIDENCE_TYPES",
    "REGULATION_FREQUENCY_DAYS", "REGULATION_NAME", "REGULATION_OBLIGATION", "REGULATION_PENALTY",
    "REGULATION_TEXT", "RETIREMENT_EMAIL_DATE", "RETIREMENT_EMAIL_SUBJECT", "RETIREMENT_HEADLINE",
    "RETIREMENT_KNOWLEDGE_RISK", "RCA_CONTRIBUTING_FACTORS", "RCA_CORRECTIVE_ACTIONS", "RCA_DATE",
    "RCA_FIVE_WHYS", "RCA_ID", "RCA_ROOT_CAUSE", "ReadingFact", "SHIFT_LOG_DATE",
    "SHIFT_LOG_ENTRIES", "SHIFT_LOG_SHIFT", "SOP_EFFECTIVE_DATE", "SOP_ID", "SOP_REVISION",
    "SOP_SAFETY_NOTES", "SOP_STEPS", "SOP_TITLE", "SOP_TOOLS", "SOP_TOTAL_MINUTES", "VALVE_TAGS",
    "WORK_ORDER_DATE", "WORK_ORDER_FINDINGS", "WORK_ORDER_HANDWRITTEN_NOTE", "WORK_ORDER_ID",
    "WORK_ORDER_NO_REPLACEMENT_SENTENCE", "WORK_ORDER_PRIORITY", "WORK_ORDER_RECOMMENDATIONS",
    "WORK_ORDER_STATUS", "as_manifest_facts", "consistency_report", "days_since", "document_spec",
    "equipment_by_tag", "maintenance_for", "person_by_name", "readings_for", "summary_lines",
]
