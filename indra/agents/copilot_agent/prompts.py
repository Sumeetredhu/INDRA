"""Every prompt and generated-text template the Copilot Agent emits.

Nothing else in this package builds prompt text. A handler that needs to talk to a model imports a
named constant from here and fills it through :func:`render`. That rule exists for three reasons:

* **Auditability.** An industrial answer is only as trustworthy as the instruction that produced it.
  Reviewers read one file, not seven handlers.
* **Versioning.** Every constant carries a ``_V<n>`` suffix. Changing a prompt means adding ``_V2``
  and moving callers, so a regression can be bisected to an exact instruction change.
* **Determinism.** Prompts assembled inline drift; prompts assembled from one template do not.

Templates use :meth:`str.format` placeholders. Literal braces are doubled. JSON *schemas* live here
too — a schema is half of the instruction, and splitting the pair across modules is how the two
silently diverge.
"""

from __future__ import annotations

from typing import Any, Final

from indra.core.exceptions import ConfigurationError

#: Bumped whenever any constant below changes semantically. Recorded on reasoning steps so an
#: answer produced last week can be attributed to the instruction set that produced it.
PROMPT_VERSION: Final[str] = "v1"


def render(template: str, /, **values: object) -> str:
    """Fill a prompt template.

    A missing or malformed placeholder is a programming error in this package, not a runtime
    condition, so it surfaces as :class:`ConfigurationError` with the offending key named rather
    than as a ``KeyError`` from somewhere deep in a handler.
    """
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        raise ConfigurationError(
            "Copilot prompt template could not be rendered; a placeholder is missing or malformed. "
            "Check the constant in indra.agents.copilot_agent.prompts against its call site.",
            context={"missing": str(exc), "supplied": sorted(values)},
            cause=exc,
        ) from exc


# ======================================================================================
# Shared system instructions
# ======================================================================================

GROUNDED_SYSTEM_V1: Final[str] = """\
You are INDRA, an industrial plant intelligence assistant used by maintenance engineers and \
operators on a live process plant. Your answers are acted on by people holding tools next to \
running machinery.

Rules you must follow without exception:
1. Use ONLY the numbered passages supplied as CONTEXT. You have no other knowledge of this plant.
2. Cite every factual claim with the passage number in square brackets, like [2]. A sentence that \
states a number, a date, a limit, or a cause and carries no citation is a defect.
3. If the context does not contain the answer, say so plainly and state what is missing. Never \
infer a specification, a threshold, or a root cause that is not written in the context.
4. Never invent a passage number. Only cite numbers that appear in the CONTEXT block.
5. Preserve equipment tags, part numbers and units exactly as written. Do not reformat P-101.
6. Be terse. An operator reads this on a phone in a plant. No preamble, no restating the question, \
no closing pleasantries.
7. Write in English. Use plain sentences and, where a sequence matters, numbered lines.
"""

CLASSIFICATION_SYSTEM_V1: Final[str] = """\
You classify questions asked of an industrial plant knowledge system into exactly one category. \
You return JSON only. You never explain outside the JSON object.
"""

EXTRACTION_SYSTEM_V1: Final[str] = """\
You extract structured data from industrial maintenance evidence. You return JSON only, strictly \
matching the requested schema. You never add fields, never guess values that are absent from the \
evidence, and never soften a finding.
"""


# ======================================================================================
# Query classification
# ======================================================================================

CLASSIFICATION_PROMPT_V1: Final[str] = """\
Classify the plant question below into exactly one category.

CATEGORIES
- factual: asks for a stored fact, specification, limit, rating, date, owner or count.
- diagnostic: asks why something happened, failed, degraded or behaved abnormally; asks for a root
  cause; asks to troubleshoot an observed symptom.
- procedural: asks how to perform work — a procedure, SOP, sequence of steps, isolation, replacement.
- predictive: asks about the future — will it fail, when, what is the risk, what is the remaining
  life, what should we expect.
- comparative: asks to compare two or more assets, periods, options or readings against each other.
- compliance: asks about regulations, statutory clauses, audits, inspection obligations or evidence
  of conformity.
- knowledge_gap: asks what is NOT known, NOT documented, missing, undocumented, or where the
  records are incomplete.

A deterministic keyword analysis already ran and produced a prior. Treat it as a colleague's
opinion: confirm it when the wording supports it, override it when the wording clearly does not.

PRIOR CATEGORY: {prior_type}
PRIOR REASONING: {prior_rationale}

QUESTION: {query}

Return JSON with:
- "query_type": one of the category ids above
- "confidence": your confidence in that label, 0.0 to 1.0
- "rationale": one short sentence naming the wording that decided it
- "equipment_tags": plant tags mentioned in the question, exactly as written, or an empty list
"""

CLASSIFICATION_SCHEMA_V1: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "query_type": {
            "type": "string",
            "enum": [
                "factual",
                "diagnostic",
                "procedural",
                "predictive",
                "comparative",
                "compliance",
                "knowledge_gap",
            ],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "equipment_tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["query_type", "confidence", "rationale"],
    "additionalProperties": False,
}


# ======================================================================================
# Context rendering
# ======================================================================================

CONTEXT_HEADER_V1: Final[str] = "CONTEXT — numbered passages retrieved from the plant document set:"

CONTEXT_PASSAGE_V1: Final[str] = """\
[{index}] {title}{page} | type={doc_type} | date={doc_date} | relevance={relevance}
{text}
"""

CONTEXT_PATHS_HEADER_V1: Final[str] = (
    "GRAPH CONNECTIONS — relationships the knowledge graph found between the entities in this "
    "question. These are structural facts, not passages, and carry no citation number:"
)

CONTEXT_PATH_V1: Final[str] = "- {narrative} ({hops} hop(s), confidence {confidence})"


# ======================================================================================
# Handler prompts
# ======================================================================================

FACTUAL_PROMPT_V1: Final[str] = """\
{context}

STRUCTURED PLANT RECORDS (authoritative; cite the passage that documents them where one exists):
{records}

QUESTION: {query}

Answer the question directly in at most three sentences. Lead with the value or fact asked for,
including its units. Cite each fact. If the context does not contain the fact, say exactly what is
missing and name the closest thing the context does contain.
"""

PROCEDURAL_PROMPT_V1: Final[str] = """\
{context}

QUESTION: {query}

Reconstruct the procedure from the context as a numbered list of steps, in order, using the wording
of the source as closely as possible. Do not invent, merge, reorder or "improve" a step.

After the steps, add — only where the context supports it — these sections, each on its own line:
TOOLS: comma-separated tools and parts required
SAFETY: each safety precaution, isolation or permit requirement, one per line
DURATION: the estimated time

If the context contains only a fragment of the procedure, give the fragment and state plainly which
part of the procedure is not documented. A partial procedure honestly labelled is safe; a completed
one is not.
"""

COMPARATIVE_PROMPT_V1: Final[str] = """\
{context}

SUBJECTS UNDER COMPARISON: {subjects}

ALIGNED PARAMETERS EXTRACTED FROM PLANT RECORDS:
{table}

QUESTION: {query}

Compare the subjects on the parameters that the context actually supports. Structure the answer as:
1. A one-line verdict answering the question asked.
2. A markdown table with one row per parameter and one column per subject. Put a dash where a
   subject has no documented value — never carry a value across from the other subject.
3. Two or three sentences on what explains the difference, cited.

Only compare parameters that are documented for at least one subject. State explicitly when a
parameter is documented for one subject and not the other, because that asymmetry is itself a
finding.
"""

PREDICTIVE_PROMPT_V1: Final[str] = """\
{context}

RISK ASSESSMENT PRODUCED BY THE PROACTIVE INTELLIGENCE AGENT:
{prediction}

QUESTION: {query}

Explain the risk to a maintenance planner. Structure:
1. The headline: the failure mode, the probability and the horizon, stated in one sentence.
2. The drivers, one per line, each tied to the evidence that supports it and cited.
3. What would change the answer — the observation or measurement that would raise or lower this
   risk most.

The probability came from a model over plant history, not from the passages. Do not restate it as
though a document asserted it, and do not adjust it. Explain what drove it.
"""

COMPLIANCE_PROMPT_V1: Final[str] = """\
{context}

COMPLIANCE ASSESSMENT PRODUCED BY THE COMPLIANCE AGENT:
{assessment}

QUESTION: {query}

Answer as a compliance officer briefing an inspector. Structure:
1. The status in one sentence: compliant, or not, and against which clause.
2. Each requirement in scope, with its evidence and evidence date, cited. Where evidence is
   missing or expired, say so in those words.
3. The corrective action and its deadline, where a gap exists.

Never describe an obligation as met unless the evidence for it is in the context. "Probably
compliant" is not an answer an inspector can use.
"""

KNOWLEDGE_GAP_PROMPT_V1: Final[str] = """\
{context}

DOCUMENTED COVERAGE MEASURED ACROSS THE PLANT RECORD SET:
{coverage}

QUESTION: {query}

Report what is NOT known. This is an inverted retrieval: the absence of a document is the finding,
not a failure. Structure:
1. One sentence on the overall state of documentation for the subject.
2. The specific gaps, one per line — the missing document class, the stale record, the failure with
   no root-cause analysis, the expertise that exists only in a person's head.
3. The single gap that most deserves to be closed first, and why.

Do not pad this with what IS documented, beyond the one line needed for context. Never speculate
about what a missing document would have said.
"""

DIAGNOSTIC_PROMPT_V1: Final[str] = """\
{context}

STRUCTURED EVIDENCE ASSEMBLED FOR {tag} BY THE DIAGNOSTIC CHAIN:
{evidence}

QUESTION: {query}

You are writing the root-cause narrative that a reliability engineer will act on.

Structure:
1. THE CAUSE — one sentence naming the most probable root cause. If the evidence supports two
   competing causes, name the stronger one here and the other in point 4.
2. THE CHAIN — the sequence of events that led to it, one line per link, oldest first, each cited.
   Where the evidence is a semantic match between a current symptom and a historical precursor,
   say so explicitly and name the historical event and its date. That cross-document link is the
   most valuable line in this answer.
3. WHAT CONFIRMS IT — the measurements or observations that corroborate the cause, cited.
4. WHAT WOULD FALSIFY IT — the competing explanation the evidence does not rule out, and the one
   check that would settle it.

The structured evidence above was assembled by deterministic queries against the plant record set.
Treat it as fact. The passages are its provenance — cite them. Never assert a cause that neither
the structured evidence nor a passage supports; if the evidence stops short of a root cause, say
which link in the chain is missing.
"""


# ======================================================================================
# Structured post-generation extraction
# ======================================================================================

RECOMMENDED_ACTIONS_PROMPT_V1: Final[str] = """\
FINDINGS:
{findings}

EQUIPMENT: {tag}
CRITICALITY: {criticality}

Propose the concrete next actions a maintenance team should take, most urgent first, at most
{limit}. Each action must be something a named role can start today — an inspection, a
measurement, a part order, a procedure to run, a person to interview. Reject anything of the form
"monitor the situation" or "consider reviewing": those are not actions.

For each action return:
- "action": the imperative instruction, naming the equipment tag
- "urgency": one of INFO, LOW, WARNING, HIGH, CRITICAL
- "owner_role": the role that owns it, e.g. Reliability Engineer, Shift Supervisor
- "due_within_days": integer days, or null when it is not time-bound
- "rationale": the single finding that justifies it
"""

RECOMMENDED_ACTIONS_SCHEMA_V1: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "urgency": {
                        "type": "string",
                        "enum": ["INFO", "LOW", "WARNING", "HIGH", "CRITICAL"],
                    },
                    "owner_role": {"type": "string"},
                    "due_within_days": {"type": ["integer", "null"], "minimum": 0},
                    "rationale": {"type": "string"},
                },
                "required": ["action", "urgency"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["actions"],
    "additionalProperties": False,
}

ALTERNATIVE_INTERPRETATIONS_PROMPT_V1: Final[str] = """\
CONCLUSION REACHED: {conclusion}

EVIDENCE IT RESTS ON:
{evidence}

Name the alternative explanations this evidence does not rule out, at most {limit}. Each must be a
genuinely competing account of the same evidence, not a restatement or a weaker version of the
conclusion. For each, state the observation that would distinguish it from the conclusion above.

If the evidence is strong enough that no serious alternative remains, return an empty list. Padding
this list with contrived alternatives is worse than returning none.
"""

ALTERNATIVE_INTERPRETATIONS_SCHEMA_V1: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "interpretation": {"type": "string"},
                    "distinguishing_check": {"type": "string"},
                },
                "required": ["interpretation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["alternatives"],
    "additionalProperties": False,
}


# ======================================================================================
# Deterministic answer templates — used when no model is reachable, and for verbatim rendering
# ======================================================================================

NO_EVIDENCE_ANSWER_V1: Final[str] = """\
I don't know — and I am not going to guess.

Nothing in the ingested plant record set answers this question. Here is exactly what I searched:

{search_report}

What would let me answer it:
{suggestions}
"""

NO_EVIDENCE_SEARCH_LINE_V1: Final[str] = "- {label}: {detail}"

NO_EVIDENCE_SUGGESTION_V1: Final[str] = "- {suggestion}"

EXTRACTIVE_FALLBACK_V1: Final[str] = """\
No language model was reachable, so this is a verbatim extract of the highest-scoring passages
rather than a composed answer. The citations are exact; the connective reasoning is not present.

{extracts}
"""

EXTRACTIVE_PASSAGE_V1: Final[str] = "{text} [{index}]"

PROCEDURE_ANSWER_V1: Final[str] = """\
{title}{revision}

{steps}
{tools}{safety}{duration}"""

PROCEDURE_STEP_V1: Final[str] = "{number}. {step}"

PROCEDURE_TOOLS_V1: Final[str] = "\nTOOLS: {tools}\n"

PROCEDURE_SAFETY_V1: Final[str] = "\nSAFETY:\n{notes}\n"

PROCEDURE_SAFETY_NOTE_V1: Final[str] = "- {note}"

PROCEDURE_DURATION_V1: Final[str] = "\nDURATION: approximately {minutes} minutes\n"

COMPARISON_ROW_V1: Final[str] = "| {parameter} | {values} |"

COMPARISON_HEADER_V1: Final[str] = "| Parameter | {subjects} |"

COMPARISON_DIVIDER_V1: Final[str] = "|---|{cells}|"

DIGEST_SECTION_V1: Final[str] = "### {title}\n{body}\n"

DIGEST_EMPTY_V1: Final[str] = "(searched, nothing found)"

DIGEST_BULLET_V1: Final[str] = "- {text}"

DEGRADED_ANSWER_V1: Final[str] = """\
I could not complete this answer.

{detail}

The evidence I had retrieved before the failure is listed below, unsummarised, so the retrieval work
is not lost:

{extracts}
"""


__all__ = [
    "ALTERNATIVE_INTERPRETATIONS_PROMPT_V1",
    "ALTERNATIVE_INTERPRETATIONS_SCHEMA_V1",
    "CLASSIFICATION_PROMPT_V1",
    "CLASSIFICATION_SCHEMA_V1",
    "CLASSIFICATION_SYSTEM_V1",
    "COMPARATIVE_PROMPT_V1",
    "COMPARISON_DIVIDER_V1",
    "COMPARISON_HEADER_V1",
    "COMPARISON_ROW_V1",
    "COMPLIANCE_PROMPT_V1",
    "CONTEXT_HEADER_V1",
    "CONTEXT_PASSAGE_V1",
    "CONTEXT_PATHS_HEADER_V1",
    "CONTEXT_PATH_V1",
    "DEGRADED_ANSWER_V1",
    "DIAGNOSTIC_PROMPT_V1",
    "DIGEST_BULLET_V1",
    "DIGEST_EMPTY_V1",
    "DIGEST_SECTION_V1",
    "EXTRACTION_SYSTEM_V1",
    "EXTRACTIVE_FALLBACK_V1",
    "EXTRACTIVE_PASSAGE_V1",
    "FACTUAL_PROMPT_V1",
    "GROUNDED_SYSTEM_V1",
    "KNOWLEDGE_GAP_PROMPT_V1",
    "NO_EVIDENCE_ANSWER_V1",
    "NO_EVIDENCE_SEARCH_LINE_V1",
    "NO_EVIDENCE_SUGGESTION_V1",
    "PREDICTIVE_PROMPT_V1",
    "PROCEDURAL_PROMPT_V1",
    "PROCEDURE_ANSWER_V1",
    "PROCEDURE_DURATION_V1",
    "PROCEDURE_SAFETY_NOTE_V1",
    "PROCEDURE_SAFETY_V1",
    "PROCEDURE_STEP_V1",
    "PROCEDURE_TOOLS_V1",
    "PROMPT_VERSION",
    "RECOMMENDED_ACTIONS_PROMPT_V1",
    "RECOMMENDED_ACTIONS_SCHEMA_V1",
    "render",
]
