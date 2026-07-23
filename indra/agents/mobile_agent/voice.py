"""The multilingual voice round trip.

``audio → STT → detect → translate to English → Copilot → translate the answer text → TTS``

Two decisions in here are the whole of ``docs/DECISIONS.md`` **D11**:

1. **Plant tags are masked before translation and restored after.** Handled inside
   :mod:`indra.agents.mobile_agent.translation`; this module just relies on it.
2. **INDRA reasons in English and translates only at render time.** The :class:`Answer` that comes
   back from the Copilot — its reasoning chain, its source snippets, its recommended actions — stays
   in English. Only ``answer_text`` is translated, into
   :attr:`~indra.core.models.VoiceQueryResponse.spoken_text`. Translating a ten-step reasoning chain
   costs ten more LLM round trips, and a technician standing next to a running pump does not have
   that time. The frontend translates a step on demand when the operator opens the explain panel.

Every stage degrades independently. No Whisper, no network, no Copilot, no TTS — the endpoint still
returns a :class:`VoiceQueryResponse` with an honest account of what was lost.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from indra.core.config import Settings
from indra.core.exceptions import IndraError, SpeechError, TranslationError
from indra.core.logging import get_logger
from indra.core.models import (
    Answer,
    Confidence,
    QueryRequest,
    QueryType,
    Severity,
    UncertaintyFlag,
    UncertaintySource,
    VoiceQueryResponse,
)

from indra.agents.mobile_agent.stt import NullSTT, TranscriptCapableSTT
from indra.agents.mobile_agent.translation import ScriptTranslator
from indra.agents.mobile_agent.tts import MIME_TEXT, TTSChain

if TYPE_CHECKING:  # pragma: no cover - typing only
    from indra.core.contracts import CopilotService

logger = get_logger(__name__)

#: Confidence assigned to a voice answer we could not route to the Copilot. Deliberately low: the
#: technician must see immediately that this is a plumbing failure, not plant knowledge.
_DEGRADED_CONFIDENCE: Final[float] = 0.05

#: Below this STT confidence the transcript is surfaced with a "please confirm" caveat. Whisper's
#: exp(avg_logprob) sits around 0.6-0.9 on clean speech and collapses on plant noise, so this is the
#: line between "act on it" and "read it back to me".
_STT_CONFIDENCE_WARN: Final[float] = 0.55

_EMPTY_TRANSCRIPT_MESSAGE: Final[str] = (
    "I could not make out any speech in that recording. Speak again closer to the microphone, or "
    "type the question instead."
)

_NO_COPILOT_MESSAGE: Final[str] = (
    "The reasoning service is not reachable from this device right now. Your question was captured "
    "and can be replayed once connectivity returns."
)


@dataclass(slots=True)
class VoiceStageTimings:
    """Per-stage latency, logged and useful when tuning the field experience."""

    stt_ms: float = 0.0
    detect_ms: float = 0.0
    translate_in_ms: float = 0.0
    copilot_ms: float = 0.0
    translate_out_ms: float = 0.0
    tts_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return round(
            self.stt_ms + self.detect_ms + self.translate_in_ms
            + self.copilot_ms + self.translate_out_ms + self.tts_ms,
            2,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "stt_ms": round(self.stt_ms, 2),
            "detect_ms": round(self.detect_ms, 2),
            "translate_in_ms": round(self.translate_in_ms, 2),
            "copilot_ms": round(self.copilot_ms, 2),
            "translate_out_ms": round(self.translate_out_ms, 2),
            "tts_ms": round(self.tts_ms, 2),
            "total_ms": self.total_ms,
        }


@dataclass(slots=True)
class _PipelineState:
    """Mutable working set for one voice query."""

    transcript: str = ""
    language: str = "en"
    stt_confidence: float = 0.0
    english_query: str | None = None
    flags: list[UncertaintyFlag] = field(default_factory=list)
    timings: VoiceStageTimings = field(default_factory=VoiceStageTimings)


class VoicePipeline:
    """Runs one voice query end to end and reports every intermediate.

    The pipeline owns no state between calls apart from its collaborators, so it is safe to share one
    instance across concurrent requests.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        stt: TranscriptCapableSTT,
        tts: TTSChain,
        translator: ScriptTranslator,
    ) -> None:
        self._settings = settings
        self._stt = stt
        self._tts = tts
        self._translator = translator
        self._fallback_stt = NullSTT(settings)
        self._copilot: CopilotService | None = None

    def bind(self, *, copilot: CopilotService | None) -> None:
        """Attach the Copilot. Called by the agent's ``bind()``; never imported directly."""
        self._copilot = copilot

    @property
    def has_copilot(self) -> bool:
        return self._copilot is not None

    async def run(
        self,
        audio: bytes,
        *,
        language_hint: str | None = None,
        equipment_tag: str | None = None,
        transcript: str | None = None,
        session_id: str | None = None,
        respond_with_audio: bool = True,
    ) -> VoiceQueryResponse:
        """Execute the full round trip. Never raises for a degraded dependency."""
        state = _PipelineState(language=self._normalise_language(language_hint))

        await self._stage_transcribe(state, audio, language_hint=language_hint, transcript=transcript)
        if not state.transcript:
            return await self._empty_transcript_response(state, respond_with_audio=respond_with_audio)

        await self._stage_detect(state, language_hint=language_hint)
        await self._stage_translate_in(state)

        answer = await self._stage_answer(
            state, equipment_tag=equipment_tag, session_id=session_id
        )
        spoken_text = await self._stage_translate_out(state, answer.answer_text)

        for flag in state.flags:
            answer.uncertainty_flags.append(flag)

        audio_b64, mime = await self._stage_speak(
            state, spoken_text, respond_with_audio=respond_with_audio
        )

        logger.info(
            "voice query complete",
            extra={
                "detected_language": state.language,
                "equipment_tag": equipment_tag,
                "translated": state.english_query is not None,
                "answer_id": answer.answer_id,
                "provider": answer.provider_used,
                **state.timings.as_dict(),
            },
        )
        return VoiceQueryResponse(
            transcript=state.transcript,
            detected_language=state.language,
            translated_query=state.english_query,
            answer=answer,
            spoken_text=spoken_text,
            audio_base64=audio_b64,
            audio_mime=mime,
            stt_confidence=_clamp(state.stt_confidence),
        )

    # -- stages ------------------------------------------------------------------------
    async def _stage_transcribe(
        self,
        state: _PipelineState,
        audio: bytes,
        *,
        language_hint: str | None,
        transcript: str | None,
    ) -> None:
        started = time.perf_counter()
        try:
            text, language, confidence = await self._stt.transcribe(
                audio, language_hint=language_hint, transcript=transcript
            )
        except SpeechError as exc:
            logger.warning(
                "speech recognition failed; falling back to the text path",
                extra={"error": exc.message, "audio_bytes": len(audio)},
            )
            text, language, confidence = await self._fallback_stt.transcribe(
                audio, language_hint=language_hint, transcript=transcript
            )
            state.flags.append(
                UncertaintyFlag(
                    source=UncertaintySource.LOW_OCR_CONFIDENCE,
                    message=f"Speech recognition was unavailable: {exc.message}",
                    severity=Severity.WARNING,
                    suggested_action="Confirm the question was understood before acting on the answer.",
                )
            )
        state.timings.stt_ms = (time.perf_counter() - started) * 1000.0
        state.transcript = text.strip()
        state.stt_confidence = _clamp(confidence)
        if language:
            state.language = self._normalise_language(language)
        if state.transcript and state.stt_confidence < _STT_CONFIDENCE_WARN:
            state.flags.append(
                UncertaintyFlag(
                    source=UncertaintySource.LOW_OCR_CONFIDENCE,
                    message=(
                        f"Speech recognised with low confidence ({state.stt_confidence:.2f}). "
                        f'Heard: "{state.transcript}"'
                    ),
                    severity=Severity.WARNING,
                    affected_claim=state.transcript,
                    suggested_action="Confirm the question before acting on the answer.",
                )
            )

    async def _stage_detect(self, state: _PipelineState, *, language_hint: str | None) -> None:
        """Trust the transcript's script over the STT engine's language guess.

        Whisper reports the language of the *audio*; on a noisy plant floor it mislabels short
        utterances often enough to matter. The transcript itself is unambiguous — Devanagari is not
        Tamil — so script detection is the authority, with the STT guess kept only as a tiebreak.
        """
        started = time.perf_counter()
        detected = await self._translator.detect(state.transcript)
        state.timings.detect_ms = (time.perf_counter() - started) * 1000.0
        if detected != state.language:
            logger.info(
                "language resolved from transcript script",
                extra={
                    "stt_language": state.language,
                    "script_language": detected,
                    "language_hint": language_hint,
                },
            )
        state.language = detected

    async def _stage_translate_in(self, state: _PipelineState) -> None:
        """Translate the question into English, with plant tags masked (D11)."""
        if state.language == self._settings.default_language:
            return
        started = time.perf_counter()
        try:
            state.english_query = await self._translator.translate(
                state.transcript, target=self._settings.default_language, source=state.language
            )
        except TranslationError as exc:
            logger.warning(
                "inbound translation failed; querying with the original text",
                extra={"source_language": state.language, "error": exc.message},
            )
            state.english_query = None
            state.flags.append(
                UncertaintyFlag(
                    source=UncertaintySource.TRANSLATED_CONTENT,
                    message=(
                        "The question could not be translated into English, so it was answered in "
                        f"{state.language}. Retrieval quality may be reduced."
                    ),
                    severity=Severity.WARNING,
                    suggested_action="Re-ask in English if the answer looks off-topic.",
                )
            )
        finally:
            state.timings.translate_in_ms = (time.perf_counter() - started) * 1000.0

    async def _stage_answer(
        self, state: _PipelineState, *, equipment_tag: str | None, session_id: str | None
    ) -> Answer:
        query_text = state.english_query or state.transcript
        request = QueryRequest(
            query=query_text[:2000],
            language=self._settings.default_language if state.english_query else state.language,
            equipment_tag=equipment_tag,
            session_id=session_id,
            channel="voice",
            include_graph_preview=False,
        )
        started = time.perf_counter()
        if self._copilot is None:
            state.timings.copilot_ms = (time.perf_counter() - started) * 1000.0
            logger.warning("voice query received but no Copilot is bound", extra={"channel": "voice"})
            return _degraded_answer(query_text, _NO_COPILOT_MESSAGE)
        try:
            answer = await self._copilot.answer(request)
        except IndraError as exc:
            state.timings.copilot_ms = (time.perf_counter() - started) * 1000.0
            logger.warning(
                "copilot failed on a voice query",
                extra={"error": exc.message, "error_code": exc.error_code},
            )
            return _degraded_answer(
                query_text,
                f"I could not complete the reasoning for that question: {exc.message}",
            )
        except Exception as exc:  # defensive: a sibling agent must not crash the voice endpoint
            state.timings.copilot_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "copilot raised an untyped error on a voice query",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return _degraded_answer(
                query_text, "I hit an unexpected internal error answering that question."
            )
        state.timings.copilot_ms = (time.perf_counter() - started) * 1000.0
        return answer

    async def _stage_translate_out(self, state: _PipelineState, answer_text: str) -> str:
        """Translate **only** the answer text back into the technician's language (D11).

        The reasoning chain, sources, and recommended actions stay in English on purpose.
        """
        if state.language == self._settings.default_language or not answer_text.strip():
            return answer_text
        started = time.perf_counter()
        try:
            spoken = await self._translator.translate(
                answer_text, target=state.language, source=self._settings.default_language
            )
        except TranslationError as exc:
            logger.warning(
                "outbound translation failed; speaking the English answer",
                extra={"target_language": state.language, "error": exc.message},
            )
            state.flags.append(
                UncertaintyFlag(
                    source=UncertaintySource.TRANSLATED_CONTENT,
                    message=(
                        f"The answer could not be translated into {state.language}; it is shown in "
                        "English."
                    ),
                    severity=Severity.LOW,
                    suggested_action="Ask a colleague to interpret, or re-ask in English.",
                )
            )
            return answer_text
        finally:
            state.timings.translate_out_ms = (time.perf_counter() - started) * 1000.0

        state.flags.append(
            UncertaintyFlag(
                source=UncertaintySource.TRANSLATED_CONTENT,
                message=(
                    f"Spoken answer was machine-translated from English into {state.language}. "
                    "Equipment tags were preserved verbatim; wording may differ from the original."
                ),
                severity=Severity.INFO,
                suggested_action="Open the explain panel to read the English reasoning chain.",
            )
        )
        return spoken

    async def _stage_speak(
        self, state: _PipelineState, spoken_text: str, *, respond_with_audio: bool
    ) -> tuple[str | None, str]:
        if not respond_with_audio or not spoken_text.strip():
            return None, MIME_TEXT
        started = time.perf_counter()
        audio, mime = await self._tts.synthesize(spoken_text, language=state.language)
        state.timings.tts_ms = (time.perf_counter() - started) * 1000.0
        if mime == MIME_TEXT:
            # NullTTS: the client renders the text; sending it back as base64 too would be noise.
            return None, MIME_TEXT
        return base64.b64encode(audio).decode("ascii"), mime

    # -- degraded paths ----------------------------------------------------------------
    async def _empty_transcript_response(
        self, state: _PipelineState, *, respond_with_audio: bool
    ) -> VoiceQueryResponse:
        answer = _degraded_answer("", _EMPTY_TRANSCRIPT_MESSAGE)
        for flag in state.flags:
            answer.uncertainty_flags.append(flag)
        audio_b64, mime = await self._stage_speak(
            state, _EMPTY_TRANSCRIPT_MESSAGE, respond_with_audio=respond_with_audio
        )
        logger.warning("voice query produced no transcript", extra=state.timings.as_dict())
        return VoiceQueryResponse(
            transcript="",
            detected_language=state.language,
            translated_query=None,
            answer=answer,
            spoken_text=_EMPTY_TRANSCRIPT_MESSAGE,
            audio_base64=audio_b64,
            audio_mime=mime,
            stt_confidence=0.0,
        )

    def _normalise_language(self, code: str | None) -> str:
        if not code:
            return self._settings.default_language
        lowered = code.strip().lower().split("-")[0]
        return lowered if lowered in set(self._settings.supported_languages) else self._settings.default_language


def _degraded_answer(query: str, message: str) -> Answer:
    """Build an :class:`Answer` that is honest about having no evidence behind it.

    ``Answer``'s own validator adds the ``SPARSE_EVIDENCE`` flag because there are no sources, which
    is exactly the signal the UI needs to render this differently from a grounded answer.
    """
    return Answer(
        query=query or "(no speech detected)",
        query_type=QueryType.FACTUAL,
        answer_text=message,
        confidence=Confidence(
            value=_DEGRADED_CONFIDENCE,
            rationale="Degraded voice path: no evidence was retrieved for this response.",
            method="heuristic",
        ),
        provider_used="none",
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["VoicePipeline", "VoiceStageTimings"]
