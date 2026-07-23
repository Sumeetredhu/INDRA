"""Speech-to-text for the hands-free field interface.

Whisper is an *optional* dependency and a large one. The rule that governs this module is
CLAUDE.md rule 6: a missing wheel degrades one modality and never breaks a request. So there are two
engines behind :class:`indra.core.contracts.SpeechToText`:

* :class:`WhisperSTT` — real transcription, loaded lazily so importing this module costs nothing and
  works on a machine with no model weights and no ``ffmpeg``.
* :class:`NullSTT` — accepts a text transcript supplied alongside the audio (the mobile client can
  use the browser's own Web Speech API, or the technician can type). It also recognises the case
  where the "audio" payload is in fact UTF-8 text, which is what the demo and the tests post.

``WhisperSTT`` delegates to ``NullSTT`` whenever Whisper is absent, so callers hold one object and
the voice endpoint never returns 500 because a wheel is missing.
"""

from __future__ import annotations

import asyncio
import math
import tempfile
from pathlib import Path
from typing import Any, Final, Protocol

from indra.core.config import Settings
from indra.core.exceptions import SpeechError
from indra.core.logging import get_logger

try:
    import whisper

    _HAS_WHISPER = True
except ImportError:  # pragma: no cover - optional dependency
    whisper = None  # type: ignore[assignment]
    _HAS_WHISPER = False

logger = get_logger(__name__)

#: Confidence attached to a transcript the caller supplied directly. It is not a recognition result,
#: it is the text itself, so there is no recognition uncertainty to report.
_SUPPLIED_TRANSCRIPT_CONFIDENCE: Final[float] = 1.0

#: Confidence for a payload that decoded cleanly as UTF-8 text. Slightly below 1.0 because the
#: decode is an inference about the caller's intent rather than an explicit declaration.
_DECODED_TEXT_CONFIDENCE: Final[float] = 0.95

#: Magic prefixes of the container formats a phone actually records. Used to avoid mistaking binary
#: audio for text; the list is a fast-path negative check, not format validation.
_AUDIO_MAGIC: Final[tuple[bytes, ...]] = (
    b"RIFF", b"OggS", b"fLaC", b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\x1aE\xdf\xa3",
)

#: A payload is only treated as text when nearly all of it is printable. Control-character soup that
#: happens to be valid UTF-8 is audio, not a transcript.
_PRINTABLE_RATIO_MIN: Final[float] = 0.9

#: Whisper's own no-speech probability above which the segment is discarded as silence.
_NO_SPEECH_MAX: Final[float] = 0.6


class TranscriptCapableSTT(Protocol):
    """The mobile agent's internal STT surface.

    A superset of :class:`indra.core.contracts.SpeechToText`: the extra ``transcript`` keyword is how
    a client that already has text (browser speech API, manual entry, an offline queue replay) feeds
    the same pipeline. It is optional with a default, so this protocol still satisfies the contract.
    """

    name: str

    @property
    def available(self) -> bool:
        """True when real speech recognition is possible in this process."""
        ...

    async def transcribe(
        self,
        audio: bytes,
        *,
        language_hint: str | None = None,
        transcript: str | None = None,
    ) -> tuple[str, str, float]:
        """Return ``(transcript, detected_language, confidence)``."""
        ...


def looks_like_text(payload: bytes) -> str | None:
    """Return the decoded string when ``payload`` is UTF-8 text, else ``None``.

    The mobile client posts audio; the demo, the tests, and the offline replay path post the
    transcript directly. Sniffing rather than requiring a second endpoint keeps one code path warm.
    """
    if not payload:
        return None
    if any(payload.startswith(magic) for magic in _AUDIO_MAGIC):
        return None
    if len(payload) > 4 and payload[4:8] == b"ftyp":  # MP4/M4A container
        return None
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not decoded.strip():
        return None
    printable = sum(1 for ch in decoded if ch.isprintable() or ch in "\n\r\t")
    if printable / len(decoded) < _PRINTABLE_RATIO_MIN:
        return None
    return decoded.strip()


class NullSTT:
    """Degraded speech-to-text: no model, no network, never an exception.

    Order of preference: an explicit ``transcript`` argument, then a payload that decodes as text.
    Failing both it returns an empty transcript with zero confidence, and the voice pipeline turns
    that into an honest "I could not hear that" rather than an error page.
    """

    name: Final[str] = "null_stt"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def available(self) -> bool:
        return False

    async def transcribe(
        self,
        audio: bytes,
        *,
        language_hint: str | None = None,
        transcript: str | None = None,
    ) -> tuple[str, str, float]:
        """Return ``(transcript, language, confidence)`` without ever raising."""
        language = language_hint or self._settings.default_language
        if transcript and transcript.strip():
            logger.info(
                "using caller-supplied transcript; speech recognition is unavailable",
                extra={"chars": len(transcript), "language_hint": language_hint},
            )
            return transcript.strip(), language, _SUPPLIED_TRANSCRIPT_CONFIDENCE

        decoded = looks_like_text(audio)
        if decoded is not None:
            logger.info(
                "voice payload decoded as text; treating it as the transcript",
                extra={"chars": len(decoded)},
            )
            return decoded, language, _DECODED_TEXT_CONFIDENCE

        logger.warning(
            "no speech recognition available and no transcript supplied; returning empty transcript",
            extra={"audio_bytes": len(audio), "install_hint": "pip install openai-whisper"},
        )
        return "", language, 0.0


class WhisperSTT:
    """Whisper-backed :class:`indra.core.contracts.SpeechToText`.

    The model is loaded lazily on first use, inside a thread, under a lock — loading is seconds of
    CPU and must not block the event loop or happen twice. When Whisper is not installed every call
    is delegated to :class:`NullSTT`, which is why the mobile router can hold this object
    unconditionally.
    """

    name: Final[str] = "whisper"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_name = settings.whisper_model
        self._model: Any | None = None
        self._load_failed = False
        self._lock = asyncio.Lock()
        self._fallback = NullSTT(settings)

    @property
    def available(self) -> bool:
        return _HAS_WHISPER and not self._load_failed

    async def transcribe(
        self,
        audio: bytes,
        *,
        language_hint: str | None = None,
        transcript: str | None = None,
    ) -> tuple[str, str, float]:
        """Transcribe ``audio``.

        A caller-supplied ``transcript`` always wins: it costs nothing and it is what an offline
        replay or a browser-side recogniser already produced.

        Raises:
            SpeechError: Whisper is installed but failed on this payload. The voice pipeline catches
                this and degrades to :class:`NullSTT` rather than failing the request.
        """
        if transcript and transcript.strip():
            return await self._fallback.transcribe(
                audio, language_hint=language_hint, transcript=transcript
            )
        if not self.available:
            return await self._fallback.transcribe(audio, language_hint=language_hint)
        if not audio:
            raise SpeechError(
                "Empty audio payload. Record at least one second of speech before submitting.",
                context={"engine": self.name},
            )

        model = await self._ensure_model()
        if model is None:
            return await self._fallback.transcribe(audio, language_hint=language_hint)

        try:
            return await asyncio.to_thread(self._transcribe_sync, model, audio, language_hint)
        except SpeechError:
            raise
        except Exception as exc:  # external boundary: whisper + ffmpeg + filesystem
            raise SpeechError(
                "Whisper failed to transcribe the audio. Check that ffmpeg is on PATH and that the "
                "recording is a supported container (wav, mp3, m4a, ogg, webm).",
                context={"engine": self.name, "model": self._model_name, "audio_bytes": len(audio)},
                cause=exc,
            ) from exc

    async def _ensure_model(self) -> Any | None:
        """Load the Whisper model once. Returns ``None`` when loading is impossible."""
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                return None
            try:
                self._model = await asyncio.to_thread(whisper.load_model, self._model_name)
            except Exception as exc:  # external boundary: model download + torch init
                self._load_failed = True
                logger.warning(
                    "whisper model failed to load; falling back to text transcripts",
                    extra={"model": self._model_name, "error": str(exc)},
                )
                return None
            logger.info("whisper model loaded", extra={"model": self._model_name})
            return self._model

    def _transcribe_sync(self, model: Any, audio: bytes, language_hint: str | None) -> tuple[str, str, float]:
        """Blocking transcription. Always called through :func:`asyncio.to_thread`."""
        supported = set(self._settings.supported_languages)
        language = language_hint if language_hint in supported else None
        with tempfile.TemporaryDirectory(prefix="indra_stt_") as tmp:
            path = Path(tmp) / "utterance.audio"
            path.write_bytes(audio)
            result: dict[str, Any] = model.transcribe(
                str(path),
                language=language,
                task="transcribe",
                fp16=False,
            )
        text = str(result.get("text", "")).strip()
        detected = str(result.get("language") or language or self._settings.default_language)
        if detected not in supported:
            logger.warning(
                "whisper detected an unsupported language",
                extra={"detected_language": detected, "supported": sorted(supported)},
            )
        return text, detected, _segment_confidence(result)


def _segment_confidence(result: dict[str, Any]) -> float:
    """Derive a 0–1 confidence from Whisper's per-segment log-probabilities.

    Whisper reports ``avg_logprob`` (mean token log-probability) and ``no_speech_prob`` per segment.
    ``exp(avg_logprob)`` converts the former back into a probability; segments the model believes are
    silence are dropped before averaging so a pause at the end of a recording does not drag an
    otherwise clean transcription down.
    """
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return 0.0
    scores: list[float] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if float(segment.get("no_speech_prob", 0.0)) > _NO_SPEECH_MAX:
            continue
        avg_logprob = segment.get("avg_logprob")
        if avg_logprob is None:
            continue
        scores.append(math.exp(min(0.0, float(avg_logprob))))
    if not scores:
        return 0.0
    return round(max(0.0, min(1.0, sum(scores) / len(scores))), 4)


def build_stt(settings: Settings) -> TranscriptCapableSTT:
    """Return the best speech-to-text engine available in this process.

    Always returns an object: :class:`WhisperSTT` when the wheel is importable (it self-degrades if
    the weights cannot be fetched), otherwise :class:`NullSTT`.
    """
    if _HAS_WHISPER:
        logger.info("speech-to-text engine selected", extra={"engine": "whisper", "model": settings.whisper_model})
        return WhisperSTT(settings)
    logger.warning(
        "whisper is not installed; voice queries accept a text transcript instead",
        extra={"engine": "null_stt", "install_hint": "pip install openai-whisper"},
    )
    return NullSTT(settings)


__all__ = ["NullSTT", "TranscriptCapableSTT", "WhisperSTT", "build_stt", "looks_like_text"]
