"""Text-to-speech for the hands-free field interface.

Three engines behind :class:`indra.core.contracts.TextToSpeech`, all optional except the last:

* :class:`GTTSEngine` — Google Translate TTS. Best Indic voices, but it is a **network** call, so it
  is skipped entirely in offline mode.
* :class:`Pyttsx3Engine` — the OS speech synthesiser. Works offline; Indic voice quality depends on
  what the host has installed.
* :class:`NullTTS` — returns the text itself with a ``text/plain`` mime type. The client renders it
  on screen. A technician who can read the answer is inconvenienced; a 500 stops the job.

:func:`build_tts` composes them into a :class:`TTSChain` that tries each in turn, so a dead network
or a missing wheel costs one fallback hop and a log line.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Any, Final, Protocol, Sequence

from indra.core.config import Settings
from indra.core.exceptions import SpeechError
from indra.core.logging import get_logger

try:
    from gtts import gTTS

    _HAS_GTTS = True
except ImportError:  # pragma: no cover - optional dependency
    gTTS = None  # type: ignore[assignment,misc]
    _HAS_GTTS = False

try:
    import pyttsx3

    _HAS_PYTTSX3 = True
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None  # type: ignore[assignment]
    _HAS_PYTTSX3 = False

logger = get_logger(__name__)

MIME_MP3: Final[str] = "audio/mpeg"
MIME_WAV: Final[str] = "audio/wav"
MIME_TEXT: Final[str] = "text/plain; charset=utf-8"

#: gTTS locale codes for the languages INDRA supports. Anything unmapped is spoken in English rather
#: than failing — a technician hearing an English sentence is better served than one hearing nothing.
_GTTS_LOCALES: Final[dict[str, str]] = {"en": "en", "hi": "hi", "ta": "ta", "kn": "kn", "mr": "mr"}

#: Hard ceiling on synthesis input. Long answers are truncated at a sentence boundary because a
#: two-minute audio clip is unusable on a plant floor and costs the whole latency budget.
_MAX_SPEECH_CHARS: Final[int] = 900

_SENTENCE_ENDINGS: Final[tuple[str, ...]] = (". ", "। ", "! ", "? ", "\n")


class SpeechEngine(Protocol):
    """The mobile agent's internal TTS surface — the contract plus an availability probe."""

    name: str

    @property
    def available(self) -> bool:
        ...

    async def synthesize(self, text: str, *, language: str = "en") -> tuple[bytes, str]:
        """Return ``(audio_bytes, mime_type)``."""
        ...


def clip_for_speech(text: str, *, limit: int = _MAX_SPEECH_CHARS) -> str:
    """Trim ``text`` to the last sentence boundary within ``limit`` characters.

    Cutting mid-word produces audio that sounds broken and erodes trust in the answer; cutting at a
    sentence boundary sounds deliberate.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    window = cleaned[:limit]
    cut = max(window.rfind(ending) for ending in _SENTENCE_ENDINGS)
    if cut <= 0:
        cut = window.rfind(" ")
    return (window[: cut + 1] if cut > 0 else window).strip()


class NullTTS:
    """No synthesis: hand the text back and let the client display it."""

    name: Final[str] = "null_tts"

    @property
    def available(self) -> bool:
        return True

    async def synthesize(self, text: str, *, language: str = "en") -> tuple[bytes, str]:
        """Return the UTF-8 bytes of ``text`` with a ``text/plain`` mime type."""
        logger.info(
            "text-to-speech unavailable; returning text for on-screen rendering",
            extra={"language": language, "chars": len(text)},
        )
        return text.encode("utf-8"), MIME_TEXT


class GTTSEngine:
    """Google Translate text-to-speech. Network-bound, best Indic voices."""

    name: Final[str] = "gtts"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def available(self) -> bool:
        return _HAS_GTTS and not self._settings.offline_mode

    async def synthesize(self, text: str, *, language: str = "en") -> tuple[bytes, str]:
        """Synthesize ``text`` to MP3.

        Raises:
            SpeechError: the wheel is missing, the process is offline, or the request failed.
        """
        if not self.available:
            raise SpeechError(
                "gTTS is unavailable (not installed, or INDRA is running in offline mode). "
                "Install gTTS or configure INDRA_TTS_ENGINE=pyttsx3 for an offline voice.",
                context={"engine": self.name, "offline_mode": self._settings.offline_mode},
            )
        payload = clip_for_speech(text)
        if not payload:
            raise SpeechError("Nothing to speak: the text was empty.", context={"engine": self.name})
        locale = _GTTS_LOCALES.get(language, _GTTS_LOCALES[self._settings.default_language])
        try:
            audio = await asyncio.to_thread(self._synthesize_sync, payload, locale)
        except Exception as exc:  # external boundary: HTTP call inside gTTS
            raise SpeechError(
                "gTTS synthesis failed. It needs outbound internet access; fall back to pyttsx3 or "
                "return the answer as text.",
                context={"engine": self.name, "language": language, "locale": locale},
                cause=exc,
            ) from exc
        return audio, MIME_MP3

    @staticmethod
    def _synthesize_sync(text: str, locale: str) -> bytes:
        buffer = io.BytesIO()
        gTTS(text=text, lang=locale).write_to_fp(buffer)
        return buffer.getvalue()


class Pyttsx3Engine:
    """Offline OS speech synthesis (SAPI5 on Windows, NSSpeechSynthesizer on macOS, eSpeak on Linux).

    ``pyttsx3`` drivers are not thread-safe and several of them keep global state, so every call
    builds and disposes its own engine inside one worker thread, serialised by a lock.
    """

    name: Final[str] = "pyttsx3"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return _HAS_PYTTSX3

    async def synthesize(self, text: str, *, language: str = "en") -> tuple[bytes, str]:
        """Synthesize ``text`` to WAV using the host's speech engine.

        Raises:
            SpeechError: the wheel is missing or the platform driver failed.
        """
        if not self.available:
            raise SpeechError(
                "pyttsx3 is not installed. Install it for offline speech, or let INDRA return the "
                "answer as text.",
                context={"engine": self.name},
            )
        payload = clip_for_speech(text)
        if not payload:
            raise SpeechError("Nothing to speak: the text was empty.", context={"engine": self.name})
        async with self._lock:
            try:
                audio = await asyncio.to_thread(self._synthesize_sync, payload, language)
            except Exception as exc:  # external boundary: platform speech driver
                raise SpeechError(
                    "pyttsx3 synthesis failed. The host may have no speech driver installed; "
                    "returning the answer as text is the correct degradation.",
                    context={"engine": self.name, "language": language},
                    cause=exc,
                ) from exc
        return audio, MIME_WAV

    def _synthesize_sync(self, text: str, language: str) -> bytes:
        engine: Any = pyttsx3.init()
        try:
            self._select_voice(engine, language)
            with tempfile.TemporaryDirectory(prefix="indra_tts_") as tmp:
                path = Path(tmp) / "answer.wav"
                engine.save_to_file(text, str(path))
                engine.runAndWait()
                return path.read_bytes()
        finally:
            try:
                engine.stop()
            except Exception:  # pragma: no cover - driver teardown is best-effort
                logger.debug("pyttsx3 engine stop failed", extra={"engine": self.name})

    @staticmethod
    def _select_voice(engine: Any, language: str) -> None:
        """Pick a voice whose id or languages mention ``language``; keep the default otherwise."""
        try:
            voices = engine.getProperty("voices") or []
        except Exception:  # pragma: no cover - driver dependent
            return
        needle = language.lower()
        for voice in voices:
            haystack = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
            languages = getattr(voice, "languages", []) or []
            tags = " ".join(
                item.decode("utf-8", "ignore") if isinstance(item, bytes) else str(item) for item in languages
            ).lower()
            if needle in haystack or needle in tags:
                engine.setProperty("voice", voice.id)
                return


class TTSChain:
    """Ordered fallback across speech engines. Always produces output.

    The final element is :class:`NullTTS`, so :meth:`synthesize` cannot fail; the worst case is that
    the caller receives text and a ``text/plain`` mime type.
    """

    name: Final[str] = "tts_chain"

    def __init__(self, engines: Sequence[SpeechEngine]) -> None:
        self._engines: tuple[SpeechEngine, ...] = tuple(engines) or (NullTTS(),)

    @property
    def available(self) -> bool:
        return any(engine.available for engine in self._engines if not isinstance(engine, NullTTS))

    @property
    def engine_names(self) -> list[str]:
        return [engine.name for engine in self._engines]

    async def synthesize(self, text: str, *, language: str = "en") -> tuple[bytes, str]:
        """Try each engine in order; return the first success."""
        errors: list[str] = []
        for engine in self._engines:
            if not engine.available:
                continue
            try:
                return await engine.synthesize(text, language=language)
            except SpeechError as exc:
                errors.append(f"{engine.name}: {exc.message}")
                logger.warning(
                    "speech engine failed; trying the next one",
                    extra={"engine": engine.name, "language": language, "error": exc.message},
                )
        if errors:
            logger.warning("every speech engine failed", extra={"failures": errors, "language": language})
        return await NullTTS().synthesize(text, language=language)


def build_tts(settings: Settings) -> TTSChain:
    """Build the speech chain implied by ``settings.tts_engine``.

    The configured engine goes first, the other real engine second, and :class:`NullTTS` last.
    ``tts_engine="none"`` skips synthesis entirely and returns text.
    """
    preference = settings.tts_engine
    if preference == "none":
        logger.info("text-to-speech disabled by configuration", extra={"tts_engine": preference})
        return TTSChain([NullTTS()])

    gtts = GTTSEngine(settings)
    offline = Pyttsx3Engine(settings)
    ordered: list[SpeechEngine] = [offline, gtts] if preference == "pyttsx3" else [gtts, offline]
    ordered.append(NullTTS())

    chain = TTSChain(ordered)
    logger.info(
        "text-to-speech chain built",
        extra={
            "preferred": preference,
            "chain": chain.engine_names,
            "gtts_available": gtts.available,
            "pyttsx3_available": offline.available,
        },
    )
    return chain


__all__ = [
    "GTTSEngine",
    "MIME_MP3",
    "MIME_TEXT",
    "MIME_WAV",
    "NullTTS",
    "Pyttsx3Engine",
    "SpeechEngine",
    "TTSChain",
    "build_tts",
    "clip_for_speech",
]
