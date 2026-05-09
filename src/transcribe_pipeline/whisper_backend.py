from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import TranscriptResult, TranscriptSegment
from .processing import build_transcript_result


@dataclass(frozen=True)
class WhisperSettings:
    model_size: str = "tiny.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = False
    cpu_threads: int = 0


class FasterWhisperTranscriber:
    backend_name = "faster-whisper"

    def __init__(self, settings: WhisperSettings) -> None:
        self.settings = settings

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install dependencies with "
                "`python -m pip install -e .`."
            ) from exc

        model_kwargs = {
            "device": self.settings.device,
            "compute_type": self.settings.compute_type,
        }
        if self.settings.cpu_threads > 0:
            model_kwargs["cpu_threads"] = self.settings.cpu_threads

        model = WhisperModel(self.settings.model_size, **model_kwargs)
        raw_segments, info = model.transcribe(
            str(audio_path),
            language=self.settings.language,
            beam_size=self.settings.beam_size,
            vad_filter=self.settings.vad_filter,
        )

        segments = [
            TranscriptSegment(start=segment.start, end=segment.end, text=segment.text)
            for segment in raw_segments
        ]

        return build_transcript_result(
            source=audio_path,
            backend=self.backend_name,
            model_size=self.settings.model_size,
            language=getattr(info, "language", self.settings.language),
            language_probability=getattr(info, "language_probability", None),
            duration_seconds=getattr(info, "duration", None),
            segments=segments,
        )
