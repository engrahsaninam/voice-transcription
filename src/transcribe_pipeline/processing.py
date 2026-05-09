from __future__ import annotations

import re
from pathlib import Path

from .models import TranscriptMetadata, TranscriptResult, TranscriptSegment


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_transcript_result(
    *,
    source: Path,
    backend: str,
    model_size: str,
    language: str | None,
    language_probability: float | None,
    duration_seconds: float | None,
    segments: list[TranscriptSegment],
) -> TranscriptResult:
    cleaned_segments = [
        TranscriptSegment(
            start=round(segment.start, 3),
            end=round(segment.end, 3),
            text=normalize_text(segment.text),
        )
        for segment in segments
        if normalize_text(segment.text)
    ]
    text = normalize_text(" ".join(segment.text for segment in cleaned_segments))

    return TranscriptResult(
        source=str(Path(source).resolve()),
        backend=backend,
        model_size=model_size,
        language=language,
        language_probability=language_probability,
        text=text,
        segments=cleaned_segments,
        metadata=TranscriptMetadata(
            duration_seconds=round(duration_seconds, 3) if duration_seconds is not None else None,
            word_count=len(text.split()) if text else 0,
            character_count=len(text),
            segment_count=len(cleaned_segments),
            processing_status="completed",
        ),
    )
