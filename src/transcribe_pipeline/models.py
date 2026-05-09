from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptMetadata:
    duration_seconds: float | None
    word_count: int
    character_count: int
    segment_count: int
    processing_status: str


@dataclass(frozen=True)
class TranscriptResult:
    source: str
    backend: str
    model_size: str
    language: str | None
    language_probability: float | None
    text: str
    segments: list[TranscriptSegment]
    metadata: TranscriptMetadata
