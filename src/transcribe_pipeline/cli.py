from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .outputs import write_json, write_text
from .whisper_backend import FasterWhisperTranscriber, WhisperSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe-pipeline",
        description="Transcribe audio locally with faster-whisper and write downstream-ready output.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="transcribe one audio file")
    transcribe.add_argument("audio", type=Path, help="path to an audio file")
    transcribe.add_argument(
        "--out",
        type=Path,
        help="output file path; stdout is used when omitted",
    )
    transcribe.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="output format",
    )
    transcribe.add_argument(
        "--model-size",
        default="tiny.en",
        help="faster-whisper model size or local model path",
    )
    transcribe.add_argument("--device", default="cpu", help="runtime device, for example cpu or cuda")
    transcribe.add_argument("--compute-type", default="int8", help="model compute type")
    transcribe.add_argument("--language", help="optional language hint such as en")
    transcribe.add_argument("--beam-size", type=int, default=5, help="beam size used by Whisper")
    transcribe.add_argument(
        "--vad-filter",
        action="store_true",
        help="enable voice activity filtering before transcription",
    )
    transcribe.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU thread count passed to faster-whisper; 0 lets the backend choose",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "transcribe":
        return run_transcribe(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


def run_transcribe(args: argparse.Namespace) -> int:
    audio_path = args.audio
    if not audio_path.exists() or not audio_path.is_file():
        print(f"Input audio file does not exist: {audio_path}", file=sys.stderr)
        return 2

    settings = WhisperSettings(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        cpu_threads=args.cpu_threads,
    )
    transcriber = FasterWhisperTranscriber(settings)

    try:
        result = transcriber.transcribe(audio_path)
    except Exception as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return 1

    if args.out:
        if args.format == "json":
            write_json(result, args.out)
        else:
            write_text(result, args.out)
    elif args.format == "json":
        from .outputs import result_to_dict
        import json

        print(json.dumps(result_to_dict(result), indent=2, ensure_ascii=False))
    else:
        print(result.text)

    return 0
