from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import TranscriptResult


def result_to_dict(result: TranscriptResult) -> dict[str, Any]:
    return asdict(result)


def write_json(result: TranscriptResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result_to_dict(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(result: TranscriptResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.text + "\n", encoding="utf-8")
