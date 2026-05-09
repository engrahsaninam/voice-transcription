# Transcription Pipeline CLI

Local Python CLI that transcribes audio with `faster-whisper` and returns JSON
or plain text output with timestamped segments.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Run

```powershell
.\.venv\Scripts\python.exe -m transcribe_pipeline transcribe samples\job_task_sample.wav --out output\transcript.json --format json --model-size tiny.en --language en
```

```powershell
.\.venv\Scripts\python.exe -m transcribe_pipeline transcribe samples\job_task_sample.wav --out output\transcript.txt --format text --model-size tiny.en --language en
```

The first run downloads the selected Whisper model.

## JSON Output

```json
{
  "source": "C:\\path\\to\\audio.wav",
  "backend": "faster-whisper",
  "model_size": "tiny.en",
  "language": "en",
  "language_probability": 0.99,
  "text": "Transcribed text...",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Transcribed text..."
    }
  ],
  "metadata": {
    "duration_seconds": 2.5,
    "word_count": 2,
    "character_count": 19,
    "segment_count": 1,
    "processing_status": "completed"
  }
}
```

## Tests

```powershell
$env:RUN_WHISPER_INTEGRATION = "1"
$env:WHISPER_SAMPLE_AUDIO = "samples/job_task_sample.wav"
.\.venv\Scripts\python.exe -m pytest -q
```
