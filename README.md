# realtime-asr-v2

A real-time Automatic Speech Recognition (ASR) experiment built for [Cursor Meetup Taipei](https://lu.ma/cursor-meetup-taipei) (June 2025), originally intended to power bilingual subtitles during the talk.

This is the second iteration. v1 was built on Google Cloud Speech-to-Text and Google Translation APIs. v2 replaces the cloud pipeline with [MediaTek Research's Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25), a Whisper-based model fine-tuned for Mandarin and Taiwanese Hokkien. In concept it is similar to [Typeless](https://www.typeless.com/) — audio in, transcript out — but without any downstream cleanup model.

## How it works

The program runs a continuous loop:

1. Records from the default microphone until 1 second of silence is detected.
2. Feeds the captured waveform into the Breeze-ASR-25 pipeline (running locally via `transformers`).
3. Prints the transcribed text and processing time, then loops back to step 1.

On Apple Silicon, inference runs on the MPS backend. On other machines it falls back to CPU.

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- A working microphone
- ~1.5 GB disk space for the model weights (downloaded on first run)

## Usage

```bash
uv run python3 main.py
```

The first run downloads the Breeze-ASR-25 model weights from Hugging Face. All subsequent runs are fully offline.

Press `Ctrl+C` to stop.

## Branches

| Branch | Notes |
|--------|-------|
| `master` | Original experiment files, committed as-is |
| `working` | Verified to compile and run on Apple M3 Max |

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` + `torchaudio` | Tensor ops and audio resampling |
| `transformers` | Breeze-ASR-25 model and Whisper pipeline |
| `pyaudio` | Microphone capture |
| `numpy` | Audio buffer manipulation |
