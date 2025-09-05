# Call Analysis Pipeline

Professional, modular pipeline for transcription, diarization, sentiment/intent analysis, tonality (audio emotion), summarization and RAG-based follow-up recommendations for call center audio.

This README explains the project, how to run it locally in **hardcoded** mode (no CLI), how files are organized, evaluation steps.

---

## Table of contents

- [Project overview](#project-overview)
- [Key features](#key-features)
- [Repository layout](#repository-layout)
- [Quick start (recommended)](#quick-start-recommended)
- [Environment & dependencies](#environment--dependencies)
- [Configuration / API keys](#configuration--api-keys)
- [Run the pipeline (hardcoded mode)](#run-the-pipeline-hardcoded-mode)
- [Run evaluation](#run-evaluation)
- [Output files (what to expect)](#output-files-what-to-expect)
- [Troubleshooting & tips](#troubleshooting--tips)
- [Notes on reproducibility & performance](#notes-on-reproducibility--performance)
- [Repository policies / large files (git / LFS)](#repository-policies--large-files-git--lfs)
- [License & attribution](#license--attribution)

---

## Project overview

This repository contains a modular call-analysis pipeline that performs the following tasks:

1. **Speech-to-text (STT)** using WhisperX (transcription + optional word-level alignment).
2. **Speaker diarization** and assignment of words/segments to speakers.
3. **Sentiment analysis** (text-level) using a Hugging Face sequence classification model.
4. **Tonality / audio-emotion** detection using a pre-trained audio classification pipeline.
5. **Intent detection** using a zero-shot classifier.
6. **Summarization** via an LLM (configurable; summarization is skipped if no API key).
7. **RAG (retrieval-augmented generation)** — creates a FAISS index over `docs.txt` and uses Cohere embeddings if the API key is provided.

The code is modular: each model or stage is loaded lazily and can be skipped if environment variables or API keys are not set.

---

## Key features

- Modular, class-based architecture (`CallAnalysisPipeline`) so stages can be reused or unit-tested.
- Automatic export of CSVs used by the evaluation script (`evaluate_pipeline.py`).
- Safe guards for missing API keys: LLM / RAG / diarization are skipped when not configured.

---

## Repository layout

```
Project root/
├─ .env                         # environment variables (not checked into git)
├─ main.py                      # pipeline script (hardcoded mode)
├─ evaluate_pipeline.py         # evaluation script (reads CSVs and prints reports)
├─ csv files/                   # CSV input / output directory (created by main.py)
│  ├─ references.csv
│  ├─ pipeline_transcripts.csv
│  ├─ segments_pred.csv
│  ├─ summary_refs.csv
│  ├─ summary_preds.csv
│  ├─ labels_gold.csv (optional)
│  └─ evaluation_report.json
├─ Text Output files/           # plain text output directory (created by main.py)
│  ├─ summary.txt
│  └─ next_steps.txt
├─ docs.txt                     # policy/FAQ docs used for RAG
├─ audio files/                 # your .mp3 / .wav files (avoid committing large files)
└─ requirements.txt             # recommended Python packages
```

> Note: The pipeline writes outputs into `csv files/` and `Text Output files/` by default. This keeps evaluation artifacts in one place.

---

## Quick start (recommended)

1. Create and activate a Python virtual environment:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Install dependencies (see `requirements.txt`). Example minimal install:

```bash
pip install -r requirements.txt
```

3. Add API keys to `.env` — see next section.

4. Place an audio file you want to analyze in the project root or `audio files/` and update `AUDIO_FILE` in `main.py` if necessary.

5. Run the pipeline:

```bash
python main.py
```

7. Run the evaluation script (reads from `csv files/`):

```bash
python evaluate_pipeline.py
```

---

## Environment & dependencies

- Python 3.8+ recommended.
- PyTorch & torchaudio must be compatible. For CPU-only machines, use the CPU wheels from PyTorch website.
- `ffmpeg` must be installed system-wide and accessible via PATH.

**Notes on transformers / model sizes**

- The pipeline uses WhisperX (`medium` by default). If you have limited resources, change `load_transcription(..., model_name="small")` in `main.py`.
- Zero-shot/instruction models and audio models may require large downloads on first run.

---

## Configuration / API keys

Create a `.env` file in repo root with the following variables if you want to enable optional features:

```
HF_TOKEN=hf_xxx            # Hugging Face token for WhisperX diarization (optional)
OPENAI_API_KEY=sk-xxx      # or OpenRouter token for summarization (optional)
COHERE_API_KEY=cohere-xxx  # for Cohere embeddings (optional)
```

If these keys are not set, summarization / diarization / RAG steps will be skipped and the pipeline will still run other analyses.

---

## Run the pipeline

Open `main.py` and update these variables at the bottom if needed:

```python
AUDIO_FILE = "ssvid.net---Hindi-Attending-Phone-Calls-Complaint-Call.mp3"
DOCS_PATH = "docs.txt"
DEVICE = "cpu"  # or "cuda" if you have a GPU and proper torch install
BATCH_SIZE = 16
COMPUTE_TYPE = "int8"  # whisperx compute hint
```

Then run:

```bash
python main.py
```

The script will:

- transcribe the audio
- run sentiment, tonality and intent analysis per segment
- attempt summarization (if OPENAI_API_KEY provided)
- (optionally) build RAG index (if COHERE_API_KEY provided)
- export CSVs to `csv files/` and text files to `Text Output files/`

---

## Run evaluation

The repo contains `evaluate_pipeline.py` which reads CSVs from `csv files/` and computes:

- WER (per-call mean + median), using `references.csv` and `pipeline_transcripts.csv`
- Classification reports for sentiment & intent (requires `labels_gold.csv` and `segments_pred.csv`)
- ROUGE scores for summaries (`summary_refs.csv` and `summary_preds.csv`)

Usage:

```bash
python evaluate_pipeline.py
```

The script writes `csv files/evaluation_report.json` and prints a human-readable summary in terminal.

---

## Output files (what to expect)

Files produced by `main.py` (paths relative to repo root):

- `csv files/segments_pred.csv` — per-segment predictions (segment_id, call_id, start, end, speaker, text, sentiment_pred, intent_pred, tonality_pred)
- `csv files/pipeline_transcripts.csv` — per-call transcript used for WER
- `csv files/summary_preds.csv` — per-call predicted summary used for ROUGE
- `csv files/analysis_output.csv` — full dataframe produced by pipeline
- `csv files/labels_combined.csv` — (optional) merged gold+preds if `labels_gold.csv` is present
- `Text Output files/summary.txt` — saved human-readable summary
- `Text Output files/next_steps.txt` — saved next-steps suggested by the LLM

---

## Troubleshooting & tips

- **torchaudio or torch import errors**: ensure `torch` and `torchaudio` versions are compatible. See the official PyTorch installation page for platform-specific instructions.
- **ffmpeg errors**: install ffmpeg system package and verify with `ffmpeg -version`.
- **Model downloads hang**: ensure internet access and enough disk space (several GBs).
- **Short evaluation reports**: if you only evaluate on 3–5 segments, classification metrics will be noisy — use more samples for stable metrics.

---

## LICENSE

MIT License
@copyright 2025 rohitrebel
