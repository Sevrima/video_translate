# Video Translation System

Complete end-to-end video translation pipeline with voice cloning. Translates video content from any language to German while preserving the original speaker's voice.

**System Requirements: NVIDIA GPU with CUDA support**

## Features

### Complete Pipeline
- **Video Processing**: Extract and replace audio tracks in video files
- **Text Translation**: NLLB-200 (distilled 600M or 3.3B parameters) from Meta AI
- **Voice Cloning**: Chatterbox TTS clones original speaker's voice
- **Audio Timing**: Automatic time-stretching to match SRT segment durations
- **Language Detection**: Automatic source language detection using fasttext
- **200+ Languages**: Translate from any language to German

### Technology Stack
- **Translation**: NLLB-200 (Meta AI) - Neural machine translation
- **Voice Cloning**: Chatterbox TTS - Zero-shot voice cloning with 23 languages
- **Audio Processing**: librosa for time-stretching, pydub for manipulation
- **Video Processing**: ffmpeg for video/audio extraction and merging

## Pipeline Overview

```
Input: video.mp4 + video.srt
  ↓
Step 1: Extract audio from video (for voice reference)
  ↓
Step 2: Translate SRT (English → German using NLLB-200)
  ↓
Step 3: Generate audio segments (Chatterbox TTS with voice cloning)
  ↓
Step 4: Time-stretch & concatenate (match SRT timing with librosa)
  ↓
Step 5: Replace video audio (ffmpeg)
  ↓
Output: video_de.mp4 (translated with cloned voice)
```

## Quick Start

### 1. Setup (First Time Only)

Run the setup script to install all dependencies:

```bash
conda create -n vid python==3.10
python setup.py
```

**This installs:**
- PyTorch with CUDA 12.4 support (~2GB)
- transformers, NLLB models (download on first use ~2.5GB)
- ffmpeg-python, pydub, librosa for audio/video processing
- chatterbox-tts for voice cloning
- Takes 5-10 minutes

**Additional requirement:** Install ffmpeg system package
- Windows: Download from https://ffmpeg.org/download.html
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

### 2. Translate Complete Video

```bash
python main.py --input data/Tanzania-2.mp4 --output output/Tanzania-2-DE.mp4 --srt data/Tanzania-caption.srt
```

**Requirements:**
- Input video file (MP4, AVI, MOV, etc.)
- SRT subtitle file (must be provided with `--srt` argument)

**The pipeline automatically:**
1. Extracts audio from video for voice cloning reference
2. Translates SRT subtitles (auto-detect source language → German)
3. Generates German audio using Chatterbox TTS with cloned voice
4. Time-stretches audio segments to match SRT timing perfectly
5. Replaces original video audio with translated audio
6. Saves final translated video

## Usage

### Complete Video Translation

```bash
# Basic usage (SRT file required)
python main.py --input video.mp4 --output video_de.mp4 --srt video.srt

# Use custom configuration
python main.py --input video.mp4 --output video_de.mp4 --srt video.srt --config my_config.json

# Keep intermediate files for debugging
python main.py --input video.mp4 --output video_de.mp4 --srt video.srt --keep-temp
```

### Text-Only Translation (SRT)

```bash
# Translate SRT file only
python -m src.translate_srt --input subtitle.srt --output subtitle_de.srt

# Specify source language
python -m src.translate_srt --input subtitle.srt --output subtitle_de.srt --src-lang eng_Latn

# Use custom config
python -m src.translate_srt --input subtitle.srt --output subtitle_de.srt --config config.json
```

### Command-Line Options (main.py)

- `--input`: Path to input video file (required)
- `--output`: Path to output translated video file (required)
- `--srt`: Path to SRT subtitle file (required)
- `--config`: Path to pipeline configuration file (default: `config.json`)
- `--keep-temp`: Keep intermediate files in temp/ directory

## Configuration

Edit `config.json` to customize translation and TTS settings:

### TTS Configuration Options

- **`exaggeration`** (default: `0.5`, range: 0.0 - 1.0): Controls voice expressiveness
  - Lower values (0.0 - 0.3): More neutral/flat voice
  - Higher values (0.7 - 1.0): More expressive/emotional voice
  - Recommended: 0.5 for natural speech

- **`cfg_weight`** (default: `0.7`, range: 0.0 - 1.0): Classifier-free guidance weight
  - Lower values (0.0 - 0.5): More creative/varied output
  - Higher values (0.7 - 1.0): More faithful to reference voice
  - Recommended: 0.7 for voice cloning accuracy

- **`use_multiple_segments`** (default: `true`): Audio generation mode
  - `true`: Generate audio per SRT segment with time-stretching to match exact timing (recommended)
  - `false`: Generate one continuous audio file (faster, but timing won't match SRT segments)

### Translation Configuration Options

- **`model_name`**: NLLB model to use
  - `facebook/nllb-200-distilled-600M`: Faster, requires ~2GB VRAM
  - `facebook/nllb-200-3.3B`: Higher quality, requires ~8GB VRAM

- **`target_language`**: Target language code (see Language Codes below)

- **`batch_size`**: Number of sentences to translate at once (higher = faster but more VRAM)

### Pipeline Configuration Options

- **`keep_intermediate_files`**: Keep temporary files for debugging (`temp/` directory)
- **`temp_dir`**: Directory for temporary files
- **`segments_dir`**: Directory for audio segment files

## Language Codes

NLLB uses language codes in the format `xxx_Yyyy`:

- English: `eng_Latn`
- German: `deu_Latn`
- Spanish: `spa_Latn`
- French: `fra_Latn`
- Italian: `ita_Latn`
- Portuguese: `por_Latn`
- Russian: `rus_Cyrl`
- Chinese: `zho_Hans`
- Japanese: `jpn_Jpan`
- Korean: `kor_Hang`
- Arabic: `arb_Arab`

[See full list of 200+ supported languages](https://github.com/facebookresearch/fairseq/tree/nllb#supported-languages)

## TTS Model Experiments

The `tts_sandbox/` directory contains experimental code for testing different TTS models (OpenVoice, XTTS, Chatterbox). See [tts_sandbox/readme.md](tts_sandbox/readme.md) for details on running standalone TTS experiments.


## System Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for NLLB-200-3.3B)
- **RAM**: 16GB system RAM
- **Storage**: 10GB free space (for models)
- **OS**: Windows, Linux, or macOS
- **Python**: 3.8+


## License

This project uses open-source models:
- **NLLB-200**: CC-BY-NC 4.0 (Meta AI)
- **fasttext**: MIT License (Meta AI)
- **Chatterbox TTS**: MIT License (Resemble AI)

