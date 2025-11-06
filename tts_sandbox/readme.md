# TTS Model Experiments

Standalone testing environment for comparing TTS models: OpenVoice v2, XTTS-v2, and Chatterbox TTS.

**Note:** The main video translation pipeline uses Chatterbox TTS. This is for independent model testing only.

**Requirements:** NVIDIA GPU with CUDA support

## Quick Start

**No installation needed** - just run the command. Dependencies install automatically on first use (5-10 minutes per model).

```bash
# Test XTTS-v2
python main.py --input audio.wav --model xtts --config config_xtts.json

# Test OpenVoice v2
python main.py --input audio.wav --model openvoice --config config_openvoice.json

# Test Chatterbox TTS
python main.py --input audio.wav --model chatterbox --config config_chatterbox.json
```

## Features

- Compare three different TTS engines with voice cloning
- Isolated virtual environments prevent dependency conflicts
- Automatic dependency installation on first use
- GPU acceleration support

## Configuration

Each model has its own config file (`config_xtts.json`, `config_openvoice.json`, `config_chatterbox.json`) with model-specific parameters:

- `device`: "cuda" or "cpu"
- `output_path`: Where to save generated audio
- `translated_text`: Text to synthesize
- Model-specific parameters (temperature, speed, etc.)

## Command Options

- `--input`: Reference audio file for voice cloning (required)
- `--model`: TTS model - `openvoice`, `xtts`, or `chatterbox` (required)
- `--config`: Configuration JSON file (required)
- `--output`: Output audio path (optional, uses config if not provided)
- `--reinstall`: Force reinstall dependencies

## Dependencies

Each model uses isolated environments:

- **OpenVoice**: PyTorch CUDA 12.1, librosa 0.9.1, OpenVoice package
- **XTTS**: PyTorch CUDA 12.1, librosa 0.10.0, TTS package
- **Chatterbox**: PyTorch CUDA 12.4, chatterbox-tts package

Separate environments prevent version conflicts between models.
