# TTS Voice Translation

Professional voice cloning and translation system supporting multiple state-of-the-art TTS models: OpenVoice v2, Coqui XTTS-v2, and Chatterbox TTS.

**System Requirements: NVIDIA GPU with CUDA support**

## Quick Start

The system handles all setup automatically. Simply run the script with your desired model:

```bash
# Using XTTS-v2
python main.py --input data/Tanzania-2.wav --model xtts --config config_xtts.json

# Using OpenVoice v2
python main.py --input data/Tanzania-2.wav --model openvoice --config config_openvoice.json

# Using Chatterbox TTS
python main.py --input audio.wav --model chatterbox --config config_chatterbox.json
```

**First Run:** The system automatically creates an isolated virtual environment and installs all dependencies (5-10 minutes).
**Subsequent Runs:** Processing begins immediately.

## Features

- **Multi-Model Support:** Three state-of-the-art TTS engines with automatic environment management
- **Isolated Dependencies:** Each model runs in its own virtual environment, preventing conflicts
- **Automatic Setup:** Zero manual installation required - dependencies are installed on first use
- **CUDA Optimization:** Full GPU acceleration support for efficient processing
- **Flexible Configuration:** JSON-based configuration system for fine-tuning model parameters

## Usage

```bash
python main.py --input <audio_file> --model <openvoice|xtts|chatterbox> --config <config_file> [OPTIONS]
```

### Command Line Options

- `--input`: Path to input audio file (required)
- `--model`: TTS model to use - `openvoice`, `xtts`, or `chatterbox` (required)
- `--config`: Path to configuration JSON file (required)
- `--output`: Path to output audio file (optional - auto-generated if not provided)
- `--reinstall`: Force reinstallation of dependencies

### Examples

```bash
# Using XTTS-v2
python main.py --input data/Tanzania-2.wav --model xtts --config config_xtts.json

# Using OpenVoice v2
python main.py --input data/Tanzania-2.wav --model openvoice --config config_openvoice.json

# Using Chatterbox TTS
python main.py --input audio.wav --model chatterbox --config config_chatterbox.json

# Specify custom output path
python main.py --input audio.wav --model xtts --config config_xtts.json --output translated.wav

# Force reinstall dependencies
python main.py --input audio.wav --model xtts --config config_xtts.json --reinstall
```

## Configuration Files

### config_xtts.json

Configuration for Coqui XTTS-v2 model:

- `device`: Device to use ("cuda" or "cpu")
- `output_dir`: Directory for output files
- `output_path`: Specific output file path (overrides output_dir)
- `model_name`: XTTS model name
- `use_hf_direct_download`: Use direct HuggingFace download (default: false, uses TTS library)
- `hf_model_repo`: HuggingFace repository for model (default: "coqui/XTTS-v2")
- `local_model_path`: Local path for downloaded models (default: "models/xtts_v2")
- `language`: Target language code (e.g., "de" for German)
- `speed`: Speech speed multiplier (1.0 = normal)
- `temperature`: Sampling temperature (0.1-1.0, higher = more variation)
- `length_penalty`: Length penalty for generation
- `repetition_penalty`: Penalty for repeated words
- `top_k`: Top-k sampling parameter
- `top_p`: Top-p (nucleus) sampling parameter
- `decoder_iterations`: Number of decoder iterations
- `gpt_cond_len`: GPT conditioning length
- `gpt_cond_chunk_len`: GPT conditioning chunk length
- `max_ref_length`: Maximum reference audio length
- `sound_norm_refs`: Normalize reference audio
- `enable_text_splitting`: Split long text into chunks
- `translated_text`: German text to synthesize

### config_openvoice.json

Configuration for OpenVoice v2 model:

- `device`: Device to use ("cuda" or "cpu")
- `output_dir`: Directory for output files
- `output_path`: Specific output file path
- `hf_base_model_repo`: HuggingFace repository for base model
- `checkpoint_path`: Path to base speaker checkpoints
- `config_path`: Path to model config file
- `converter_checkpoint`: Path to tone color converter checkpoint
- `base_speaker`: Base speaker to use
- `speed`: Speech speed multiplier
- `vad`: Enable voice activity detection
- `target_se_path`: Path to save target speaker embedding
- `source_se_path`: Path to source speaker embedding
- `temp_audio_path`: Temporary audio file path
- `conversion_message`: Status message during conversion
- `translated_text`: Text to synthesize

### config_chatterbox.json

Configuration for Chatterbox TTS model:

- `device`: Device to use ("cuda" or "cpu")
- `output_dir`: Directory for output files
- `output_path`: Specific output file path
- `model_name`: Chatterbox model variant
- `language`: Target language code
- `speed`: Speech speed multiplier
- `temperature`: Sampling temperature for generation
- `translated_text`: Text to synthesize

## Architecture

The system uses a modular architecture with isolated virtual environments:

- **main.py**: CLI entry point that validates inputs and dispatches to model-specific wrappers
- **setup.py**: Manages virtual environment creation and dependency installation for each model
- **Model Wrappers**: Isolated implementations (openvoice_wrapper.py, xtts_wrapper.py, chatterbox_wrapper.py) that run in their own environments

### Dependency Isolation

Each TTS model has unique dependency requirements that can conflict with other models:

- **OpenVoice**: Requires librosa 0.9.1, specific numpy versions, and custom PyAV setup
- **XTTS**: Requires librosa 0.10.0 and TTS-specific dependencies
- **Chatterbox**: Requires PyTorch 2.6+ with CUDA 12.4 and chatterbox-tts package

By using separate virtual environments, each model operates with its exact required dependencies without conflicts.

### Installation Process

Each model follows an optimized installation sequence:

- **OpenVoice**: 6-step process including PyTorch CUDA 12.1, PyAV, faster-whisper, OpenVoice, additional dependencies, and unidic
- **XTTS**: 2-step process with PyTorch CUDA 12.1 and model requirements
- **Chatterbox**: 3-step process with numpy, chatterbox-tts, and PyTorch CUDA 12.4

## Technical Notes

- All models require pre-translated text in the configuration file (`translated_text` parameter)
- Output files are saved to the `output/` directory by default or to a specified path
- Reference audio quality directly impacts voice cloning fidelity
- CUDA acceleration is essential for reasonable processing times
- Model checkpoints are automatically downloaded on first use
