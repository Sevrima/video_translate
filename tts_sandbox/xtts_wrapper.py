"""
Coqui XTTS-v2 TTS implementation for voice cloning and translation.
"""
import os
import torch
from pathlib import Path

# Patch torch.load for PyTorch 2.6+ compatibility BEFORE any TTS imports
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that sets weights_only=False for TTS compatibility."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load


def download_models(config):
    """
    Download XTTS-v2 models from Hugging Face if not present locally.

    Args:
        config (dict): Configuration parameters

    Returns:
        dict: Updated config with correct model paths
    """
    try:
        model_name = config.get('model_name', 'tts_models/multilingual/multi-dataset/xtts_v2')

        # Option 1: Use TTS library's built-in download (recommended)
        if not config.get('use_hf_direct_download', False):
            print(f"XTTS-v2 will be downloaded automatically by TTS library on first use.")
            print(f"Model: {model_name}")
            return config

        # Option 2: Direct download from Hugging Face
        from huggingface_hub import snapshot_download

        hf_repo = config.get('hf_model_repo', 'coqui/XTTS-v2')
        local_path = config.get('local_model_path', 'models/xtts_v2')

        if os.path.exists(local_path):
            print(f"XTTS-v2 model found locally at {local_path}")
            config['model_name'] = local_path
            return config

        print(f"Downloading XTTS-v2 from Hugging Face repo: {hf_repo}...")

        model_path = snapshot_download(
            repo_id=hf_repo,
            local_dir=local_path,
            local_dir_use_symlinks=False
        )

        print(f"✓ XTTS-v2 model downloaded to {model_path}")
        config['model_name'] = local_path

        return config

    except ImportError:
        print("Warning: huggingface_hub not installed. Using TTS library's built-in download.")
        return config
    except Exception as e:
        print(f"Warning: Failed to download from HuggingFace: {str(e)}")
        print("Falling back to TTS library's built-in download.")
        return config


class XTTSModel:
    """Coqui XTTS-v2 voice cloning model wrapper."""

    def __init__(self, config):
        """
        Initialize XTTS model with configuration.

        Args:
            config (dict): Configuration parameters for XTTS

        Raises:
            RuntimeError: If CUDA is not available
        """
        self.config = config

        # Check for CUDA availability - REQUIRED for XTTS
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available on this machine!\n\n"
                "XTTS requires a CUDA-enabled GPU to run.\n\n"
                "Possible solutions:\n"
                "1. Install CUDA toolkit from NVIDIA: https://developer.nvidia.com/cuda-downloads\n"
                "2. Make sure you have a CUDA-capable NVIDIA GPU\n"
                "3. Check that your GPU drivers are up to date\n\n"
                "To check CUDA availability, run:\n"
                "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"\n"
            )

        self.device = 'cuda'
        print(f"Using device: {self.device}")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        self.model = None

    def load_model(self):
        """Load XTTS-v2 model."""
        try:
            from TTS.api import TTS

            # Download models if needed and update config
            self.config = download_models(self.config)

            model_name = self.config.get('model_name', 'tts_models/multilingual/multi-dataset/xtts_v2')

            print(f"Loading XTTS-v2 model: {model_name}")
            print("Note: This may take a while on first run as models are downloaded...")

            # torch.load is already patched at module level for PyTorch 2.6+ compatibility
            # Load with GPU enabled (we've already verified CUDA is available)
            self.model = TTS(model_name, gpu=True)

            print(f"XTTS-v2 model loaded successfully on {self.device}")

        except ImportError:
            raise ImportError(
                "Coqui TTS not installed. Please install with: "
                "pip install TTS"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load XTTS-v2 model: {str(e)}")

    def synthesize(self, input_audio, output_path, text=None):
        """
        Synthesize German audio with voice cloning.

        Args:
            input_audio (str): Path to reference audio file for voice cloning
            output_path (str): Path to save output audio
            text (str, optional): Text to synthesize. If None, must be provided in config

        Returns:
            str: Path to output audio file
        """
        if self.model is None:
            self.load_model()

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Get text to synthesize
        if text is None:
            text = self.config.get('translated_text', None)
            if text is None:
                raise ValueError(
                    "Text must be provided either as argument or in config['translated_text']. "
                    "Implement transcription + translation pipeline or provide translated text."
                )

        # Validate reference audio exists
        if not os.path.exists(input_audio):
            raise FileNotFoundError(f"Reference audio not found: {input_audio}")

        print(f"Synthesizing German audio with voice cloning from: {input_audio}")

        # Get language and other parameters
        language = self.config.get('language', 'de')  # German
        speed = self.config.get('speed', 1.0)

        # XTTS-v2 specific parameters (only supported ones)
        temperature = self.config.get('temperature', 0.7)
        length_penalty = self.config.get('length_penalty', 1.0)
        repetition_penalty = self.config.get('repetition_penalty', 2.0)
        top_k = self.config.get('top_k', 50)
        top_p = self.config.get('top_p', 0.85)
        enable_text_splitting = self.config.get('enable_text_splitting', True)

        # Synthesize with voice cloning
        try:
            # Use only supported parameters
            self.model.tts_to_file(
                text=text,
                speaker_wav=input_audio,
                language=language,
                file_path=output_path,
                speed=speed,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                enable_text_splitting=enable_text_splitting
            )

            print(f"Audio saved to: {output_path}")
            return output_path

        except Exception as e:
            raise RuntimeError(f"Failed to synthesize audio with XTTS-v2: {str(e)}")

    def get_languages(self):
        """
        Get list of supported languages.

        Returns:
            list: List of supported language codes
        """
        if self.model is None:
            self.load_model()

        return self.model.languages if hasattr(self.model, 'languages') else ['de', 'en', 'es', 'fr', 'it', 'pt']


def process_audio(input_path, output_path, config):
    """
    Process audio file with XTTS-v2.

    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save output audio
        config (dict): Configuration parameters

    Returns:
        str: Path to output audio file
    """
    model = XTTSModel(config)

    # Get translated text if provided in config
    text = config.get('translated_text', None)

    return model.synthesize(input_path, output_path, text=text)


if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description='XTTS-v2 processor')
    parser.add_argument('--input', type=str, required=True, help='Input audio file')
    parser.add_argument('--output', type=str, required=True, help='Output audio file')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')

    args = parser.parse_args()

    try:
        # Load config
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Process audio
        result = process_audio(args.input, args.output, config)

        print(f"SUCCESS: {result}")
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
