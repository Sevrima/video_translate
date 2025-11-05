"""
Chatterbox TTS implementation for multilingual voice cloning and translation.
Supports 23 languages with zero-shot voice cloning capabilities.
"""
import os
import torch
from pathlib import Path


def download_models(config):
    """
    Download Chatterbox models from Hugging Face if needed.
    Chatterbox models are downloaded automatically via from_pretrained().

    Args:
        config (dict): Configuration parameters

    Returns:
        dict: Updated config with correct model paths
    """
    try:
        # Chatterbox models are downloaded automatically by the library
        print("Chatterbox models will be downloaded automatically on first use.")
        return config

    except Exception as e:
        raise RuntimeError(f"Failed to prepare Chatterbox models: {str(e)}")


class ChatterboxModel:
    """Chatterbox TTS model wrapper for multilingual voice cloning."""

    def __init__(self, config):
        """
        Initialize Chatterbox model with configuration.

        Args:
            config (dict): Configuration parameters for Chatterbox

        Raises:
            RuntimeError: If CUDA is not available
        """
        self.config = config

        # Check for CUDA availability - REQUIRED for Chatterbox
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available on this machine!\n\n"
                "Chatterbox requires a CUDA-enabled GPU to run.\n\n"
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
        self.multilingual = config.get('multilingual', True)

    def load_model(self):
        """Load Chatterbox TTS model."""
        try:
            from chatterbox.tts import ChatterboxTTS
            print("Loading Chatterbox TTS model...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)

            # Download models if needed
            self.config = download_models(self.config)

            print(f"Chatterbox loaded successfully on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {str(e)}\n"
                "Install with:\n"
                "  pip install chatterbox-tts\n"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Chatterbox model: {str(e)}")

    def synthesize(self, input_audio, output_path, text=None):
        """
        Synthesize audio with voice cloning using Chatterbox.

        Args:
            input_audio (str): Path to reference audio file for voice cloning
            output_path (str): Path to save output audio
            text (str, optional): Text to synthesize. If None, raises error.

        Returns:
            str: Path to output audio file
        """
        if self.model is None:
            self.load_model()

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Text is required for Chatterbox
        if text is None:
            raise ValueError(
                "Text must be provided for Chatterbox TTS. "
                "Implement transcription + translation pipeline or provide translated text in config."
            )

        print("Generating audio with Chatterbox TTS...")

        # Get configuration parameters
        exaggeration = self.config.get('exaggeration', 0.5)  # 0.0-1.0, higher = more expressive
        cfg_weight = self.config.get('cfg_weight', 0.5)      # 0.0-1.0, classifier-free guidance

        # Generate audio with voice cloning
        print(f"Synthesizing with parameters: exaggeration={exaggeration}, cfg_weight={cfg_weight}")
        wav = self.model.generate(
            text,
            audio_prompt_path=input_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )

        # Save the generated audio
        print(f"Saving audio to: {output_path}")

        # Use torchaudio to save (model.sr contains the sample rate)
        import torchaudio as ta

        output_path = os.path.abspath(output_path)
        print(f"Writing to absolute path: {output_path}")

        ta.save(output_path, wav, self.model.sr)

        print(f"Audio saved to: {output_path}")
        return output_path


def process_audio(input_path, output_path, config):
    """
    Process audio file with Chatterbox TTS.

    Args:
        input_path (str): Path to input audio file (reference for voice cloning)
        output_path (str): Path to save output audio
        config (dict): Configuration parameters

    Returns:
        str: Path to output audio file
    """
    model = ChatterboxModel(config)

    # Get translated text if provided in config
    text = config.get('translated_text', None)

    return model.synthesize(input_path, output_path, text=text)


if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description='Chatterbox TTS processor')
    parser.add_argument('--input', type=str, required=True, help='Input audio file (reference)')
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
