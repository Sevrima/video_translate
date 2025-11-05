"""
OpenVoice v2 TTS implementation for voice cloning and translation.
"""
import os
import torch
from pathlib import Path


def download_models(config):
    """
    Download OpenVoice models from Hugging Face if not present locally.

    Args:
        config (dict): Configuration parameters

    Returns:
        dict: Updated config with correct model paths
    """
    try:
        from huggingface_hub import snapshot_download

        checkpoint_path = config.get('checkpoint_path', 'checkpoints/base_speakers/EN')
        converter_checkpoint = config.get('converter_checkpoint', 'checkpoints/converter')

        # Check if models already exist
        if os.path.exists(checkpoint_path) and os.path.exists(converter_checkpoint):
            print("OpenVoice models found locally.")
            return config

        print("Downloading OpenVoice v2 models from Hugging Face...")

        # Download OpenVoice V2 models from correct HuggingFace repo
        base_model_repo = config.get('hf_base_model_repo', 'myshell-ai/OpenVoiceV2')
        print(f"Downloading OpenVoice V2 models from {base_model_repo}...")

        # Download entire repo to get v2 models
        base_path = snapshot_download(
            repo_id=base_model_repo,
            local_dir="openvoice_v2_models",
            local_dir_use_symlinks=False
        )

        print(f"[OK] Models downloaded to: {base_path}")

        print("[OK] OpenVoice models downloaded successfully!")

        # Update config paths for v2 models
        base_dir = 'openvoice_v2_models'

        # Try different possible v2 paths
        possible_paths = [
            f'{base_dir}/checkpoints/base_speakers/EN',
            f'{base_dir}/checkpoints_v2/base_speakers/EN',
            f'{base_dir}/base_speakers/EN'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config['checkpoint_path'] = path
                config['config_path'] = f'{path}/config.json'
                print(f"Using checkpoint path: {path}")
                break

        # Try different possible converter paths
        converter_paths = [
            f'{base_dir}/checkpoints/converter',
            f'{base_dir}/checkpoints_v2/converter',
            f'{base_dir}/converter'
        ]

        for path in converter_paths:
            if os.path.exists(path):
                config['converter_checkpoint'] = path
                print(f"Using converter path: {path}")
                break

        return config

    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Please install with: "
            "pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download OpenVoice models: {str(e)}")


class OpenVoiceModel:
    """OpenVoice v2 voice cloning model wrapper."""

    def __init__(self, config):
        """
        Initialize OpenVoice model with configuration.

        Args:
            config (dict): Configuration parameters for OpenVoice

        Raises:
            RuntimeError: If CUDA is not available
        """
        self.config = config

        # Check for CUDA availability - REQUIRED for OpenVoice
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available on this machine!\n\n"
                "OpenVoice requires a CUDA-enabled GPU to run.\n\n"
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
        self.tone_color_converter = None

    def load_model(self):
        """Load OpenVoice V2 model with MeloTTS."""
        try:
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
            from melo.api import TTS as MeloTTS

            # Download models if needed and update config
            self.config = download_models(self.config)

            # OpenVoice V2 uses MeloTTS as base TTS
            print("Loading MeloTTS for OpenVoice V2...")
            tts_language = self.config.get('tts_language', 'EN')  # EN, ES, FR, ZH, JP, KR
            self.model = MeloTTS(language=tts_language, device=self.device)

            # Load tone color converter for voice cloning
            converter_dir = self.config.get('converter_checkpoint', 'openvoice_v2_models/converter')
            converter_config = os.path.join(converter_dir, 'config.json')
            converter_ckpt = os.path.join(converter_dir, 'checkpoint.pth')

            self.tone_color_converter = ToneColorConverter(
                converter_config,
                device=self.device
            )
            self.tone_color_converter.load_ckpt(converter_ckpt)

            print(f"OpenVoice V2 loaded successfully on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {str(e)}\n"
                "Install with:\n"
                "  pip install git+https://github.com/myshell-ai/OpenVoice.git\n"
                "  pip install git+https://github.com/myshell-ai/MeloTTS.git\n"
                "  python -m unidic download"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenVoice V2 model: {str(e)}")

    def extract_tone_color(self, reference_audio):
        """
        Extract tone color embedding from reference audio.

        Args:
            reference_audio (str): Path to reference audio file

        Returns:
            Tone color embedding
        """
        from openvoice import se_extractor

        target_se, _ = se_extractor.get_se(
            reference_audio,
            self.tone_color_converter,
            vad=self.config.get('vad', True)
        )
        return target_se

    def synthesize(self, input_audio, output_path, text=None):
        """
        Synthesize German audio with voice cloning.

        Args:
            input_audio (str): Path to input audio file
            output_path (str): Path to save output audio
            text (str, optional): Text to synthesize. If None, extracts from audio

        Returns:
            str: Path to output audio file
        """
        if self.model is None or self.tone_color_converter is None:
            self.load_model()

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Extract tone color from reference audio
        print("Extracting tone color from reference audio...")
        target_se = self.extract_tone_color(input_audio)

        # If text is not provided, would need transcription + translation
        # For now, we'll use a placeholder or require text input
        if text is None:
            raise ValueError(
                "Text must be provided. Implement transcription + translation pipeline "
                "or provide translated text in config."
            )

        # Generate base audio with MeloTTS (OpenVoice V2)
        print("Generating base audio with MeloTTS...")
        temp_audio = self.config.get('temp_audio_path', 'temp_base.wav')

        # MeloTTS V2 API - generate speech
        speaker_id = self.model.hps.data.spk2id[self.config.get('base_speaker', 'EN-Default')]
        self.model.tts_to_file(
            text,
            speaker_id,
            temp_audio,
            speed=self.config.get('speed', 1.0)
        )

        # Apply tone color conversion (voice cloning)
        print("Applying voice cloning...")

        # Load base speaker embedding from V2 models
        base_speaker_file = self.config.get('base_speaker_se', 'en-default.pth')
        source_se_path = os.path.join('openvoice_v2_models/base_speakers/ses', base_speaker_file)

        # Load speaker embeddings as tensors
        source_se = torch.load(source_se_path, map_location=self.device)

        # Apply voice conversion with quality parameters
        # tau controls the strength of voice conversion (0.1-1.0, lower = stronger conversion)
        tau = self.config.get('tau', 0.3)

        self.tone_color_converter.convert(
            audio_src_path=temp_audio,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            tau=tau,
            message=self.config.get('conversion_message', 'Converting...')
        )

        # Cleanup temp file
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        print(f"Audio saved to: {output_path}")
        return output_path


def process_audio(input_path, output_path, config):
    """
    Process audio file with OpenVoice v2.

    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save output audio
        config (dict): Configuration parameters

    Returns:
        str: Path to output audio file
    """
    model = OpenVoiceModel(config)

    # Get translated text if provided in config
    text = config.get('translated_text', None)

    return model.synthesize(input_path, output_path, text=text)


if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description='OpenVoice v2 processor')
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
