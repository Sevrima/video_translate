"""
TTS Handler Module.

Interfaces with Chatterbox TTS for voice cloning and speech synthesis.
"""
import sys
import torch
from pathlib import Path
from tqdm import tqdm


class ChatterboxTTSHandler:
    """Handle Chatterbox TTS for voice synthesis with voice cloning."""

    def __init__(self, config=None):
        """
        Initialize Chatterbox TTS handler.

        Args:
            config (dict): Configuration dict with TTS settings
        """
        self.config = config or {}
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Read configuration (no defaults - let model use its own defaults)
        self.exaggeration = self.config.get('exaggeration')
        self.cfg_weight = self.config.get('cfg_weight')
        self.sample_rate = self.config.get('sample_rate')
        self.multilingual = self.config.get('multilingual')

    def load_model(self):
        """Load Chatterbox TTS model."""
        if self.model is not None:
            print("Chatterbox model already loaded")
            return

        print("\n▶ Loading Chatterbox TTS model...")
        print(f"  Device: {self.device}")

        if self.device == 'cpu':
            print("  ⚠ Warning: Running on CPU will be very slow. CUDA GPU recommended.")

        try:
            from chatterbox.tts import ChatterboxTTS

            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            print("✓ Chatterbox TTS model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"Chatterbox TTS not installed or import failed: {e}\n"
                "Install with: pip install chatterbox-tts"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Chatterbox model: {e}")

    def generate(self, text, reference_audio_path, output_path=None):
        """
        Generate speech from text using voice cloning.

        Args:
            text (str): Text to synthesize
            reference_audio_path (str or Path): Reference audio for voice cloning
            output_path (str or Path): Output audio file path (optional)

        Returns:
            tuple: (audio_data, sample_rate) if output_path is None, else Path to saved file
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        reference_audio_path = Path(reference_audio_path)
        if not reference_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

        try:
            # Generate audio using Chatterbox
            generate_kwargs = {
                'audio_prompt_path': str(reference_audio_path)
            }
            if self.exaggeration is not None:
                generate_kwargs['exaggeration'] = self.exaggeration
            if self.cfg_weight is not None:
                generate_kwargs['cfg_weight'] = self.cfg_weight

            wav = self.model.generate(text, **generate_kwargs)

            # Save if output path provided
            if output_path is not None:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save audio (model.sr contains the sample rate)
                import torchaudio
                torchaudio.save(
                    str(output_path),
                    wav,
                    self.model.sr
                )

                return output_path

            return wav, self.model.sr

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {e}")

    def generate_batch(self, texts, reference_audio_path, output_dir):
        """
        Generate speech for multiple text segments.

        OPTIMIZED: Clones voice once using FULL reference audio, then reuses
        the cached embedding for all segments. This provides maximum voice quality
        while avoiding redundant voice cloning.

        Args:
            texts (list): List of text strings
            reference_audio_path (str or Path): Reference audio for voice cloning (full audio)
            output_dir (str or Path): Directory to save output audio files

        Returns:
            list: List of Paths to generated audio files
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n▶ Generating speech for {len(texts)} segments...")

        # OPTIMIZATION: Prepare voice conditionals ONCE using full reference audio
        print(f"▶ Cloning voice from full reference audio: {Path(reference_audio_path).name}")
        print("  (Using complete audio for maximum voice accuracy)")

        reference_audio_path = Path(reference_audio_path)
        if not reference_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

        # Prepare conditionals with config parameters
        prepare_kwargs = {}
        if self.exaggeration is not None:
            prepare_kwargs['exaggeration'] = self.exaggeration

        self.model.prepare_conditionals(str(reference_audio_path), **prepare_kwargs)
        print("✓ Voice embedding cached from full audio")

        generated_files = []

        # Build generate kwargs (only include if specified in config)
        generate_kwargs = {}
        if self.exaggeration is not None:
            generate_kwargs['exaggeration'] = self.exaggeration
        if self.cfg_weight is not None:
            generate_kwargs['cfg_weight'] = self.cfg_weight

        # Generate each segment using the cached voice embedding
        for i, text in enumerate(tqdm(texts, desc="Generating audio")):
            if not text or not text.strip():
                # Create short silence for empty text
                import numpy as np
                default_sr = 24000  # Chatterbox default
                silence = np.zeros(int(default_sr * 0.5), dtype=np.float32)

                output_path = output_dir / f"segment_{i+1:03d}.wav"
                import soundfile as sf
                sf.write(str(output_path), silence, default_sr)
                generated_files.append(output_path)
                continue

            output_path = output_dir / f"segment_{i+1:03d}.wav"

            try:
                # CRITICAL: Call generate() WITHOUT audio_prompt_path
                # This reuses the cached voice embedding from full audio
                wav = self.model.generate(text, **generate_kwargs)

                # Save the generated audio
                import torchaudio
                torchaudio.save(
                    str(output_path),
                    wav,
                    self.model.sr
                )

                generated_files.append(output_path)

            except Exception as e:
                print(f"  ⚠ Warning: Failed to generate segment {i+1}: {e}")
                # Create short silence as fallback
                import numpy as np
                silence = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
                import soundfile as sf
                sf.write(str(output_path), silence, self.sample_rate)
                generated_files.append(output_path)

        print(f"✓ Generated {len(generated_files)} audio files")
        return generated_files

    def generate_single_pass(self, texts, reference_audio_path, output_path):
        """
        Generate speech for all text segments in a single pass.

        Instead of generating segment-by-segment, this concatenates all text
        and generates one continuous audio file. This can be faster and produce
        more natural-sounding audio across segment boundaries.

        Args:
            texts (list): List of text strings
            reference_audio_path (str or Path): Reference audio for voice cloning
            output_path (str or Path): Path to save the single output audio file

        Returns:
            Path: Path to the generated audio file
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n▶ Generating speech in single-pass mode for {len(texts)} segments...")

        # Concatenate all text segments with spaces
        full_text = " ".join([text.strip() for text in texts if text and text.strip()])

        if not full_text:
            print("⚠ Warning: No text to generate. Creating silence.")
            import numpy as np
            default_sr = 24000
            silence = np.zeros(int(default_sr * 1.0), dtype=np.float32)
            import soundfile as sf
            sf.write(str(output_path), silence, default_sr)
            return output_path

        print(f"▶ Combined text length: {len(full_text)} characters")
        print(f"▶ Cloning voice from: {Path(reference_audio_path).name}")

        reference_audio_path = Path(reference_audio_path)
        if not reference_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

        try:
            # Generate audio using Chatterbox in one go
            generate_kwargs = {
                'audio_prompt_path': str(reference_audio_path)
            }
            if self.exaggeration is not None:
                generate_kwargs['exaggeration'] = self.exaggeration
            if self.cfg_weight is not None:
                generate_kwargs['cfg_weight'] = self.cfg_weight

            print("▶ Generating audio (this may take a while for long text)...")
            wav = self.model.generate(full_text, **generate_kwargs)

            # Save the generated audio
            import torchaudio
            torchaudio.save(
                str(output_path),
                wav,
                self.model.sr
            )

            print(f"✓ Generated single audio file: {output_path}")
            print(f"  Duration: {wav.shape[-1] / self.model.sr:.2f} seconds")

            return output_path

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech in single-pass mode: {e}")

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("✓ Chatterbox model unloaded from memory")


def generate_speech(text, reference_audio, output_path, config=None):
    """
    Convenience function to generate speech.

    Args:
        text (str): Text to synthesize
        reference_audio (str): Reference audio for voice cloning
        output_path (str): Output audio file path
        config (dict): Configuration dict

    Returns:
        Path: Path to generated audio file
    """
    handler = ChatterboxTTSHandler(config)
    handler.load_model()
    result = handler.generate(text, reference_audio, output_path)
    handler.unload_model()
    return result


def generate_speech_batch(texts, reference_audio, output_dir, config=None):
    """
    Convenience function to generate speech for multiple texts.

    Args:
        texts (list): List of text strings
        reference_audio (str): Reference audio for voice cloning
        output_dir (str): Directory for output files
        config (dict): Configuration dict

    Returns:
        list: List of Paths to generated audio files
    """
    handler = ChatterboxTTSHandler(config)
    handler.load_model()
    files = handler.generate_batch(texts, reference_audio, output_dir)
    handler.unload_model()
    return files


# Example usage
if __name__ == "__main__":
    import json
    from pathlib import Path

    if len(sys.argv) < 4:
        print("Usage: python tts_handler.py <text> <reference_audio> <output_file> [config_file]")
        sys.exit(1)

    text = sys.argv[1]
    ref_audio = sys.argv[2]
    output = sys.argv[3]
    config_file = sys.argv[4] if len(sys.argv) > 4 else "config.json"

    print(f"Generating speech for: {text[:50]}...")
    print(f"Using reference: {ref_audio}")
    print(f"Output: {output}")

    try:
        # Load configuration from config.json
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"✗ Config file not found: {config_file}")
            print(f"  Please create {config_file} or specify a valid config file")
            sys.exit(1)

        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
            config = full_config.get('tts', {})
        print(f"✓ Loaded TTS config from {config_file}")

        result = generate_speech(text, ref_audio, output, config)
        print(f"\n✓ Speech generated successfully: {result}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
