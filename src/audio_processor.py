"""
Audio Processing Module.

Handles audio timing synchronization, time-stretching, and concatenation
to match SRT segment durations.
"""
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
from pathlib import Path


class AudioProcessor:
    """Process audio segments - time-stretching, concatenation, timing sync."""

    def __init__(self, sample_rate=24000):
        """
        Initialize audio processor.

        Args:
            sample_rate (int): Target sample rate (default: 24000 for Chatterbox)
        """
        self.sample_rate = sample_rate

    def load_audio(self, audio_path):
        """
        Load audio file.

        Args:
            audio_path (str or Path): Path to audio file

        Returns:
            tuple: (audio_data, sample_rate)
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        return audio, sr

    def get_duration(self, audio_path):
        """
        Get audio duration in seconds.

        Args:
            audio_path (str or Path): Path to audio file

        Returns:
            float: Duration in seconds
        """
        audio, sr = self.load_audio(audio_path)
        return len(audio) / sr

    def time_stretch_audio(self, audio_data, target_duration, current_sr=None):
        """
        Time-stretch audio to match target duration using FFmpeg (best quality).

        Args:
            audio_data (np.ndarray or str): Audio data array or path to audio file
            target_duration (float): Target duration in seconds
            current_sr (int): Current sample rate (required if audio_data is array)

        Returns:
            np.ndarray: Time-stretched audio data
        """
        import ffmpeg
        import tempfile

        # Handle input - convert to file if numpy array
        if isinstance(audio_data, (str, Path)):
            input_file = Path(audio_data)
            audio, sr = self.load_audio(audio_data)
        else:
            audio = audio_data
            sr = current_sr or self.sample_rate
            # Save to temp file
            temp_input = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_input.name, audio, sr)
            input_file = Path(temp_input.name)

        current_duration = len(audio) / sr
        speed_factor = current_duration / target_duration

        # If stretch is minimal (<2% change), skip stretching to preserve quality
        if 0.98 <= speed_factor <= 1.02:
            target_samples = int(target_duration * sr)
            current_samples = len(audio)
            if current_samples < target_samples:
                padding = target_samples - current_samples
                return np.pad(audio, (0, padding), mode='constant')
            else:
                return audio[:target_samples]

        # Limit stretch rate to avoid unnatural sound
        if speed_factor > 2.0:
            print(f"  ⚠ Warning: Large time compression needed ({speed_factor:.2f}x), capping at 2.0x")
            speed_factor = 2.0
        elif speed_factor < 0.5:
            print(f"  ⚠ Warning: Large time expansion needed ({speed_factor:.2f}x), capping at 0.5x")
            speed_factor = 0.5

        # Use FFmpeg atempo filter for high-quality time-stretching
        try:
            temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)

            # FFmpeg atempo filter (range 0.5-2.0)
            # For factors outside this range, chain multiple atempo filters
            atempo_filters = []
            remaining_factor = speed_factor

            while remaining_factor > 2.0:
                atempo_filters.append('atempo=2.0')
                remaining_factor /= 2.0

            while remaining_factor < 0.5:
                atempo_filters.append('atempo=0.5')
                remaining_factor /= 0.5

            if remaining_factor != 1.0:
                atempo_filters.append(f'atempo={remaining_factor:.6f}')

            # Build FFmpeg filter chain
            filter_chain = ','.join(atempo_filters)

            # Run FFmpeg with high quality settings
            stream = ffmpeg.input(str(input_file))
            stream = ffmpeg.filter(stream, 'aresample', 48000)  # Upsample for quality
            if atempo_filters:
                for f in atempo_filters:
                    filter_name, filter_value = f.split('=')
                    stream = ffmpeg.filter(stream, filter_name, float(filter_value))
            stream = ffmpeg.filter(stream, 'aresample', sr)  # Back to original rate
            stream = ffmpeg.output(stream, temp_output.name, acodec='pcm_s16le', ac=1, ar=sr)

            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

            # Load the stretched audio
            stretched_audio, _ = sf.read(temp_output.name)

            # Clean up temp files
            temp_output.close()
            Path(temp_output.name).unlink()
            if not isinstance(audio_data, (str, Path)):
                temp_input.close()
                input_file.unlink()

            # Ensure exact duration by padding or trimming
            target_samples = int(target_duration * sr)
            current_samples = len(stretched_audio)

            if current_samples < target_samples:
                padding = target_samples - current_samples
                stretched_audio = np.pad(stretched_audio, (0, padding), mode='constant')
            elif current_samples > target_samples:
                stretched_audio = stretched_audio[:target_samples]

            return stretched_audio

        except ffmpeg.Error as e:
            stderr_output = e.stderr.decode() if e.stderr else 'No stderr'
            # Find the actual error in stderr (skip version info)
            error_lines = [line for line in stderr_output.split('\n') if 'Error' in line or 'Invalid' in line or 'failed' in line]
            if error_lines:
                print(f"  ⚠ FFmpeg error: {error_lines[0][:100]}")
            else:
                print(f"  ⚠ FFmpeg time-stretching failed (unknown error)")
            print(f"     Using padding/trimming instead")
            # Fallback to simple padding/trimming
            target_samples = int(target_duration * sr)
            current_samples = len(audio)
            if current_samples < target_samples:
                padding = target_samples - current_samples
                return np.pad(audio, (0, padding), mode='constant')
            else:
                return audio[:target_samples]
        except Exception as e:
            print(f"  ⚠ FFmpeg time-stretching failed: {e}, using padding/trimming")
            # Fallback to simple padding/trimming
            target_samples = int(target_duration * sr)
            current_samples = len(audio)
            if current_samples < target_samples:
                padding = target_samples - current_samples
                return np.pad(audio, (0, padding), mode='constant')
            else:
                return audio[:target_samples]

    def save_audio(self, audio_data, output_path, sr=None):
        """
        Save audio data to file.

        Args:
            audio_data (np.ndarray): Audio data
            output_path (str or Path): Output file path
            sr (int): Sample rate (uses self.sample_rate if not provided)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sr = sr or self.sample_rate
        sf.write(str(output_path), audio_data, sr)

    def create_silence(self, duration_seconds):
        """
        Create silent audio segment.

        Args:
            duration_seconds (float): Duration in seconds

        Returns:
            np.ndarray: Silent audio data
        """
        samples = int(duration_seconds * self.sample_rate)
        return np.zeros(samples, dtype=np.float32)

    def concatenate_audio_segments(self, audio_segments):
        """
        Concatenate multiple audio segments.

        Args:
            audio_segments (list): List of audio data arrays (np.ndarray)

        Returns:
            np.ndarray: Concatenated audio
        """
        if not audio_segments:
            return np.array([], dtype=np.float32)

        return np.concatenate(audio_segments)

    def process_srt_segments(self, srt_segments, generated_audio_paths, output_path):
        """
        Process audio segments to match SRT timing and concatenate.

        Args:
            srt_segments (list): List of SRT segment dicts with 'start', 'end', 'text'
            generated_audio_paths (list): List of paths to generated audio files (one per segment)
            output_path (str or Path): Output path for final concatenated audio

        Returns:
            Path: Path to output audio file
        """
        if len(srt_segments) != len(generated_audio_paths):
            raise ValueError(
                f"Number of SRT segments ({len(srt_segments)}) doesn't match "
                f"number of audio files ({len(generated_audio_paths)})"
            )

        print(f"\n▶ Processing {len(srt_segments)} audio segments...")

        all_audio = []
        last_end_time = 0.0

        for i, (segment, audio_path) in enumerate(zip(srt_segments, generated_audio_paths)):
            # Parse timing from SRT
            start_ms = segment['start'].ordinal  # milliseconds
            end_ms = segment['end'].ordinal
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            target_duration = end_sec - start_sec

            print(f"  Segment {i+1}/{len(srt_segments)}: {start_sec:.2f}s - {end_sec:.2f}s (target: {target_duration:.2f}s)")

            # Add silence gap if needed
            if start_sec > last_end_time:
                gap_duration = start_sec - last_end_time
                print(f"    Adding {gap_duration:.2f}s silence gap")
                silence = self.create_silence(gap_duration)
                all_audio.append(silence)

            # Load and time-stretch audio to match segment duration
            audio, sr = self.load_audio(audio_path)
            current_duration = len(audio) / sr
            print(f"    Original duration: {current_duration:.2f}s, stretching to {target_duration:.2f}s")

            stretched_audio = self.time_stretch_audio(audio, target_duration, sr)
            all_audio.append(stretched_audio)

            last_end_time = end_sec

        # Concatenate all audio
        print(f"\n▶ Concatenating {len(all_audio)} audio segments...")
        final_audio = self.concatenate_audio_segments(all_audio)

        # Save
        output_path = Path(output_path)
        self.save_audio(final_audio, output_path)

        total_duration = len(final_audio) / self.sample_rate
        print(f"✓ Final audio saved: {output_path}")
        print(f"  Total duration: {total_duration:.2f}s")

        return output_path

    def process_single_audio(self, audio_path, output_path):
        """
        Process single audio file without segmentation (for single-pass TTS mode).

        Simply copies the single audio file to the output location without any
        time-stretching or segmentation. This is used when TTS generates one
        continuous audio file instead of segment-by-segment.

        Args:
            audio_path (str or Path): Path to the single generated audio file
            output_path (str or Path): Output path for final audio

        Returns:
            Path: Path to output audio file
        """
        import shutil

        audio_path = Path(audio_path)
        output_path = Path(output_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"\n▶ Processing single audio file (no segmentation)...")

        # Get audio info
        audio, sr = self.load_audio(audio_path)
        duration = len(audio) / sr

        print(f"  Audio duration: {duration:.2f}s")
        print(f"  Sample rate: {sr} Hz")

        # Copy to output location
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, output_path)

        print(f"✓ Audio saved: {output_path}")

        return output_path

    def trim_silence(self, audio_path, output_path=None, silence_thresh=-50, min_silence_len=1000):
        """
        Trim leading and trailing silence from audio.

        Args:
            audio_path (str or Path): Input audio file
            output_path (str or Path): Output audio file (overwrites input if None)
            silence_thresh (int): Silence threshold in dBFS (default: -50)
            min_silence_len (int): Minimum silence length in ms (default: 1000)

        Returns:
            Path: Path to trimmed audio file
        """
        audio_path = Path(audio_path)
        if output_path is None:
            output_path = audio_path
        else:
            output_path = Path(output_path)

        # Load with pydub
        audio_segment = AudioSegment.from_file(str(audio_path))

        # Detect silence
        silence_ranges = detect_silence(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )

        if not silence_ranges:
            # No silence detected, save as-is
            if output_path != audio_path:
                audio_segment.export(str(output_path), format="wav")
            return output_path

        # Trim leading silence
        start_trim = silence_ranges[0][1] if silence_ranges and silence_ranges[0][0] == 0 else 0

        # Trim trailing silence
        end_trim = silence_ranges[-1][0] if silence_ranges and silence_ranges[-1][1] == len(audio_segment) else len(audio_segment)

        trimmed = audio_segment[start_trim:end_trim]
        trimmed.export(str(output_path), format="wav")

        return output_path


# Convenience functions

def stretch_audio_to_duration(input_path, output_path, target_duration_seconds, sample_rate=24000):
    """
    Time-stretch audio file to match target duration.

    Args:
        input_path (str): Input audio file
        output_path (str): Output audio file
        target_duration_seconds (float): Target duration in seconds
        sample_rate (int): Sample rate

    Returns:
        Path: Path to output file
    """
    processor = AudioProcessor(sample_rate=sample_rate)
    audio, sr = processor.load_audio(input_path)
    stretched = processor.time_stretch_audio(audio, target_duration_seconds, sr)
    processor.save_audio(stretched, output_path, sr)
    return Path(output_path)


def concatenate_audio_files(audio_paths, output_path, sample_rate=24000):
    """
    Concatenate multiple audio files.

    Args:
        audio_paths (list): List of audio file paths
        output_path (str): Output audio file
        sample_rate (int): Sample rate

    Returns:
        Path: Path to output file
    """
    processor = AudioProcessor(sample_rate=sample_rate)

    segments = []
    for path in audio_paths:
        audio, sr = processor.load_audio(path)
        segments.append(audio)

    concatenated = processor.concatenate_audio_segments(segments)
    processor.save_audio(concatenated, output_path)

    return Path(output_path)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python audio_processor.py <input_audio> <target_duration_seconds>")
        sys.exit(1)

    input_file = sys.argv[1]
    target_duration = float(sys.argv[2])

    processor = AudioProcessor()

    # Test time-stretching
    audio, sr = processor.load_audio(input_file)
    current_duration = len(audio) / sr
    print(f"Current duration: {current_duration:.2f}s")
    print(f"Target duration: {target_duration:.2f}s")

    stretched = processor.time_stretch_audio(audio, target_duration, sr)
    output_file = Path(input_file).with_suffix('.stretched.wav')
    processor.save_audio(stretched, output_file, sr)

    print(f"Stretched audio saved to: {output_file}")
