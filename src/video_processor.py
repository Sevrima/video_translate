"""
Video Processing Module.

Handles video and audio extraction, manipulation, and merging using ffmpeg.
"""
import subprocess
import os
from pathlib import Path
import ffmpeg


class VideoProcessor:
    """Process video files - extract audio, replace audio tracks, etc."""

    def __init__(self, video_path, config=None):
        """
        Initialize video processor.

        Args:
            video_path (str or Path): Path to video file
            config (dict): Video configuration dict (optional)
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.config = config or {}
        self.video_codec = self.config.get('video_codec', 'copy')
        self.audio_codec = self.config.get('audio_codec', 'aac')
        self.audio_bitrate = self.config.get('audio_bitrate', '192k')

    def extract_audio(self, output_path=None, sample_rate=24000):
        """
        Extract audio from video file.

        Args:
            output_path (str or Path): Output audio file path (default: same name as video with .wav)
            sample_rate (int): Audio sample rate in Hz (default: 24000 for Chatterbox)

        Returns:
            Path: Path to extracted audio file
        """
        if output_path is None:
            output_path = self.video_path.with_suffix('.wav')
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"▶ Extracting audio from {self.video_path.name}...")

        try:
            # Use ffmpeg to extract audio
            stream = ffmpeg.input(str(self.video_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec='pcm_s16le',  # WAV format
                ar=sample_rate,       # Sample rate
                ac=1                  # Mono
            )
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

            print(f"✓ Audio extracted to {output_path}")
            return output_path

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Failed to extract audio: {error_msg}")

    def replace_audio(self, new_audio_path, output_video_path):
        """
        Replace video's audio track with new audio.

        Args:
            new_audio_path (str or Path): Path to new audio file
            output_video_path (str or Path): Path for output video

        Returns:
            Path: Path to output video file
        """
        new_audio_path = Path(new_audio_path)
        output_video_path = Path(output_video_path)

        if not new_audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {new_audio_path}")

        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"▶ Replacing audio in video...")

        try:
            # Load video and audio streams
            video_stream = ffmpeg.input(str(self.video_path)).video
            audio_stream = ffmpeg.input(str(new_audio_path)).audio

            # Merge video and audio
            stream = ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_video_path),
                vcodec=self.video_codec,      # Video codec from config
                acodec=self.audio_codec,       # Audio codec from config
                audio_bitrate=self.audio_bitrate,  # Audio bitrate from config
                strict='experimental'
            )

            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

            print(f"✓ Video with new audio saved to {output_video_path}")
            return output_video_path

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Failed to replace audio: {error_msg}")

    def get_duration(self):
        """
        Get video duration in seconds.

        Returns:
            float: Duration in seconds
        """
        try:
            probe = ffmpeg.probe(str(self.video_path))
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = float(video_info['duration'])
            return duration

        except Exception as e:
            raise RuntimeError(f"Failed to get video duration: {e}")

    def get_video_info(self):
        """
        Get detailed video information.

        Returns:
            dict: Video information (duration, resolution, fps, codec, etc.)
        """
        try:
            probe = ffmpeg.probe(str(self.video_path))

            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

            info = {
                'duration': float(probe['format']['duration']),
                'size': int(probe['format']['size']),
                'format': probe['format']['format_name']
            }

            if video_stream:
                info['video'] = {
                    'codec': video_stream['codec_name'],
                    'width': video_stream['width'],
                    'height': video_stream['height'],
                    'fps': eval(video_stream['r_frame_rate'])
                }

            if audio_stream:
                info['audio'] = {
                    'codec': audio_stream['codec_name'],
                    'sample_rate': int(audio_stream['sample_rate']),
                    'channels': audio_stream['channels']
                }

            return info

        except Exception as e:
            raise RuntimeError(f"Failed to get video info: {e}")

    def print_info(self):
        """Print video information in a readable format."""
        try:
            info = self.get_video_info()

            print(f"\nVideo Information: {self.video_path.name}")
            print(f"  Duration: {info['duration']:.2f} seconds")
            print(f"  Size: {info['size'] / (1024*1024):.2f} MB")
            print(f"  Format: {info['format']}")

            if 'video' in info:
                v = info['video']
                print(f"  Video: {v['width']}x{v['height']} @ {v['fps']:.2f} fps ({v['codec']})")

            if 'audio' in info:
                a = info['audio']
                print(f"  Audio: {a['sample_rate']} Hz, {a['channels']} channels ({a['codec']})")

        except Exception as e:
            print(f"⚠ Could not retrieve video info: {e}")


def extract_audio_from_video(video_path, output_path=None, sample_rate=24000):
    """
    Convenience function to extract audio from video.

    Args:
        video_path (str): Path to video file
        output_path (str): Output audio file path
        sample_rate (int): Audio sample rate

    Returns:
        Path: Path to extracted audio file
    """
    processor = VideoProcessor(video_path)
    return processor.extract_audio(output_path, sample_rate)


def replace_video_audio(video_path, audio_path, output_path):
    """
    Convenience function to replace video audio.

    Args:
        video_path (str): Path to original video
        audio_path (str): Path to new audio
        output_path (str): Path for output video

    Returns:
        Path: Path to output video
    """
    processor = VideoProcessor(video_path)
    return processor.replace_audio(audio_path, output_path)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_processor.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]

    try:
        processor = VideoProcessor(video_file)
        processor.print_info()

        # Test audio extraction
        audio_file = processor.extract_audio()
        print(f"\nExtracted audio to: {audio_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
