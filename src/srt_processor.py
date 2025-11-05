"""
SRT File Processor Module.

Handles reading, parsing, modifying, and writing SRT subtitle files.
Preserves timing information while allowing text content to be replaced.
"""
import pysrt
from pathlib import Path


class SRTProcessor:
    """Process SRT subtitle files."""

    def __init__(self, srt_path):
        """
        Initialize SRT processor.

        Args:
            srt_path (str or Path): Path to SRT file
        """
        self.srt_path = Path(srt_path)
        self.subtitles = None
        self.original_encoding = 'utf-8'

    def load(self):
        """
        Load SRT file into memory.

        Returns:
            pysrt.SubRipFile: Loaded subtitle file

        Raises:
            FileNotFoundError: If SRT file doesn't exist
            ValueError: If SRT file cannot be parsed
        """
        if not self.srt_path.exists():
            raise FileNotFoundError(f"SRT file not found: {self.srt_path}")

        try:
            # Try UTF-8 first
            self.subtitles = pysrt.open(str(self.srt_path), encoding='utf-8')
            self.original_encoding = 'utf-8'
        except UnicodeDecodeError:
            # Fallback to latin-1
            try:
                self.subtitles = pysrt.open(str(self.srt_path), encoding='latin-1')
                self.original_encoding = 'latin-1'
                print(f"⚠ Loaded SRT with latin-1 encoding (non-UTF8)")
            except Exception as e:
                raise ValueError(f"Could not parse SRT file: {e}")

        if not self.subtitles:
            raise ValueError(f"SRT file is empty or invalid: {self.srt_path}")

        print(f"✓ Loaded {len(self.subtitles)} subtitle segments from {self.srt_path.name}")
        return self.subtitles

    def get_all_text(self):
        """
        Extract all text content from subtitles.

        Returns:
            list: List of text strings from each subtitle

        Raises:
            RuntimeError: If subtitles not loaded
        """
        if self.subtitles is None:
            raise RuntimeError("Subtitles not loaded. Call load() first.")

        return [sub.text for sub in self.subtitles]

    def get_segments(self):
        """
        Get subtitle segments with timing information.

        Returns:
            list: List of dicts with 'index', 'start', 'end', 'text'
                  start and end are SubRipTime objects with .ordinal attribute

        Raises:
            RuntimeError: If subtitles not loaded
        """
        if self.subtitles is None:
            raise RuntimeError("Subtitles not loaded. Call load() first.")

        segments = []
        for sub in self.subtitles:
            segments.append({
                'index': sub.index,
                'start': sub.start,  # Keep as SubRipTime object
                'end': sub.end,      # Keep as SubRipTime object
                'text': sub.text
            })

        return segments

    def replace_text(self, new_texts):
        """
        Replace subtitle text with translated content.

        Args:
            new_texts (list): List of translated text strings (same length as subtitles)

        Raises:
            RuntimeError: If subtitles not loaded
            ValueError: If number of texts doesn't match number of subtitles
        """
        if self.subtitles is None:
            raise RuntimeError("Subtitles not loaded. Call load() first.")

        if len(new_texts) != len(self.subtitles):
            raise ValueError(
                f"Number of translated texts ({len(new_texts)}) doesn't match "
                f"number of subtitles ({len(self.subtitles)})"
            )

        for subtitle, new_text in zip(self.subtitles, new_texts):
            subtitle.text = new_text

        print(f"✓ Replaced text in {len(self.subtitles)} subtitle segments")

    def save(self, output_path, encoding='utf-8'):
        """
        Save modified subtitles to new SRT file.

        Args:
            output_path (str or Path): Path for output SRT file
            encoding (str): Output file encoding (default: utf-8)

        Raises:
            RuntimeError: If subtitles not loaded
        """
        if self.subtitles is None:
            raise RuntimeError("Subtitles not loaded. Call load() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.subtitles.save(str(output_path), encoding=encoding)
            print(f"✓ Saved translated subtitles to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save SRT file: {e}")

    def print_summary(self):
        """Print summary of loaded subtitles."""
        if self.subtitles is None:
            print("No subtitles loaded")
            return

        print("\nSubtitle Summary:")
        print(f"  Total segments: {len(self.subtitles)}")
        print(f"  Duration: {self.subtitles[-1].end}")
        print(f"  Encoding: {self.original_encoding}")

        # Show first segment as sample
        if len(self.subtitles) > 0:
            first = self.subtitles[0]
            print(f"\n  First segment:")
            print(f"    Time: {first.start} --> {first.end}")
            print(f"    Text: {first.text[:100]}{'...' if len(first.text) > 100 else ''}")


def validate_srt(srt_path):
    """
    Validate that an SRT file exists and is readable.

    Args:
        srt_path (str or Path): Path to SRT file

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        processor = SRTProcessor(srt_path)
        processor.load()
        return True
    except Exception as e:
        print(f"✗ SRT validation failed: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Test with the Tanzania example file
    test_file = Path("data/Tanzania-caption.srt")

    if test_file.exists():
        processor = SRTProcessor(test_file)
        processor.load()
        processor.print_summary()

        # Get text segments
        texts = processor.get_all_text()
        print(f"\nExtracted {len(texts)} text segments")

        # Example: Replace with dummy translation
        dummy_translations = [f"[German translation of: {text[:30]}...]" for text in texts]
        processor.replace_text(dummy_translations)

        # Save to test output
        output = Path("data/test_output.srt")
        processor.save(output)
        print(f"\nTest complete! Check {output}")
    else:
        print(f"Test file not found: {test_file}")
