"""
Video Translation System - Source Modules

This package contains the core modules for the complete video translation pipeline:
- srt_processor: SRT file handling and processing
- language_detector: Language detection using HuggingFace fasttext
- text_translator: Text translation using NLLB-200-3.3B
- video_processor: Video/audio extraction and merging
- audio_processor: Audio timing synchronization and concatenation
- tts_handler: Chatterbox TTS integration for voice cloning
- translate_srt: SRT translation module
"""

from .srt_processor import SRTProcessor
from .language_detector import LanguageDetector
from .text_translator import TextTranslator
from .video_processor import VideoProcessor
from .audio_processor import AudioProcessor
from .tts_handler import ChatterboxTTSHandler

__all__ = [
    'SRTProcessor',
    'LanguageDetector',
    'TextTranslator',
    'VideoProcessor',
    'AudioProcessor',
    'ChatterboxTTSHandler'
]
