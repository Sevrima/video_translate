"""
SRT Translation Module.

Translates SRT subtitle files from source language to target language using:
- Language detection (fasttext from HuggingFace)
- NLLB-200-3.3B translation model
- Preserves timing information
"""
import argparse
import json
import sys
from pathlib import Path

from .srt_processor import SRTProcessor
from .language_detector import LanguageDetector
from .text_translator import TextTranslator


def load_config(config_path="config.json"):
    """
    Load configuration from JSON file.

    Args:
        config_path (str): Path to config file

    Returns:
        dict: Configuration parameters
    """
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"⚠ Config file not found: {config_path}")
        print("  Using default configuration")
        return {
            "translation": {
                "model_name": "facebook/nllb-200-3.3B",
                "target_language": "deu_Latn",
                "device": "cuda",
                "batch_size": 8,
                "max_length": 512
            },
            "language_detection": {
                "method": "fasttext"
            }
        }

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def print_header(message):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70 + "\n")


def translate_srt_file(input_path, output_path, config_path="config.json",
                       src_lang=None, tgt_lang=None, device=None):
    """
    Translate an SRT file.

    Args:
        input_path (str): Path to input SRT file
        output_path (str): Path to output translated SRT file
        config_path (str): Path to configuration file
        src_lang (str): Source language code (auto-detect if None)
        tgt_lang (str): Target language code (overrides config if specified)
        device (str): Device to use ('cuda' or 'cpu')

    Returns:
        tuple: (success, src_lang, translated_texts) - success boolean, detected/specified source language, list of translated texts
    """
    # Load configuration
    config = load_config(config_path)
    translation_config = config.get('translation', {})
    detection_config = config.get('language_detection', {})

    # Override config with parameters
    if device:
        translation_config['device'] = device
    if tgt_lang:
        translation_config['target_language'] = tgt_lang

    # Load SRT file
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    processor = SRTProcessor(input_path)
    processor.load()

    # Get all text segments
    texts = processor.get_all_text()

    # Detect source language if not specified
    if src_lang is None:
        detector = LanguageDetector(method=detection_config.get('method', 'fasttext'))
        src_lang = detector.detect_from_texts(texts)
        print(f"✓ Detected source language: {src_lang}")
    else:
        print(f"Using specified source language: {src_lang}")

    tgt_lang_final = translation_config.get('target_language', 'deu_Latn')
    print(f"✓ Target language: {tgt_lang_final}")

    # Check if source and target are the same
    if src_lang == tgt_lang_final:
        print(f"\n⚠ Source and target languages are the same ({src_lang})")
        print("  Skipping translation")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processor.save(output_path)
        return True, src_lang, texts

    # Translate text
    print(f"\nTranslating {len(texts)} segments...")
    translator = TextTranslator(
        model_name=translation_config.get('model_name', 'facebook/nllb-200-3.3B'),
        device=translation_config.get('device', 'cuda')
    )

    translator.load_model()

    translated_texts = translator.translate_batch(
        texts,
        src_lang=src_lang,
        tgt_lang=tgt_lang_final,
        batch_size=translation_config.get('batch_size', 8),
        max_length=translation_config.get('max_length', 512)
    )

    translator.unload_model()

    # Save translated SRT
    processor.replace_text(translated_texts)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processor.save(output_path)

    return True, src_lang, translated_texts


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Translate SRT subtitle files using NLLB-200-3.3B',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.translate_srt --input data/Tanzania-caption.srt --output data/Tanzania-caption-DE.srt
  python -m src.translate_srt --input subtitle.srt --output subtitle_de.srt --config custom_config.json
  python -m src.translate_srt --input subtitle.srt --output subtitle_de.srt --src-lang eng_Latn
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input SRT file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output translated SRT file'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration JSON file (default: config.json)'
    )

    parser.add_argument(
        '--src-lang',
        type=str,
        default=None,
        help='Source language code (e.g., eng_Latn). If not specified, will be auto-detected'
    )

    parser.add_argument(
        '--tgt-lang',
        type=str,
        default=None,
        help='Target language code (e.g., deu_Latn). Overrides config file if specified'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use for translation (overrides config)'
    )

    args = parser.parse_args()

    try:
        # Print header
        print_header("SRT Translation Pipeline")

        # Load configuration
        print("▶ Loading configuration...")
        config = load_config(args.config)
        translation_config = config.get('translation', {})

        # Override config with command-line arguments
        if args.device:
            translation_config['device'] = args.device
        if args.tgt_lang:
            translation_config['target_language'] = args.tgt_lang

        print(f"  Model: {translation_config.get('model_name', 'facebook/nllb-200-3.3B')}")
        print(f"  Target: {translation_config.get('target_language', 'deu_Latn')}")
        print(f"  Device: {translation_config.get('device', 'cuda')}")

        # Translate
        print_header("Translating SRT")
        success, src_lang, translated_texts = translate_srt_file(
            args.input,
            args.output,
            args.config,
            args.src_lang,
            args.tgt_lang,
            args.device
        )

        # Final summary
        print_header("Translation Complete!")
        print(f"✓ Input:  {args.input}")
        print(f"✓ Output: {args.output}")
        print(f"✓ Segments translated: {len(translated_texts)}")
        print(f"✓ Languages: {src_lang} → {translation_config.get('target_language', 'deu_Latn')}\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Translation interrupted by user\n")
        return 1

    except Exception as e:
        print(f"\n\n✗ Translation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
