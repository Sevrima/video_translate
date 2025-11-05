"""
Main Video Translation Pipeline.

Complete end-to-end video translation pipeline:
1. Extract audio from video
2. Translate SRT subtitles
3. Generate translated audio using Chatterbox TTS with voice cloning
4. Match audio timing to SRT segments
5. Replace video audio with translated audio
"""
import argparse
import json
import sys
import shutil
from pathlib import Path

from src.video_processor import VideoProcessor
from src.audio_processor import AudioProcessor
from src.tts_handler import ChatterboxTTSHandler
from src.srt_processor import SRTProcessor
from src.translate_srt import translate_srt_file


def load_config(config_path="config.json"):
    """
    Load pipeline configuration.

    Args:
        config_path (str): Path to config file

    Returns:
        dict: Configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please create {config_path} with your pipeline configuration."
        )

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def print_header(message):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70 + "\n")


def main():
    """Main CLI entry point for video translation pipeline."""
    parser = argparse.ArgumentParser(
        description='Complete video translation pipeline with voice cloning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/Tanzania-2.mp4 --output data/Tanzania-2-DE.mp4
  python main.py --input video.mp4 --output video_de.mp4 --srt input.srt --config my_config.json
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input video file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output translated video file'
    )

    parser.add_argument(
        '--srt',
        type=str,
        default=None,
        help='Path to input SRT file (default: same name as video with .srt extension)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration JSON file (default: config.json)'
    )

    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep intermediate temporary files'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        print_header("Video Translation Pipeline")
        print("▶ Loading configuration...")
        config = load_config(args.config)

        translation_config = config.get('translation', {})
        tts_config = config.get('tts', {})
        audio_config = config.get('audio', {})
        video_config = config.get('video', {})
        pipeline_config = config.get('pipeline', {})

        # Override keep_temp from command line
        if args.keep_temp:
            pipeline_config['keep_intermediate_files'] = True

        # Setup paths
        input_video = Path(args.input)
        output_video = Path(args.output)

        if not input_video.exists():
            print(f"✗ Input video not found: {input_video}")
            return 1

        # Determine SRT file path
        if args.srt:
            input_srt = Path(args.srt)
        else:
            input_srt = input_video.with_suffix('.srt')

        if not input_srt.exists():
            print(f"✗ SRT file not found: {input_srt}")
            print("  Please provide SRT file with --srt argument")
            return 1

        # Setup temp directory
        temp_dir = Path(pipeline_config.get('temp_dir', 'temp'))
        temp_dir.mkdir(parents=True, exist_ok=True)

        segments_dir = Path(pipeline_config.get('segments_dir', 'temp/segments'))
        segments_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract audio from video
        print_header("Step 1/5: Extracting Audio from Video")

        video_processor = VideoProcessor(input_video, config=video_config)
        video_processor.print_info()

        reference_audio = temp_dir / "reference_audio.wav"
        video_processor.extract_audio(
            reference_audio,
            sample_rate=tts_config.get('sample_rate', 24000)
        )

        # Step 2: Translate SRT subtitles
        print_header("Step 2/5: Translating SRT Subtitles")

        translated_srt = temp_dir / "translated.srt"
        success, src_lang, translated_texts = translate_srt_file(
            input_srt,
            translated_srt,
            config_path=args.config
        )

        if not success:
            print("✗ SRT translation failed")
            return 1

        print(f"✓ Translated {len(translated_texts)} subtitle segments")

        # Clear VRAM before loading TTS model
        print("\n▶ Clearing GPU memory before loading TTS model...")
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Show available VRAM
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_free = memory_total - memory_allocated
            print(f"  GPU Memory: {memory_free:.2f} GiB free / {memory_total:.2f} GiB total")

        # Step 3: Generate audio for each SRT segment using Chatterbox
        print_header("Step 3/5: Generating Translated Audio")

        # Load SRT segments with timing
        srt_processor = SRTProcessor(translated_srt)
        srt_processor.load()
        srt_segments = srt_processor.get_segments()

        print(f"Generating speech for {len(translated_texts)} segments using Chatterbox TTS...")
        print(f"  Voice cloning from: {reference_audio}")
        if 'exaggeration' in tts_config:
            print(f"  Exaggeration: {tts_config['exaggeration']}")
        if 'cfg_weight' in tts_config:
            print(f"  CFG Weight: {tts_config['cfg_weight']}")

        # Initialize Chatterbox TTS
        tts_handler = ChatterboxTTSHandler(tts_config)
        tts_handler.load_model()

        # Check if single-pass mode is enabled
        single_pass_mode = tts_config.get('single_pass_mode', False)

        if single_pass_mode:
            print("\n▶ Single-pass TTS mode enabled")
            print("  Generating one continuous audio file for all text")

            # Generate single audio file for all text
            final_audio = temp_dir / "translated_audio.wav"
            tts_handler.generate_single_pass(
                translated_texts,
                reference_audio,
                final_audio
            )

            tts_handler.unload_model()

            # Step 4: Skip segmentation processing
            print_header("Step 4/5: Processing Audio (Single-Pass Mode)")
            print("✓ Using single audio file - no segmentation or time-stretching needed")

        else:
            # Generate audio for each segment (original behavior)
            generated_audio_files = tts_handler.generate_batch(
                translated_texts,
                reference_audio,
                segments_dir
            )

            tts_handler.unload_model()

            print(f"✓ Generated {len(generated_audio_files)} audio segments")

            # Step 4: Process audio timing and concatenate
            print_header("Step 4/5: Processing Audio Timing")

            audio_processor = AudioProcessor(
                sample_rate=audio_config.get('sample_rate', 24000)
            )

            final_audio = temp_dir / "translated_audio.wav"
            audio_processor.process_srt_segments(
                srt_segments,
                generated_audio_files,
                final_audio
            )

        # Step 5: Replace video audio
        print_header("Step 5/5: Creating Final Video")

        output_video.parent.mkdir(parents=True, exist_ok=True)

        video_processor.replace_audio(final_audio, output_video)

        # Cleanup temp files if not keeping them
        if not pipeline_config.get('keep_intermediate_files', False):
            print("\n▶ Cleaning up temporary files...")
            shutil.rmtree(temp_dir)
            print("✓ Temporary files removed")

        # Final summary
        print_header("Video Translation Complete!")
        print(f"✓ Input video:  {input_video}")
        print(f"✓ Input SRT:    {input_srt}")
        print(f"✓ Output video: {output_video}")
        print(f"✓ Segments:     {len(translated_texts)}")
        print(f"✓ Languages:    {src_lang} → {translation_config.get('target_language', 'deu_Latn')}")

        if pipeline_config.get('keep_intermediate_files', False):
            print(f"\n📁 Intermediate files saved in: {temp_dir}")
            print(f"   - Reference audio: {reference_audio}")
            print(f"   - Translated SRT:  {translated_srt}")
            print(f"   - Audio segments:  {segments_dir}")
            print(f"   - Final audio:     {final_audio}")

        print()
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user\n")
        return 1

    except Exception as e:
        print(f"\n\n✗ Pipeline failed: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
