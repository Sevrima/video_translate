"""
Main CLI script for voice translation using OpenVoice v2, Coqui XTTS-v2, or Chatterbox TTS.

This script manages separate virtual environments for each TTS model to avoid dependency conflicts.
Each model has a single implementation file (openvoice_wrapper.py, xtts_wrapper.py, chatterbox_wrapper.py) that runs
in its own isolated environment.
"""
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

# Import setup utilities
from setup import get_venv_paths, setup_environment


def load_config(config_path):
    """
    Load configuration from JSON file.

    Args:
        config_path (str): Path to config.json file

    Returns:
        dict: Configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def validate_input(input_path):
    """
    Validate input audio file exists.

    Args:
        input_path (str): Path to input audio file

    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    # Check if it's a valid audio format
    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    ext = os.path.splitext(input_path)[1].lower()

    if ext not in valid_extensions:
        print(f"Warning: {ext} might not be supported. Recommended formats: {', '.join(valid_extensions)}")


def get_output_path(input_path, model_name, config):
    """
    Generate output path if not specified.

    Args:
        input_path (str): Path to input file
        model_name (str): Name of the model being used
        config (dict): Configuration parameters

    Returns:
        str: Output file path
    """
    # Check if output path is specified in config and is not None
    if config.get('output_path'):
        return config['output_path']

    # Generate output path based on input
    input_file = Path(input_path)
    output_dir = config.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{input_file.stem}_{model_name}_de.wav"
    return os.path.join(output_dir, output_filename)


def run_model_subprocess(model_name, input_path, output_path, config_path):
    """
    Run the model in its own virtual environment using subprocess.

    Args:
        model_name (str): Name of the model ('openvoice', 'xtts', or 'chatterbox')
        input_path (str): Path to input audio file
        output_path (str): Path to output audio file
        config_path (str): Path to config file

    Returns:
        str: Path to output audio file

    Raises:
        RuntimeError: If subprocess execution fails
    """
    # Get the Python executable path
    _, python_exe = get_venv_paths(model_name)

    if not python_exe.exists():
        raise RuntimeError(
            f"Virtual environment Python executable not found: {python_exe}\n"
            f"This should not happen after setup. Please try deleting the venv folder and running again."
        )

    # Get the wrapper script path
    script_dir = Path(__file__).parent
    wrapper_script = script_dir / f"{model_name}_wrapper.py"

    if not wrapper_script.exists():
        raise FileNotFoundError(f"Wrapper script not found: {wrapper_script}")

    # Build the command
    cmd = [
        str(python_exe),
        str(wrapper_script),
        '--input', input_path,
        '--output', output_path,
        '--config', config_path
    ]

    print(f"Running {model_name} in virtual environment...")

    # Run the subprocess
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Print stdout
        if result.stdout:
            print(result.stdout)

        return output_path

    except subprocess.CalledProcessError as e:
        error_msg = f"Model execution failed with exit code {e.returncode}\n"
        if e.stdout:
            error_msg += f"\nStdout:\n{e.stdout}"
        if e.stderr:
            error_msg += f"\nStderr:\n{e.stderr}"
        raise RuntimeError(error_msg)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Voice translation using OpenVoice v2, Coqui XTTS-v2, or Chatterbox TTS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/Tanzania-2.wav --model openvoice --config config_openvoice.json
  python main.py --input data/Tanzania-2.wav --model xtts --config config_xtts.json
  python main.py --input audio.wav --model chatterbox --config config_chatterbox.json --output translated.wav
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input audio file'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['openvoice', 'xtts', 'chatterbox'],
        help='TTS model to use (openvoice, xtts, or chatterbox)'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration JSON file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output audio file (optional, will be generated if not provided)'
    )

    parser.add_argument(
        '--reinstall',
        action='store_true',
        help='Force reinstallation of dependencies even if environment exists'
    )

    args = parser.parse_args()

    try:
        # Validate input file
        print(f"Validating input: {args.input}")
        validate_input(args.input)

        # Load configuration
        print(f"Loading configuration from: {args.config}")
        model_config = load_config(args.config)

        # Determine output path
        output_path = args.output or get_output_path(args.input, args.model, model_config)
        print(f"Output will be saved to: {output_path}")

        # Set up virtual environment if needed (first time setup)
        print(f"\nChecking environment for {args.model.upper()} model...")
        setup_environment(args.model, reinstall=args.reinstall)

        # Run the appropriate model in its virtual environment
        print(f"\nProcessing with {args.model.upper()} model...")

        # Use subprocess to run model in its own environment
        result_path = run_model_subprocess(
            args.model,
            args.input,
            output_path,
            args.config
        )

        print(f"\nSuccess! Translated audio saved to: {result_path}")
        return 0

    except FileNotFoundError as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        return 1

    except RuntimeError as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
