"""
Setup script for Text Translation Pipeline.

This script automatically installs and configures all dependencies required for
the SRT translation system. It handles PyTorch with CUDA, HuggingFace models,
and all supporting libraries.
"""
import subprocess
import sys
import os
from pathlib import Path


def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70 + "\n")


def run_command(command, description, check=True):
    """
    Run a shell command and display progress.

    Args:
        command (list): Command and arguments to execute
        description (str): Human-readable description of the command
        check (bool): Whether to raise error on failure

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"▶ {description}")
    print(f"  Command: {' '.join(command)}\n")

    try:
        result = subprocess.run(
            command,
            check=check,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} - Complete\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - Failed")
        print(f"  Exit code: {e.returncode}\n")
        return False


def check_cuda_available():
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}\n")
        else:
            print("⚠ CUDA is not available - will use CPU (slower)")
            print("  For GPU acceleration, install CUDA Toolkit from:")
            print("  https://developer.nvidia.com/cuda-downloads\n")
        return cuda_available
    except ImportError:
        print("⚠ PyTorch not yet installed, cannot check CUDA\n")
        return False


def main():
    """Main setup routine."""
    print_header("Text Translation Pipeline Setup")
    print("Installing dependencies...\n")

    script_dir = Path(__file__).parent
    requirements_file = script_dir / "requirements.txt"

    if not requirements_file.exists():
        print(f"✗ Requirements file not found: {requirements_file}")
        print("  Please ensure requirements.txt is in the same directory as setup.py\n")
        return 1

    # Step 1: Upgrade pip
    print_header("Step 1/3: Upgrading pip")
    run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip to latest version"
    )

    # Step 2: Install PyTorch with CUDA
    print_header("Step 2/3: Installing PyTorch with CUDA Support")
    print("Installing PyTorch with CUDA 12.4 (required for GPU acceleration)")
    print("This may take several minutes (~2GB download)...\n")

    success = run_command(
        [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu124",
            "--force-reinstall", "--no-cache-dir"
        ],
        "Installing PyTorch with CUDA 12.4"
    )

    if not success:
        print("⚠ PyTorch installation failed, but continuing...\n")

    # Check CUDA
    check_cuda_available()

    # Step 3: Install other requirements
    print_header("Step 3/5: Installing Other Dependencies")
    run_command(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "--upgrade"],
        "Installing transformers, sentencepiece, pysrt, ffmpeg-python, and other dependencies"
    )

    # Step 4: Install numpy (required for chatterbox-tts build dependencies)
    print_header("Step 4/6: Installing numpy")
    print("-" * 60)
    success = run_command(
        [sys.executable, "-m", "pip", "install", "numpy"],
        "Installing numpy"
    )
    if not success:
        print("Failed to install numpy")
        return 1
    print("numpy installed successfully!\n")

    # Step 5: Install Chatterbox TTS dependencies individually (skip pkuseg)
    print_header("Step 5/6: Installing Chatterbox TTS")
    print("Installing chatterbox-tts dependencies individually...")
    print("Note: Skipping pkuseg (Chinese text segmentation) due to build issues on Windows")
    print("-" * 60)

    # Install all chatterbox-tts dependencies except pkuseg
    deps_to_install = [
        "s3tokenizer",
        "transformers==4.46.3",
        "diffusers==0.29.0",
        "resemble-perth==1.0.1",
        "conformer==0.3.2",
        "safetensors==0.5.3"
    ]

    success = run_command(
        [sys.executable, "-m", "pip", "install"] + deps_to_install,
        "Installing chatterbox-tts dependencies"
    )

    if not success:
        print("Failed to install chatterbox-tts dependencies")
        return 1

    # Now install chatterbox-tts itself without dependencies
    print("\nInstalling chatterbox-tts package...")
    success = run_command(
        [sys.executable, "-m", "pip", "install", "chatterbox-tts", "--no-deps"],
        "Installing chatterbox-tts (without dependencies)"
    )

    if not success:
        print("Failed to install chatterbox-tts")
        return 1

    print("chatterbox-tts installed successfully (without pkuseg)!\n")
    print("Note: Chinese text support may be limited without pkuseg\n")

    # Step 6: Install/Upgrade PyTorch with CUDA 12.4
    print_header("Step 6/6: Upgrading PyTorch to CUDA 12.4")
    print("This may take several minutes (downloading ~2GB)...")
    print("-" * 60)
    success = run_command(
        [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ],
        "Upgrading PyTorch with CUDA 12.4"
    )
    if not success:
        print("⚠ PyTorch upgrade failed, but continuing...")
    print("-" * 60)

    # Final summary
    print_header("Setup Complete!")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Setup failed with error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
