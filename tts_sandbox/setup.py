"""
Setup utilities for TTS Voice Translation system.

This module manages virtual environments and dependencies for different TTS models:
- OpenVoice: 6-step installation including PyTorch, PyAV, faster-whisper, and unidic
- XTTS: 2-step installation with PyTorch and model-specific requirements
- Chatterbox: 3-step installation with numpy, chatterbox-tts, and PyTorch 2.6+

Each model uses its own isolated virtual environment to avoid dependency conflicts.
"""
import sys
import subprocess
from pathlib import Path


def get_venv_paths(model_name):
    """
    Get virtual environment directory and Python executable paths for a model.

    Args:
        model_name (str): Name of the model ('openvoice', 'xtts', or 'chatterbox')

    Returns:
        tuple: (venv_path, python_exe_path) as Path objects
    """
    script_dir = Path(__file__).parent
    venv_name = f"venv_{model_name}"
    venv_path = script_dir / venv_name

    # Platform-specific Python executable path
    if sys.platform == 'win32':
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"

    return venv_path, python_exe


def check_venv_exists(model_name):
    """
    Check if virtual environment exists for the given model.

    Args:
        model_name (str): Name of the model ('openvoice', 'xtts', or 'chatterbox')

    Returns:
        bool: True if venv exists, False otherwise
    """
    _, python_exe = get_venv_paths(model_name)
    return python_exe.exists()


def create_venv(model_name):
    """
    Create virtual environment for the given model.

    Args:
        model_name (str): Name of the model ('openvoice', 'xtts', or 'chatterbox')

    Returns:
        bool: True if successful, False otherwise
    """
    venv_path, _ = get_venv_paths(model_name)

    print(f"Creating virtual environment: {venv_path}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Virtual environment created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create virtual environment!", file=sys.stderr)
        if e.stdout:
            print(f"Stdout: {e.stdout}", file=sys.stderr)
        if e.stderr:
            print(f"Stderr: {e.stderr}", file=sys.stderr)
        return False


def install_requirements(model_name):
    """
    Install requirements for the given model in its virtual environment.
    Each model has a specific installation sequence optimized for its dependencies.

    Args:
        model_name (str): Name of the model ('openvoice', 'xtts', or 'chatterbox')

    Returns:
        bool: True if successful, False otherwise
    """
    _, python_exe = get_venv_paths(model_name)
    script_dir = Path(__file__).parent
    requirements_file = script_dir / f"requirements_{model_name}.txt"

    # Validate requirements file exists (except for chatterbox which doesn't use it)
    if model_name != "chatterbox" and not requirements_file.exists():
        print(f"Requirements file not found: {requirements_file}", file=sys.stderr)
        return False

    # Model-specific installation sequences
    if model_name == "openvoice":
        return _install_openvoice(python_exe, requirements_file)
    elif model_name == "xtts":
        return _install_xtts(python_exe, requirements_file)
    elif model_name == "chatterbox":
        return _install_chatterbox(python_exe)
    else:
        print(f"Unknown model: {model_name}", file=sys.stderr)
        return False


def _install_openvoice(python_exe, requirements_file):
    """Install OpenVoice and its dependencies."""
    print("Step 1/6: Installing PyTorch with CUDA 12.1 support...")
    print("This may take several minutes (downloading ~2GB)...")
    print("-" * 60)
    if not _install_pytorch(python_exe, "cu121"):
        return False
    print("-" * 60)
    print("PyTorch with CUDA installed successfully!\n")

    print("Step 2/6: Installing PyAV pre-built wheel (av>=11.0.0)...")
    print("-" * 60)
    try:
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "av>=11.0.0"],
            check=True
        )
        print("PyAV installed successfully!\n")
    except subprocess.CalledProcessError:
        print("Warning: Could not install PyAV\n")

    print("Step 3/6: Installing faster-whisper (>=1.0.0)...")
    print("-" * 60)
    try:
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "faster-whisper>=1.0.0"],
            check=True
        )
        print("faster-whisper installed successfully!\n")
    except subprocess.CalledProcessError:
        print("Warning: Could not install faster-whisper\n")

    print("Step 4/6: Installing OpenVoice without dependencies...")
    print("-" * 60)
    try:
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "--no-deps",
             "git+https://github.com/myshell-ai/OpenVoice.git"],
            check=True
        )
        print("OpenVoice installed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not install OpenVoice: {e}\n")

    print(f"Step 5/6: Installing other dependencies from {requirements_file.name}...")
    print("This may take several minutes...")
    print("-" * 60)
    try:
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "-r", str(requirements_file)],
            check=True
        )
        print("-" * 60)
        print("Dependencies installed successfully!\n")
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"Failed to install dependencies!", file=sys.stderr)
        print(f"Exit code: {e.returncode}", file=sys.stderr)
        return False

    print("Step 6/6: Downloading unidic for MeloTTS (Japanese support)...")
    print("-" * 60)
    try:
        subprocess.run(
            [str(python_exe), "-m", "unidic", "download"],
            check=True
        )
        print("unidic downloaded successfully!\n")
    except subprocess.CalledProcessError:
        print("Warning: Could not download unidic (Japanese support may not work)\n")

    print("All dependencies installed successfully!\n")
    return True


def _install_xtts(python_exe, requirements_file):
    """Install XTTS and its dependencies."""
    print("Step 1/2: Installing PyTorch with CUDA 12.1 support...")
    print("This may take several minutes (downloading ~2GB)...")
    print("-" * 60)
    if not _install_pytorch(python_exe, "cu121"):
        return False
    print("-" * 60)
    print("PyTorch with CUDA installed successfully!\n")

    print(f"Step 2/2: Installing other dependencies from {requirements_file.name}...")
    print("This may take several minutes...")
    print("-" * 60)
    try:
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "-r", str(requirements_file)],
            check=True
        )
        print("-" * 60)
        print("All dependencies installed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"Failed to install dependencies!", file=sys.stderr)
        print(f"Exit code: {e.returncode}", file=sys.stderr)
        return False


def _install_chatterbox(python_exe):
    """Install Chatterbox TTS and its dependencies."""
    print("Step 1/3: Installing numpy...")
    print("-" * 60)
    try:
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "numpy"],
            check=True
        )
        print("numpy installed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install numpy: {e}", file=sys.stderr)
        return False

    print("Step 2/3: Installing chatterbox-tts...")
    print("-" * 60)
    try:
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "chatterbox-tts"],
            check=True
        )
        print("chatterbox-tts installed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install chatterbox-tts: {e}", file=sys.stderr)
        return False

    print("Step 3/3: Installing PyTorch with CUDA 12.4...")
    print("This may take several minutes (downloading ~2GB)...")
    print("-" * 60)
    if not _install_pytorch(python_exe, "cu124"):
        return False
    print("-" * 60)
    print("All dependencies installed successfully!\n")
    return True


def _install_pytorch(python_exe, cuda_version):
    """
    Install PyTorch with specified CUDA version.

    Args:
        python_exe: Path to Python executable
        cuda_version: CUDA version string (e.g., 'cu121', 'cu124')

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        subprocess.run(
            [
                str(python_exe), "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", f"https://download.pytorch.org/whl/{cuda_version}"
            ],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install PyTorch with CUDA!", file=sys.stderr)
        print(f"Exit code: {e.returncode}", file=sys.stderr)
        return False


def setup_environment(model_name, reinstall=False):
    """
    Set up the virtual environment for the specified TTS model.

    This function orchestrates the complete setup process:
    1. Creates a model-specific virtual environment if it doesn't exist
    2. Installs all required dependencies in the correct order

    Args:
        model_name (str): Name of the model ('openvoice', 'xtts', or 'chatterbox')
        reinstall (bool): If True, reinstall dependencies even if environment exists

    Returns:
        bool: True if environment is ready to use

    Raises:
        RuntimeError: If virtual environment creation or dependency installation fails
    """
    # Skip setup if environment exists and reinstall not requested
    if check_venv_exists(model_name) and not reinstall:
        print(f"Virtual environment for {model_name} already exists.")
        return True

    # Display setup header
    if reinstall:
        print(f"\nReinstalling dependencies for {model_name.upper()} model")
    else:
        print(f"\nFirst time setup for {model_name.upper()} model")
    print("=" * 60)

    # Create virtual environment if needed
    if not check_venv_exists(model_name):
        if not create_venv(model_name):
            raise RuntimeError(
                f"Failed to create virtual environment for {model_name}. "
                f"Please check the error messages above and try again."
            )

    # Install all dependencies for the model
    if not install_requirements(model_name):
        raise RuntimeError(
            f"Failed to install dependencies for {model_name}. "
            f"Please check the error messages above and try again."
        )

    print("=" * 60)
    print(f"Setup completed successfully for {model_name}!\n")
    return True
