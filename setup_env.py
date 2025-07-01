# setup_env.py
"""
Fixed environment and secrets setup for PEER Gemma training
"""
import os
from pathlib import Path
from dotenv import load_dotenv


def setup_environment():
    """Setup environment variables and paths"""

    # If we're in a PBS job, change to the working directory first
    if 'PBS_O_WORKDIR' in os.environ:
        pbs_workdir = os.environ['PBS_O_WORKDIR']
        print(f"📁 PBS detected, changing to working directory: {pbs_workdir}")
        os.chdir(pbs_workdir)

    # Try multiple locations for .env file
    env_locations = [
        Path(".env"),  # Current directory
        Path.cwd() / ".env",  # Current working directory
        Path(__file__).parent / ".env",  # Same directory as this script
    ]

    env_loaded = False
    for env_file in env_locations:
        if env_file.exists():
            load_dotenv(env_file)
            print(f"✅ Loaded .env file from: {env_file.absolute()}")
            env_loaded = True
            break

    if not env_loaded:
        print("⚠️ No .env file found in any location:")
        for loc in env_locations:
            print(f"   - {loc.absolute()}")

    # Determine appropriate cache directory
    home_dir = Path.home()

    # Check if we're in PBS environment and cache dirs are already set
    if 'PBS_JOBID' in os.environ and os.getenv("HF_HOME"):
        # PBS script already set cache directories - respect them
        print(f"🚀 Using PBS-configured cache directories:")
        print(f"   HF_HOME: {os.getenv('HF_HOME')}")
        print(f"   TORCH_HOME: {os.getenv('TORCH_HOME', 'not set')}")
        # Don't override - just ensure consistency
        os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
        os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]
        if not os.getenv("TORCH_HOME"):
            os.environ["TORCH_HOME"] = str(Path(os.environ["HF_HOME"]).parent / "torch")
    else:
        # Set cache directories ourselves
        if 'PBS_JOBID' in os.environ:
            # We're in PBS but no cache dirs set - use scratch if available
            scratch_dir = os.getenv("SCRATCH_DIR")
            if scratch_dir:
                cache_base = Path(scratch_dir) / ".cache"
                print(f"🚀 Using HPC scratch space for cache: {cache_base}")
            else:
                cache_base = home_dir / ".cache"
                print(f"⚠️ No SCRATCH_DIR found, using home cache: {cache_base}")
        else:
            # Local development - use home cache
            cache_base = home_dir / ".cache"
            print(f"💻 Local development, using home cache: {cache_base}")

        # Set cache directories
        hf_cache_dir = str(cache_base / "huggingface")
        torch_cache_dir = str(cache_base / "torch")

        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
        os.environ["TORCH_HOME"] = torch_cache_dir

    # Create cache directories
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(torch_cache_dir, exist_ok=True)

    print(f"✅ HuggingFace cache: {hf_cache_dir}")
    print(f"✅ PyTorch cache: {torch_cache_dir}")

    # Check tokens (warn but don't fail)
    hf_token = os.getenv("HF_TOKEN")
    wandb_key = os.getenv("WANDB_API_KEY")

    if not hf_token:
        print("⚠️ HF_TOKEN not found - some models may not be accessible")
    else:
        print("✅ HF_TOKEN found")

    if not wandb_key:
        print("⚠️ WANDB_API_KEY not found - logging will be disabled")
    else:
        print("✅ WANDB_API_KEY found")

    print("✅ Environment setup complete")


if __name__ == "__main__":
    setup_environment()