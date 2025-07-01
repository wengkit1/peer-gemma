# setup_env.py
"""
Minimal environment setup - only handles .env loading
PBS script handles all path configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv


def setup_environment():
    """Load .env file and validate tokens - paths handled by PBS script"""

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

    # Handle cache directories - use existing if set by PBS, otherwise create defaults
    if 'PBS_JOBID' in os.environ:
        # We're in PBS - use the paths that were already set
        print("📁 Cache directories (set by PBS):")
        cache_vars = ["HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"]
        for var in cache_vars:
            value = os.getenv(var)
            if value:
                print(f"   {var}: {value}")
                # Ensure the directory exists
                try:
                    os.makedirs(value, exist_ok=True)
                except Exception as e:
                    print(f"   ⚠️ Could not create {var} directory: {e}")
            else:
                print(f"   {var}: not set")
    else:
        # Local development - set up local cache directories
        print("💻 Setting up local development cache directories")
        home_dir = Path.home()
        cache_base = home_dir / ".cache"

        # Only set if not already set
        if not os.getenv("HF_HOME"):
            hf_cache_dir = str(cache_base / "huggingface")
            os.environ["HF_HOME"] = hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
            os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
            os.makedirs(hf_cache_dir, exist_ok=True)
            print(f"   HF_HOME: {hf_cache_dir}")

        if not os.getenv("TORCH_HOME"):
            torch_cache_dir = str(cache_base / "torch")
            os.environ["TORCH_HOME"] = torch_cache_dir
            os.makedirs(torch_cache_dir, exist_ok=True)
            print(f"   TORCH_HOME: {torch_cache_dir}")

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