# setup_env.py
"""
Fixed environment and secrets setup for PEER Gemma training
"""
import os
from pathlib import Path
from dotenv import load_dotenv


def setup_environment():
    """Setup environment variables and paths"""

    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv()
        print("✅ Loaded .env file")
    else:
        print("⚠️ No .env file found")

    # Set default cache directories if not set
    home_dir = Path.home()

    # HuggingFace cache
    hf_cache_dir = os.getenv("HF_HOME", str(home_dir / ".cache" / "huggingface"))
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
    os.environ["HF_DATASETS_CACHE"] = hf_cache_dir

    # PyTorch cache
    torch_cache_dir = os.getenv("TORCH_HOME", str(home_dir / ".cache" / "torch"))
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