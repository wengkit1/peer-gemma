# File: setup_env.py
"""
Environment and secrets setup for PEER Gemma training
"""
import os
from pathlib import Path
from tty import IFLAG

from dotenv import load_dotenv


def setup_environment():
    """Setup environment variables and paths"""

    # Load .env file if it exists
    load_dotenv()

    # Set cache directories
    hf_cache_dir = os.getenv("HF_HOME")
    torch_cache_dir = os.getenv("TORCH_HOME")

    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(torch_cache_dir, exist_ok=True)

    # Set cache paths
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
    os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
    os.environ["TORCH_HOME"] = torch_cache_dir

    # Check that required tokens are loaded
    if not os.getenv("HF_TOKEN"):
        raise ValueError("HF_TOKEN not found! Please set it in your .env file")

    if not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY not found! Please set it in your .env file")

    print(f"✅ HuggingFace cache: {hf_cache_dir}")
    print(f"✅ Torch cache: {torch_cache_dir}")
    print(f"✅ Tokens loaded from .env file")
    print(f"✅ Environment setup complete")

if __name__ == "__main__":
    setup_environment()