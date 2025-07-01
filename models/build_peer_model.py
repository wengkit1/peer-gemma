#!/usr/bin/env python3
import os
import torch
from setup_env import setup_environment

setup_environment()

from peer_gemma import PEERGemmaForCausalLM
from loguru import logger


def build_peer_model():
    """Pre-build PEER model to avoid surgery during distributed training"""

    scratch_dir = os.getenv("SCRATCH_DIR", f"/scratch/users/nus/{os.getenv('USER', 'e0686150')}")
    model_dir = f"{scratch_dir}/models/peer_gemma_7b_ready"

    # Check if already exists
    if os.path.exists(f"{model_dir}/config.json"):
        logger.info(f"PEER model already exists at {model_dir}")
        return model_dir

    logger.info("Building PEER model...")
    os.makedirs(model_dir, exist_ok=True)

    # Create PEER model with surgery
    peer_config = {
        "num_experts": 1_000_000,
        "heads": 16,
        "num_experts_per_head": 16,
        "dim_key": 128,
        "pre_rmsnorm": True
    }

    peer_model = PEERGemmaForCausalLM.from_pretrained_with_surgery(
        "google/gemma-7b",
        replace_layers="middle",
        peer_config=peer_config,
        torch_dtype=torch.bfloat16,
        token=os.getenv("HF_TOKEN")
    )

    # Save the model
    peer_model.save_pretrained(model_dir)
    logger.success(f"PEER model saved to: {model_dir}")
    return model_dir


if __name__ == "__main__":
    build_peer_model()