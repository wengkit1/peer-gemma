"""NSCC experiment configuration"""
from configs.config_schema import ExperimentConfig
import os
scratch_dir = os.getenv("SCRATCH_DIR", "/scratch/users/nus/e0686150")
nscc_experiment = ExperimentConfig(
    wandb_project="peer-gemma-7b-nscc",
    experiment_name="gemma_7b_peer",
    wandb_tags=["nscc", "peer", "gemma-7b", "production"],
    wandb_notes="Training PEER Gemma 7B on NSCC ASPIRE 2A",
    log_every_n_steps=50,
    save_top_k=3,
    output_dir=f"{scratch_dir}/peer_gemma_experiments",  # Use the actual scratch dir
    seed=42,
)
