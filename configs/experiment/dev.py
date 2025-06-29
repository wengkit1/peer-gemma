# configs/experiment/dev.py
"""Development experiment configuration - using direct dataclass instances"""
from configs.config_schema import ExperimentConfig

dev_experiment = ExperimentConfig(
    wandb_project="peer-gemma-dev",
    experiment_name="development",
    wandb_tags=["dev", "peer", "gemma"],
    wandb_notes="Development experiments with PEER Gemma",
    log_every_n_steps=10,
    save_top_k=3,
    output_dir="/tmp/peer_experiments",
    seed=42,
)