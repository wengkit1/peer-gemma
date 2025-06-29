from hydra_zen import builds
from configs.config_schema import ExperimentConfig
"""Development experiment configuration"""
dev_experiment = builds(
    ExperimentConfig,
    wandb_project="peer-gemma-dev",
    experiment_name="development",
    wandb_tags=["dev", "peer", "gemma"],
    wandb_notes="Development experiments with PEER Gemma",
    log_every_n_steps=10,
    save_top_k=3,
    output_dir="/Volumes/t9/peer_experiments",
    seed=42,
    populate_full_signature=True
)