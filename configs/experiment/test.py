# configs/experiment/test.py
"""Test experiment configuration - using direct dataclass instances"""
from loguru._datetime import datetime

from configs.config_schema import ExperimentConfig

test_experiment = ExperimentConfig(
    wandb_project= "peer-gemma-20250630",
    experiment_name="test_run",
    wandb_tags=["test", "peer", "mock_data"],
    wandb_notes="Testing PEER Gemma implementation with mock data",
    log_every_n_steps=5,
    save_top_k=1,
    output_dir="/tmp/peer_experiments",
    seed=42,
)