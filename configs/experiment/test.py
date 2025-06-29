"""Test experiment configuration"""
from hydra_zen import builds
from configs.config_schema import ExperimentConfig

test_experiment = builds(
    ExperimentConfig,
    wandb_project="peer-gemma-test",
    experiment_name="test_run",
    wandb_tags=["test", "peer", "mock_data"],
    wandb_notes="Testing PEER Gemma implementation with mock data",
    log_every_n_steps=5,
    save_top_k=1,
    output_dir="/Volumes/t9/peer_experiments",
    seed=42,
    populate_full_signature=True
)