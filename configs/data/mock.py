# File: configs/data/mock.py
"""Mock data configuration"""
from hydra_zen import builds
from configs.config_schema import DataConfig

mock_data = builds(
    DataConfig,
    sequence_length=128,
    vocab_size=2000,
    batch_size=4,
    num_samples=1000,
    use_mock_data=True,
    mock_data_seed=42,
    mock_patterns=["repeat", "arithmetic", "random", "structured"],
    populate_full_signature=True
)
