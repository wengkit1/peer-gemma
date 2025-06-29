"""Tiny data for quick testing"""
from hydra_zen import builds

from configs.config_schema import DataConfig

tiny_data = builds(
    DataConfig,
    sequence_length=64,
    vocab_size=1000,
    batch_size=2,
    num_samples=200,
    use_mock_data=True,
    mock_data_seed=42,
    mock_patterns=["repeat", "arithmetic"],
    populate_full_signature=True
)
