"""Real data configuration for actual datasets"""
from configs.config_schema import DataConfig
c4_data = DataConfig(
    sequence_length=2048,
    vocab_size=256000,
    batch_size=2,
    num_samples=100000,

    dataset_name="allenai/c4",
    dataset_config="en",
    tokenizer_name="google/gemma-2b",
    )