"""Real data configuration for actual datasets"""
from configs.config_schema import DataConfig
c4_data = DataConfig(
    sequence_length=256,
    vocab_size=256000,
    batch_size=4,
    num_samples=20000,

    use_mock_data=False,
    dataset_name="c4",
    dataset_config="en",
    tokenizer_name="google/gemma-2b",

    mock_data_seed=42,
    )