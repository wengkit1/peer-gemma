"""Gemma 7B model configuration"""
from configs.config_schema import ModelConfig

gemma_7b_model = ModelConfig(
    # Pretrained model settings
    use_pretrained=True,
    model_name_or_path="google/gemma-7b",
    tokenizer_name="google/gemma-7b",

    # Model architecture (will be loaded from pretrained)
    hidden_size=3072,  # Actual Gemma 7B hidden size
    num_layers=28,  # Actual Gemma 7B layers
    num_heads=16,  # Actual Gemma 7B heads
    intermediate_size=24576,  # Actual Gemma 7B FFN size
    vocab_size=256000,  # Actual Gemma 7B vocab
    max_position_embeddings=8192,

    # PEER config - scaled for 7B model
    replace_layers="middle",  # Replace middle layers
    peer_enabled=True,
    peer_num_experts=100_000,  # 1M experts as in paper
    peer_heads=16,  # Match model heads
    peer_num_experts_per_head=16,
    peer_dim_key=128,
    peer_pre_rmsnorm=True,
)