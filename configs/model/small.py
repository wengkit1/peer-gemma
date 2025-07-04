# configs/model/small.py
"""Small model configuration for testing - using direct dataclass instances"""
from configs.config_schema import ModelConfig

small_model = ModelConfig(
    hidden_size=256,
    num_layers=6,
    num_heads=4,
    intermediate_size=512,
    vocab_size=2000,
    max_position_embeddings=512,

    # PEER config - smaller for testing
    peer_num_experts=2500,
    peer_heads=4,
    peer_num_experts_per_head=8,
    peer_dim_key=64,
    replace_layers="middle",
)