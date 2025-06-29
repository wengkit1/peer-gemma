# File: configs/model/tiny.py
"""Tiny model for Mac testing"""
from hydra_zen import builds
from configs.config_schema import ModelConfig

tiny_model = builds(
    ModelConfig,
    hidden_size=128,
    num_layers=4,
    num_heads=2,
    intermediate_size=256,
    vocab_size=1000,
    max_position_embeddings=256,

    # PEER config - very small
    peer_num_experts=2500,
    peer_heads=2,
    peer_num_experts_per_head=4,
    peer_dim_key=32,
    replace_layers="first_half",
    populate_full_signature=True
)

