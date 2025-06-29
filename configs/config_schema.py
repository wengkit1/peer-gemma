# File: configs/config_schema.py
"""
Hydra-zen configuration dataclasses for PEER Gemma training
"""
from dataclasses import dataclass
from typing import List, Optional

from huggingface_hub.constants import HF_HOME


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Base Gemma config
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: int = 1024
    vocab_size: int = 2000
    max_position_embeddings: int = 1024

    # PEER-specific config
    replace_layers: str = "middle"  # "middle", "all", "first_half", "last_half" or list
    peer_enabled: bool = True

    # PEER layer config
    peer_num_experts: int = 10000
    peer_heads: int = 8
    peer_num_experts_per_head: int = 16
    peer_dim_key: int = 128
    peer_pre_rmsnorm: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    sequence_length: int = 256
    vocab_size: int = 2000
    batch_size: int = 4  # Small for Mac testing
    num_samples: int = 1000  # For mock data

    # Mock data settings
    use_mock_data: bool = True
    mock_data_seed: int = 42
    mock_patterns: List[str] = None  # Will be set in __post_init__

    def __post_init__(self):
        if self.mock_patterns is None:
            self.mock_patterns = [
                "repeat",  # Repeating sequences
                "arithmetic",  # Simple arithmetic patterns
                "random",  # Random sequences
                "structured"  # Structured patterns
            ]


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Lightning trainer settings
    max_epochs: int = 5
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "bf16-mixed"  # Use bf16 on Mac M1/M2

    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Scheduler settings
    scheduler_type: str = "cosine"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1

    # Validation settings
    val_check_interval: float = 0.25  # Check 4 times per epoch
    limit_val_batches: int = 50

import os
@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration"""
    # Wandb settings
    wandb_project: str = os.getenv("WANDB_PROJECT")
    wandb_entity: Optional[str] = None  # Will use default from wandb login
    wandb_tags: List[str] = None
    wandb_notes: str = ""

    # Experiment identification
    experiment_name: str = "peer_gemma_test"
    run_name: Optional[str] = None  # Auto-generated if None
    seed: int = 42

    # Logging settings
    log_every_n_steps: int = 10
    save_top_k: int = 2
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"

    output_dir: str = HF_HOME
    checkpoint_dir: Optional[str] = None

    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = ["peer", "gemma", "moe", "test"]

        if self.checkpoint_dir is None:
            self.checkpoint_dir = f"{self.output_dir}/checkpoints"

        if self.run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.experiment_name}_{timestamp}"


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    # Device settings
    accelerator: str = "auto"  # "auto", "cpu", "gpu", "mps"
    devices: int = 1
    num_workers: int = 2  # For DataLoader

    # Memory settings
    pin_memory: bool = True
    persistent_workers: bool = True

    # Mac-specific optimizations
    use_mps: bool = True  # Use Metal Performance Shaders on Mac
    compile_model: bool = False  # Disable for compatibility


# Create hydra-zen configs with proper typing
from hydra_zen import builds

ModelConfigZen = builds(ModelConfig, populate_full_signature=True)
DataConfigZen = builds(DataConfig, populate_full_signature=True)
TrainingConfigZen = builds(TrainingConfig, populate_full_signature=True)
ExperimentConfigZen = builds(ExperimentConfig, populate_full_signature=True)
SystemConfigZen = builds(SystemConfig, populate_full_signature=True)


@dataclass
class Config:
    """Main configuration combining all sub-configs"""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    experiment: ExperimentConfig
    system: SystemConfig


# Main config for hydra-zen
MainConfig = builds(
    Config,
    model=ModelConfigZen,
    data=DataConfigZen,
    training=TrainingConfigZen,
    experiment=ExperimentConfigZen,
    system=SystemConfigZen,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"model": "small"},
        {"data": "mock"},
        {"training": "quick"},
        {"experiment": "test"},
        {"system": "mac"}
    ]
)