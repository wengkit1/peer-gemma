# configs/config_schema.py
"""
Updated Hydra-zen configuration dataclasses for PEER Gemma training
"""
from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import MISSING

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Pretrained model settings
    use_pretrained: bool = True
    model_name_or_path: str = "google/gemma-7b"
    tokenizer_name: Optional[str] = None

    # Base Gemma config (used for custom models or loaded from pretrained)
    hidden_size: int = 3072
    num_layers: int = 28
    num_heads: int = 16
    intermediate_size: int = 24576
    vocab_size: int = 256000
    max_position_embeddings: int = 8192

    # PEER-specific config
    replace_layers: str = "middle"  # "middle", "all", "first_half", "last_half" or list
    peer_enabled: bool = True

    # PEER layer config
    peer_num_experts: int = 1_000_000
    peer_heads: int = 16
    peer_num_experts_per_head: int = 16
    peer_dim_key: int = 128
    peer_pre_rmsnorm: bool = True

@dataclass
class DataConfig:
    """Data configuration"""
    sequence_length: int = 2048
    vocab_size: int = 256000
    batch_size: int = 2  # Smaller for 7B model
    num_samples: int = 100000

    # Data source settings
    dataset_name: str = "c4"
    dataset_config: Optional[str] = "en"
    tokenizer_name: Optional[str] = None

    # Data loading settings
    streaming: bool = False
    cache_dir: Optional[str] = None

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Lightning trainer settings
    max_epochs: int = 3
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "bf16-mixed"

    # Optimizer settings
    learning_rate: float = 1e-5  # Lower for pretrained models
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Scheduler settings
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    # Validation settings
    val_check_interval: float = 0.25
    limit_val_batches: int = 100

@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration"""
    # Wandb settings
    wandb_project: str = "peer-gemma-7b"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ["peer", "gemma", "7b"])
    wandb_notes: str = ""

    # Experiment identification
    experiment_name: str = "peer_gemma_7b"
    run_name: Optional[str] = None
    seed: int = 42

    # Logging settings
    log_every_n_steps: int = 50
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"

    # Paths
    output_dir: str = "/scratch/$USER/peer_experiments"
    checkpoint_dir: Optional[str] = None

    def __post_init__(self):
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
    accelerator: str = "gpu"
    devices: int = 4

    # Data loading settings
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True

    # Platform-specific
    use_mps: bool = False
    compile_model: bool = False

@dataclass
class Config:
    """Main configuration combining all sub-configs"""
    model: ModelConfig = MISSING
    data: DataConfig = MISSING
    training: TrainingConfig = MISSING
    experiment: ExperimentConfig = MISSING
    system: SystemConfig = MISSING