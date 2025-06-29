# configs/training/full.py
"""Full training configuration - using direct dataclass instances"""
from configs.config_schema import TrainingConfig

full_training = TrainingConfig(
    max_epochs=10,
    learning_rate=5e-4,
    weight_decay=0.01,
    scheduler_type="cosine",
    warmup_steps=200,
    precision="bf16-mixed",
    gradient_clip_val=1.0,
    val_check_interval=0.25,
    limit_val_batches=100,
)