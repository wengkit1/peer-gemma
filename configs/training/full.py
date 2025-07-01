# configs/training/full.py
"""Full training configuration - using direct dataclass instances"""
from configs.config_schema import TrainingConfig

full_training = TrainingConfig(
    max_epochs=5,
    learning_rate=1e-5,
    weight_decay=0.01,
    scheduler_type="cosine",
    warmup_steps=500,
    precision="bf16-mixed",
    gradient_clip_val=1.0,
    accumulate_grad_batches=8,
    val_check_interval=0.25,
    limit_val_batches=100,
)