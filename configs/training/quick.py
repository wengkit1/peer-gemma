# configs/training/quick.py
"""Quick training configuration for testing - using direct dataclass instances"""
from configs.config_schema import TrainingConfig

quick_training = TrainingConfig(
    max_epochs=3,
    learning_rate=1e-3,
    weight_decay=0.01,
    scheduler_type="cosine",
    warmup_steps=50,
    precision="bf16-mixed",
    gradient_clip_val=1.0,
    val_check_interval=0.5,
    limit_val_batches=20,
)