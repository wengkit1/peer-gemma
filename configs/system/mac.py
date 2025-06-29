"""Mac system configuration - using direct dataclass instances"""
from configs.config_schema import SystemConfig

mac_system = SystemConfig(
    accelerator="mps",
    devices=1,
    num_workers=2,
    pin_memory=False,  # MPS doesn't support pin_memory
    persistent_workers=True,
    use_mps=True,
    compile_model=False,
)