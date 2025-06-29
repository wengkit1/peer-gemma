# configs/system/cpu.py
"""CPU-only configuration - using direct dataclass instances"""
from configs.config_schema import SystemConfig

cpu_system = SystemConfig(
    accelerator="cpu",
    devices=1,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    use_mps=False,
    compile_model=False,
)