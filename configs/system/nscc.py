"""NSCC system configuration"""
from configs.config_schema import SystemConfig

nscc_system = SystemConfig(
    accelerator="gpu",
    devices=4,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    use_mps=False,
    compile_model=False,
)
