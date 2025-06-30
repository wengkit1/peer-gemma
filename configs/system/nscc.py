"""NSCC system configuration"""
from configs.config_schema import SystemConfig

nscc_system = SystemConfig(
    accelerator="gpu",
    devices=4,  # Use 4 GPUs per node
    num_workers=8,  # More workers for larger model
    pin_memory=True,
    persistent_workers=True,
    use_mps=False,  # Not needed on NSCC
    compile_model=False,  # Disable for stability
)
