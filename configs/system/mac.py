"""Mac system configuration"""
from hydra_zen import builds
from configs.config_schema import SystemConfig

mac_system = builds(
    SystemConfig,
    accelerator="mps",  # Use Metal Performance Shaders
    devices=1,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    use_mps=True,
    compile_model=False,  # Disable for Mac compatibility
    populate_full_signature=True
)