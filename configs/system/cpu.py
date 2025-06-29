"""CPU-only configuration"""
from hydra_zen import builds

from configs.config_schema import SystemConfig

cpu_system = builds(
    SystemConfig,
    accelerator="cpu",
    devices=1,
    num_workers=0,  # No multiprocessing on CPU
    pin_memory=False,
    persistent_workers=False,
    use_mps=False,
    compile_model=False,
    populate_full_signature=True
)