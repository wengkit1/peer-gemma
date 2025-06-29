#!/usr/bin/env python3
"""
Main training script for PEER Gemma with Hydra-Zen configuration management
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from hydra_zen import make_config, store, zen
from pathlib import Path
from loguru import logger
import sys

# Setup environment first
from setup_env import setup_environment

setup_environment()

# Import our modules
from data.mock_data import MockDataModule
from models.peer_gemma_lightning import PEERGemmaLightningModule
from configs.config_schema import Config, MainConfig

# Import all config variants
from configs.model.small import small_model
from configs.model.tiny import tiny_model
from configs.data.mock import mock_data
from configs.data.tiny import tiny_data
from configs.training.quick import quick_training
from configs.training.full import full_training
from configs.experiment.test import test_experiment
from configs.experiment.dev import dev_experiment
from configs.system.mac import mac_system
from configs.system.cpu import cpu_system

# Store configurations - do this at module level, not in main
cs = store(group="model")
cs(small_model, name="small")
cs(tiny_model, name="tiny")

cs = store(group="data")
cs(mock_data, name="mock")
cs(tiny_data, name="tiny")

cs = store(group="training")
cs(quick_training, name="quick")
cs(full_training, name="full")

cs = store(group="experiment")
cs(test_experiment, name="test")
cs(dev_experiment, name="dev")

cs = store(group="system")
cs(mac_system, name="mac")
cs(cpu_system, name="cpu")

# Store main config
cs = store()
cs(MainConfig, name="config")


def setup_logging(experiment_config):
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    # Add file logging
    log_dir = Path(experiment_config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / f"{experiment_config.run_name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        level="DEBUG"
    )


def setup_wandb(experiment_config):
    """Setup Weights & Biases logging"""
    try:
        # Initialize wandb
        wandb_logger = WandbLogger(
            project=experiment_config.wandb_project,
            entity=experiment_config.wandb_entity,
            name=experiment_config.run_name,
            tags=experiment_config.wandb_tags,
            notes=experiment_config.wandb_notes,
            save_dir=experiment_config.output_dir,
        )

        logger.success("✅ Wandb logger initialized")
        return wandb_logger

    except Exception as e:
        logger.error(f"❌ Failed to initialize wandb: {e}")
        logger.info("💡 Make sure WANDB_API_KEY is set in your environment")
        logger.info("💡 Run: wandb login")
        return None


def create_callbacks(experiment_config):
    """Create Lightning callbacks"""
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_config.checkpoint_dir,
        filename=f"{experiment_config.experiment_name}_{{epoch:02d}}_{{val_loss:.3f}}",
        monitor=experiment_config.monitor_metric,
        mode=experiment_config.monitor_mode,
        save_top_k=experiment_config.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor=experiment_config.monitor_metric,
        mode=experiment_config.monitor_mode,
        patience=3,
        verbose=True,
        min_delta=0.001,
    )
    callbacks.append(early_stopping)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    return callbacks


def validate_config(cfg):
    """Validate configuration before training"""
    logger.info("🔍 Validating configuration...")

    # Check vocab size consistency
    if cfg.model.vocab_size != cfg.data.vocab_size:
        logger.warning(f"Model vocab_size ({cfg.model.vocab_size}) != Data vocab_size ({cfg.data.vocab_size})")
        logger.info("Using model vocab_size for consistency")
        cfg.data.vocab_size = cfg.model.vocab_size

    # Check sequence length
    if cfg.data.sequence_length > cfg.model.max_position_embeddings:
        logger.error(
            f"Sequence length ({cfg.data.sequence_length}) > max_position_embeddings ({cfg.model.max_position_embeddings})")
        raise ValueError("Sequence length too long for model")

    # Check PEER configuration
    if cfg.model.peer_enabled:
        if cfg.model.peer_heads * cfg.model.peer_num_experts_per_head > cfg.model.peer_num_experts:
            logger.warning("PEER active experts may exceed total experts")

    # Create output directories
    Path(cfg.experiment.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.experiment.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    logger.success("✅ Configuration validated")


def main(cfg: Config) -> None:
    """Main training function"""

    # Setup
    setup_logging(cfg.experiment)
    validate_config(cfg)

    logger.info("🚀 Starting PEER Gemma training")
    logger.info(f"📋 Experiment: {cfg.experiment.experiment_name}")
    logger.info(f"🏃 Run: {cfg.experiment.run_name}")

    # Set random seed
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Setup wandb
    wandb_logger = setup_wandb(cfg.experiment)

    # Create data module
    logger.info("📊 Creating data module...")
    from data.mock_data import create_smart_data_module

    data_module = create_smart_data_module(
        num_samples=cfg.data.num_samples,
        sequence_length=cfg.data.sequence_length,
        vocab_size=cfg.data.vocab_size,
        batch_size=cfg.data.batch_size,
        patterns=cfg.data.mock_patterns,
        seed=cfg.data.mock_data_seed,
        # Platform-optimized settings will be auto-detected
    )

    # Create model
    logger.info("🤖 Creating model...")
    model = PEERGemmaLightningModule(
        # Model config
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        intermediate_size=cfg.model.intermediate_size,
        vocab_size=cfg.model.vocab_size,
        max_position_embeddings=cfg.model.max_position_embeddings,

        # PEER config
        replace_layers=cfg.model.replace_layers,
        peer_enabled=cfg.model.peer_enabled,
        peer_num_experts=cfg.model.peer_num_experts,
        peer_heads=cfg.model.peer_heads,
        peer_num_experts_per_head=cfg.model.peer_num_experts_per_head,
        peer_dim_key=cfg.model.peer_dim_key,
        peer_pre_rmsnorm=cfg.model.peer_pre_rmsnorm,

        # Training config
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        beta1=cfg.training.beta1,
        beta2=cfg.training.beta2,
        eps=cfg.training.eps,
        scheduler_type=cfg.training.scheduler_type,
        warmup_steps=cfg.training.warmup_steps,
        min_lr_ratio=cfg.training.min_lr_ratio,
        max_epochs=cfg.training.max_epochs,
    )

    # Create callbacks
    callbacks = create_callbacks(cfg.experiment)

    # Create trainer
    logger.info("⚡ Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.system.accelerator,
        devices=cfg.system.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        limit_val_batches=cfg.training.limit_val_batches,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
    )

    # Log configuration to wandb
    if wandb_logger:
        wandb_logger.experiment.config.update({
            "model": cfg.model,
            "data": cfg.data,
            "training": cfg.training,
            "system": cfg.system
        })

    # Train model
    logger.info("🎯 Starting training...")
    try:
        trainer.fit(model, data_module)
        logger.success("🎉 Training completed successfully!")

        # Test model
        logger.info("🧪 Running test...")
        trainer.test(model, data_module)

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

    finally:
        # Cleanup
        if wandb_logger:
            wandb.finish()
        logger.info("🧹 Cleanup completed")


def quick_test():
    """Quick test function"""
    logger.info("🔬 Running quick test...")

    # Create test config by instantiating the components
    from hydra_zen import instantiate

    test_cfg = Config(
        model=instantiate(tiny_model),
        data=instantiate(tiny_data),
        training=instantiate(quick_training),
        experiment=instantiate(test_experiment),
        system=instantiate(mac_system)
    )

    # Modify for very quick test
    test_cfg.training.max_epochs = 1
    test_cfg.data.num_samples = 50
    test_cfg.training.limit_val_batches = 5
    test_cfg.experiment.log_every_n_steps = 1

    try:
        main(test_cfg)
        logger.success("✅ Quick test passed!")
        return True
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Check if we want to run quick test
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        quick_test()
    else:
        # Create config directly from our stored components
        from hydra_zen import instantiate

        # Instantiate default config
        cfg = Config(
            model=instantiate(tiny_model),
            data=instantiate(mock_data),
            training=instantiate(quick_training),
            experiment=instantiate(test_experiment),
            system=instantiate(mac_system)
        )

        # Run main with the config
        main(cfg)