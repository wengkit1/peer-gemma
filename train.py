#!/usr/bin/env python3
"""
FINAL FIXED training script using hydra-zen correctly
No more ValidationError - uses proper dataclass instances
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
from loguru import logger
import sys

# Setup environment first
from setup_env import setup_environment

setup_environment()

# Import hydra-zen components
from hydra_zen import zen, store

# Import our config schema and instances
from configs.config_schema import Config, ModelConfig, DataConfig, TrainingConfig, ExperimentConfig, SystemConfig

# Import all config instances (these are now proper dataclass instances)
from configs.model.tiny import tiny_model
from configs.model.small import small_model
from configs.data.mock import mock_data
from configs.data.tiny import tiny_data
from configs.training.quick import quick_training
from configs.training.full import full_training
from configs.experiment.test import test_experiment
from configs.experiment.dev import dev_experiment
from configs.system.mac import mac_system
from configs.system.cpu import cpu_system

# Import our modules
from data.mock_data import create_smart_data_module
from models.peer_gemma_lightning import PEERGemmaLightningModule

# Create store and add all configs
cs = store(group="model")
cs(tiny_model, name="tiny")
cs(small_model, name="small")

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

# Create main config with proper defaults list for Hydra
from hydra_zen import builds
from omegaconf import MISSING

# Create the main config with hydra defaults
main_config = builds(
    Config,
    model=MISSING,  # Will be filled by defaults
    data=MISSING,  # Will be filled by defaults
    training=MISSING,  # Will be filled by defaults
    experiment=MISSING,  # Will be filled by defaults
    system=MISSING,  # Will be filled by defaults
    hydra_defaults=[
        "_self_",  # This config itself
        {"model": "tiny"},  # Default to tiny model
        {"data": "mock"},  # Default to mock data
        {"training": "quick"},  # Default to quick training
        {"experiment": "test"},  # Default to test experiment
        {"system": "mac"}  # Default to mac system
    ]
)

# Store the main config
cs = store()
cs(main_config, name="config")


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
    # Check if wandb is available and configured
    if not os.getenv("WANDB_API_KEY"):
        logger.info("🔕 WANDB_API_KEY not found, skipping wandb logging")
        return None

    try:
        # Debug info
        import wandb
        logger.info(f"🔍 Wandb user: {wandb.api.default_entity}")
        logger.info(f"🔍 Trying to create/use project: {experiment_config.wandb_project}")

        wandb_logger = WandbLogger(
            project=experiment_config.wandb_project,
            name=experiment_config.run_name,
            tags=experiment_config.wandb_tags,
            notes=experiment_config.wandb_notes,
            save_dir=experiment_config.output_dir,
            entity=os.getenv("WANDB_ENTITY"),
        )
        logger.success(f"✅ Wandb logger initialized for project: {experiment_config.wandb_project}")
        return wandb_logger
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize wandb: {e}")
        logger.info("🔕 Continuing without wandb logging")
        return None


def create_callbacks(experiment_config):
    """Create Lightning callbacks"""
    callbacks = []

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

    early_stopping = EarlyStopping(
        monitor=experiment_config.monitor_metric,
        mode=experiment_config.monitor_mode,
        patience=3,
        verbose=True,
        min_delta=0.001,
    )
    callbacks.append(early_stopping)

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
        total_active = cfg.model.peer_heads * cfg.model.peer_num_experts_per_head
        if total_active > cfg.model.peer_num_experts:
            logger.warning(f"PEER active experts ({total_active}) > total experts ({cfg.model.peer_num_experts})")

    # Create output directories
    Path(cfg.experiment.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.experiment.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    logger.success("✅ Configuration validated")


# The function signature matches the main config structure
def train_task(model: ModelConfig, data: DataConfig, training: TrainingConfig,
               experiment: ExperimentConfig, system: SystemConfig):
    """
    Main training function - zen extracts these from the config automatically
    """

    # Setup logging
    setup_logging(experiment)

    # Build config for validation
    cfg = Config(
        model=model,
        data=data,
        training=training,
        experiment=experiment,
        system=system
    )

    validate_config(cfg)

    logger.info("🚀 Starting PEER Gemma training")
    logger.info(f"📋 Experiment: {experiment.experiment_name}")
    logger.info(f"🎯 Model: {model.hidden_size}d, {model.num_layers} layers")
    logger.info(f"📊 Data: {data.num_samples} samples, batch_size={data.batch_size}")

    # Set random seed
    pl.seed_everything(experiment.seed, workers=True)

    # Setup wandb
    wandb_logger = setup_wandb(experiment)

    # Create data module
    logger.info("📊 Creating data module...")
    data_module = create_smart_data_module(
        num_samples=data.num_samples,
        sequence_length=data.sequence_length,
        vocab_size=data.vocab_size,
        batch_size=data.batch_size,
        patterns=data.mock_patterns,
        seed=data.mock_data_seed,
    )

    # Create model
    logger.info("🤖 Creating model...")
    model_module = PEERGemmaLightningModule(
        # Model config
        hidden_size=model.hidden_size,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        intermediate_size=model.intermediate_size,
        vocab_size=model.vocab_size,
        max_position_embeddings=model.max_position_embeddings,

        # PEER config
        replace_layers=model.replace_layers,
        peer_enabled=model.peer_enabled,
        peer_num_experts=model.peer_num_experts,
        peer_heads=model.peer_heads,
        peer_num_experts_per_head=model.peer_num_experts_per_head,
        peer_dim_key=model.peer_dim_key,
        peer_pre_rmsnorm=model.peer_pre_rmsnorm,

        # Training config
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        beta1=training.beta1,
        beta2=training.beta2,
        eps=training.eps,
        scheduler_type=training.scheduler_type,
        warmup_steps=training.warmup_steps,
        min_lr_ratio=training.min_lr_ratio,
        max_epochs=training.max_epochs,
    )

    # Create callbacks
    callbacks = create_callbacks(experiment)

    # Create trainer
    logger.info("⚡ Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=training.max_epochs,
        accelerator=system.accelerator,
        devices=system.devices,
        precision=training.precision,
        gradient_clip_val=training.gradient_clip_val,
        accumulate_grad_batches=training.accumulate_grad_batches,
        val_check_interval=training.val_check_interval,
        limit_val_batches=training.limit_val_batches,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        log_every_n_steps=experiment.log_every_n_steps,
    )

    # Log configuration to wandb (only if wandb is available)
    if wandb_logger:
        try:
            wandb_logger.experiment.config.update({
                "model_hidden_size": model.hidden_size,
                "model_num_layers": model.num_layers,
                "peer_num_experts": model.peer_num_experts,
                "training_lr": training.learning_rate,
                "training_epochs": training.max_epochs,
                "data_batch_size": data.batch_size,
            })
            logger.info("✅ Config logged to wandb")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log config to wandb: {e}")
            # Continue without wandb
            wandb_logger = None

    # Train model
    logger.info("🎯 Starting training...")
    try:
        trainer.fit(model_module, data_module)
        logger.success("🎉 Training completed successfully!")

        # Test model
        logger.info("🧪 Running test...")
        trainer.test(model_module, data_module)

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

    finally:
        # Cleanup
        if wandb_logger:
            wandb.finish()
        logger.info("🧹 Cleanup completed")


if __name__ == "__main__":
    # Add configs to hydra store
    store.add_to_hydra_store()

    # Use zen decorator to create CLI
    zen(train_task).hydra_main(
        config_path=None,
        config_name="config",
        version_base="1.1"
    )