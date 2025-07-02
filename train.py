#!/usr/bin/env python3
"""
Training script for PEER Gemma 7B on NSCC
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb
from pathlib import Path
from loguru import logger
import sys
from transformers import AutoTokenizer

# Setup environment first
from setup_env import setup_environment

setup_environment()

# Import hydra-zen components
from hydra_zen import zen, store

# Import our config schema and instances
from configs.config_schema import Config, ModelConfig, DataConfig, TrainingConfig, ExperimentConfig, SystemConfig
from configs.model.gemma_7b import gemma_7b_model
from configs.data.c4 import c4_data
from configs.training.full import full_training
from configs.experiment.nscc import nscc_experiment
from configs.system.nscc import nscc_system

from models.peer_gemma_lightning import PEERGemmaLightningModule
from data.data import create_data_module

# Create store and add all configs
cs = store(group="model")
cs(gemma_7b_model, name="gemma_7b")

cs = store(group="data")
cs(c4_data, name="c4")

cs = store(group="training")
cs(full_training, name="full")

cs = store(group="experiment")
cs(nscc_experiment, name="nscc")

cs = store(group="system")
cs(nscc_system, name="nscc")

# Create the main config
from hydra_zen import builds
from omegaconf import MISSING

main_config = builds(
    Config,
    model=MISSING,
    data=MISSING,
    training=MISSING,
    experiment=MISSING,
    system=MISSING,
    hydra_defaults=[
        "_self_",
        {"model": "gemma_7b"},
        {"data": "c4"},
        {"training": "full"},
        {"experiment": "nscc"},
        {"system": "nscc"}
    ]
)

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
    if not os.getenv("WANDB_API_KEY"):
        logger.info("🔕 WANDB_API_KEY not found, skipping wandb logging")
        return None

    try:
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
        patience=5,  # Increased for larger models
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

    # Check if using real Gemma model
    if cfg.model.use_pretrained:
        logger.info(f"Using pretrained model: {cfg.model.model_name_or_path}")

        # Check HF token
        if not os.getenv("HF_TOKEN"):
            logger.error("HF_TOKEN required for Gemma models")
            raise ValueError("Missing HuggingFace token")
    else:
        logger.info("Using custom model configuration")

    # Validate GPU requirements
    if cfg.system.accelerator == "gpu" and not torch.cuda.is_available():
        logger.error("GPU requested but CUDA not available")
        raise ValueError("CUDA not available")

    # Create output directories
    Path(cfg.experiment.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.experiment.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    logger.success("✅ Configuration validated")


def train_task(model: ModelConfig, data: DataConfig, training: TrainingConfig,
               experiment: ExperimentConfig, system: SystemConfig):
    """Main training function"""

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

    logger.info("🚀 Starting PEER Gemma 7B training on NSCC")
    logger.info(f"📋 Experiment: {experiment.experiment_name}")
    logger.info(f"🎯 Model: {model.model_name_or_path if model.use_pretrained else 'Custom'}")
    logger.info(f"📊 Data: {data.dataset_name}")

    # Set random seed
    pl.seed_everything(experiment.seed, workers=True)

    # Setup wandb
    wandb_logger = setup_wandb(experiment)

    # Create tokenizer
    logger.info("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model.tokenizer_name or model.model_name_or_path,
        token=os.getenv("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create data module with tokenizer
    logger.info(" Creating tokenized data module...")
    data_module = create_data_module(
        tokenizer=tokenizer,
        dataset_name=data.dataset_name,
        dataset_config=data.dataset_config,
        sequence_length=data.sequence_length,
        batch_size=data.batch_size,
        num_samples=data.num_samples,
        num_workers=system.num_workers,
        pin_memory=system.pin_memory,
        persistent_workers=system.persistent_workers,
        cache_dir=os.getenv("HF_DATASETS_CACHE", "/scratch"),
        seed=experiment.seed,
    )

    logger.info(" Creating PEER Gemma model...")
    model_module = PEERGemmaLightningModule(
        use_pretrained=model.use_pretrained,
        model_name_or_path=model.model_name_or_path,
        tokenizer_name=model.tokenizer_name,

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

    # Create strategy for multi-GPU
    strategy = DDPStrategy(
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,

    ) if system.devices > 1 else "auto"

    # Create trainer
    logger.info("⚡ Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=training.max_epochs,
        accelerator=system.accelerator,
        devices=system.devices,
        strategy=strategy,
        precision=training.precision,
        gradient_clip_val=training.gradient_clip_val,
        accumulate_grad_batches=training.accumulate_grad_batches,
        val_check_interval=training.val_check_interval,
        limit_val_batches=training.limit_val_batches,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # Set to False for better performance
        log_every_n_steps=experiment.log_every_n_steps,
        num_sanity_val_steps=2,
    )

    # Log configuration to wandb
    if wandb_logger:
        try:
            wandb_logger.experiment.config.update({
                "model_name": model.model_name_or_path if model.use_pretrained else "custom",
                "peer_num_experts": model.peer_num_experts,
                "peer_heads": model.peer_heads,
                "training_lr": training.learning_rate,
                "training_epochs": training.max_epochs,
                "data_batch_size": data.batch_size,
                "system_devices": system.devices,
            })
        except Exception as e:
            logger.warning(f"⚠️ Failed to log config to wandb: {e}")

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