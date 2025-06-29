"""
PyTorch Lightning module for PEER Gemma training
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from typing import Dict
from loguru import logger

import sys

sys.path.append('.')
from peer_gemma import PEERGemmaForCausalLM, create_custom_gemma_config


class PEERGemmaLightningModule(pl.LightningModule):
    """Lightning module for PEER Gemma language modeling"""

    def __init__(
            self,
            # Model config
            hidden_size: int = 512,
            num_layers: int = 8,
            num_heads: int = 8,
            intermediate_size: int = 1024,
            vocab_size: int = 2000,
            max_position_embeddings: int = 1024,

            # PEER config
            replace_layers: str = "middle",
            peer_enabled: bool = True,
            peer_num_experts: int = 10000,
            peer_heads: int = 8,
            peer_num_experts_per_head: int = 16,
            peer_dim_key: int = 128,
            peer_pre_rmsnorm: bool = True,

            # Training config
            learning_rate: float = 1e-4,
            weight_decay: float = 0.01,
            beta1: float = 0.9,
            beta2: float = 0.95,
            eps: float = 1e-8,
            scheduler_type: str = "cosine",
            warmup_steps: int = 100,
            min_lr_ratio: float = 0.1,
            max_epochs: int = 5,

            # Logging
            log_detailed_metrics: bool = True,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model = self._create_model()

        # Training settings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.max_epochs = max_epochs
        self.log_detailed_metrics = log_detailed_metrics

        # Metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []

        logger.info(f"Created PEERGemmaLightningModule with {self._count_parameters():,} parameters")

    def _create_model(self) -> PEERGemmaForCausalLM:
        """Create the PEER Gemma model"""

        # Create base config
        config = create_custom_gemma_config(
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.num_heads,
            intermediate_size=self.hparams.intermediate_size,
            vocab_size=self.hparams.vocab_size
        )
        config.max_position_embeddings = self.hparams.max_position_embeddings

        if self.hparams.peer_enabled:
            # Validate PEER configuration to prevent "k out of range" errors
            total_active_experts = self.hparams.peer_heads * self.hparams.peer_num_experts_per_head

            if total_active_experts > self.hparams.peer_num_experts:
                logger.warning(
                    f"Active experts ({total_active_experts}) > total experts ({self.hparams.peer_num_experts})")
                logger.info("Adjusting configuration to prevent errors...")

                # Option 1: Reduce experts per head
                adjusted_experts_per_head = max(1, self.hparams.peer_num_experts // self.hparams.peer_heads)
                logger.info(
                    f"Reducing experts_per_head from {self.hparams.peer_num_experts_per_head} to {adjusted_experts_per_head}")

                peer_config = {
                    "dim": self.hparams.hidden_size,
                    "heads": self.hparams.peer_heads,
                    "num_experts": self.hparams.peer_num_experts,
                    "num_experts_per_head": adjusted_experts_per_head,
                    "dim_key": self.hparams.peer_dim_key,
                    "pre_rmsnorm": self.hparams.peer_pre_rmsnorm
                }
            else:
                # Configuration is valid
                peer_config = {
                    "dim": self.hparams.hidden_size,
                    "heads": self.hparams.peer_heads,
                    "num_experts": self.hparams.peer_num_experts,
                    "num_experts_per_head": self.hparams.peer_num_experts_per_head,
                    "dim_key": self.hparams.peer_dim_key,
                    "pre_rmsnorm": self.hparams.peer_pre_rmsnorm
                }

            # Log final PEER config
            logger.info(f"PEER config validation:")
            logger.info(f"  Total experts: {peer_config['num_experts']}")
            logger.info(f"  Heads: {peer_config['heads']}")
            logger.info(f"  Experts per head: {peer_config['num_experts_per_head']}")
            logger.info(f"  Active experts: {peer_config['heads'] * peer_config['num_experts_per_head']}")

            # Create PEER model
            model = PEERGemmaForCausalLM(
                config,
                replace_layers=self.hparams.replace_layers,
                peer_config=peer_config
            )

            logger.info(f"Created PEER Gemma with surgery info:")
            surgery_info = model.get_surgery_info()
            logger.info(f"  Replaced layers: {surgery_info['replaced_layer_indices']}")
            logger.info(f"  PEER parameters: {surgery_info['parameter_counts']['peer_parameters']:,}")
            logger.info(f"  PEER ratio: {surgery_info['parameter_counts']['peer_ratio']:.3f}")

        else:
            # Create standard Gemma
            from transformers import GemmaForCausalLM
            model = GemmaForCausalLM(config)
            logger.info("Created standard Gemma model (no PEER)")

        return model

    def _count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass"""
        try:
            outputs = self.model(input_ids=input_ids, **kwargs)
            return outputs.logits
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            logger.error(f"Input shape: {input_ids.shape}")
            logger.error(f"Input device: {input_ids.device}")
            logger.error(f"Model device: {next(self.model.parameters()).device}")
            raise

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss"""
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
        return loss

    def _compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute additional metrics"""
        metrics = {}

        # Accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).float()
        accuracy = correct.mean()
        metrics['accuracy'] = accuracy

        # Perplexity
        loss = self._compute_loss(logits, targets)
        perplexity = torch.exp(loss)
        metrics['perplexity'] = perplexity

        if self.log_detailed_metrics:
            # Top-k accuracy
            _, top5_preds = torch.topk(logits, 5, dim=-1)
            top5_correct = (top5_preds == targets.unsqueeze(-1)).any(dim=-1).float()
            metrics['top5_accuracy'] = top5_correct.mean()

            # Entropy of predictions
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            metrics['pred_entropy'] = entropy

        return metrics

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step"""
        input_ids, targets = batch

        # Forward pass
        logits = self(input_ids)

        # Compute loss
        loss = self._compute_loss(logits, targets)

        # Compute metrics
        metrics = self._compute_metrics(logits, targets)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for name, value in metrics.items():
            self.log(f'train_{name}', value, on_step=True, on_epoch=True)

        # Store for epoch end
        self.training_step_outputs.append({
            'loss': loss.detach(),
            **{k: v.detach() for k, v in metrics.items()}
        })

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step"""
        input_ids, targets = batch

        # Forward pass
        logits = self(input_ids)

        # Compute loss
        loss = self._compute_loss(logits, targets)

        # Compute metrics
        metrics = self._compute_metrics(logits, targets)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in metrics.items():
            self.log(f'val_{name}', value, on_step=False, on_epoch=True)

        # Store for epoch end
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            **{k: v.detach() for k, v in metrics.items()}
        })

        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch"""
        if self.training_step_outputs:
            # Log PEER-specific metrics if available
            if hasattr(self.model, 'get_peer_layers'):
                peer_layers = self.model.get_peer_layers()
                if peer_layers:
                    self.log('peer_layers_count', len(peer_layers))

                    # Log expert usage if available
                    # This would require modifications to PEER to track usage
                    # For now, we'll just log the number of experts
                    for i, peer_layer in enumerate(peer_layers[:2]):  # Log first 2 layers
                        if hasattr(peer_layer, 'num_experts'):
                            self.log(f'peer_layer_{i}_num_experts', peer_layer.num_experts)

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch"""
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""

        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'norm', 'ln']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                'params': decay_params,
                'weight_decay': self.weight_decay,
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
            }
        ]

        # Create optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.eps
        )

        # Create scheduler
        if self.scheduler_type == "cosine":
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
                num_cycles=0.5,
                last_epoch=-1
            )
        elif self.scheduler_type == "linear":
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            # No scheduler
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_start(self) -> None:
        """Called when training starts"""
        # Log model info to wandb
        if self.logger and hasattr(self.logger, 'experiment'):
            try:
                # Log model architecture summary
                sample_input = torch.randint(0, self.hparams.vocab_size, (1, 32))
                if self.device.type != 'cpu':
                    sample_input = sample_input.to(self.device)

                with torch.no_grad():
                    self.model.eval()
                    _ = self(sample_input)
                    self.model.train()

                # Log additional info
                self.logger.experiment.log({
                    "model_parameters": self._count_parameters(),
                    "peer_enabled": self.hparams.peer_enabled,
                })

                if hasattr(self.model, 'get_surgery_info'):
                    surgery_info = self.model.get_surgery_info()
                    self.logger.experiment.log({
                        "peer_surgery_info": surgery_info
                    })

            except Exception as e:
                logger.warning(f"Could not log model info to wandb: {e}")


def test_lightning_module():
    """Test the Lightning module"""
    logger.info("Testing PEER Gemma Lightning module...")

    # Create small model for testing with SAFE PEER config
    model = PEERGemmaLightningModule(
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        intermediate_size=256,
        vocab_size=100,
        # SAFE PEER config - ensure active experts <= total experts
        peer_num_experts=10000,  # Total experts
        peer_heads=2,  # Number of heads
        peer_num_experts_per_head=4,  # 2 * 4 = 8 active experts (< 100 total)
        peer_dim_key=32,  # Smaller dim_key
        replace_layers="middle",  # Only replace 1 layer for testing
        learning_rate=1e-3,
        max_epochs=1
    )

    # Test forward pass first
    logger.info("Testing forward pass...")
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    try:
        with torch.no_grad():
            logits = model(input_ids)
            logger.info(f"Forward pass successful! Logits shape: {logits.shape}")
            expected_shape = (batch_size, seq_len, 100)
            assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        return False

    # Test training step
    logger.info("Testing training step...")
    targets = torch.randint(0, 100, (batch_size, seq_len))
    batch = (input_ids, targets)

    try:
        # Set model to training mode
        model.train()
        loss = model.training_step(batch, 0)
        logger.info(f"Training step loss: {loss.item():.4f}")
        assert loss.item() > 0, "Loss should be positive"
    except Exception as e:
        logger.error(f"Training step failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    # Test validation step
    logger.info("Testing validation step...")
    try:
        model.eval()
        val_loss = model.validation_step(batch, 0)
        logger.info(f"Validation step loss: {val_loss.item():.4f}")
    except Exception as e:
        logger.error(f"Validation step failed: {e}")
        return False

    # Test optimizer configuration
    logger.info("Testing optimizer configuration...")
    try:
        # Mock trainer for optimizer test
        from unittest.mock import Mock
        mock_trainer = Mock()
        mock_trainer.estimated_stepping_batches = 100
        model.trainer = mock_trainer

        optimizers = model.configure_optimizers()
        assert 'optimizer' in optimizers, "Should return optimizer"
        assert 'lr_scheduler' in optimizers, "Should return scheduler"
        logger.info("Optimizer configuration successful!")
    except Exception as e:
        logger.error(f"Optimizer configuration failed: {e}")
        return False

    logger.success("✅ Lightning module test passed!")
    return True


if __name__ == "__main__":
    test_lightning_module()