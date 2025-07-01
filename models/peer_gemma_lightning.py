"""
Updated PyTorch Lightning module for PEER Gemma training with pretrained models
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from typing import Dict, Optional
from loguru import logger
import os

import sys
sys.path.append('.')
from peer_gemma import PEERGemmaForCausalLM, create_custom_gemma_config


class PEERGemmaLightningModule(pl.LightningModule):
    """Lightning module for PEER Gemma language modeling with pretrained support"""

    def __init__(
            self,
            # Training config
            learning_rate: float = 1e-5,  # Lower LR for pretrained models
            weight_decay: float = 0.01,
            beta1: float = 0.9,
            beta2: float = 0.95,
            eps: float = 1e-8,
            scheduler_type: str = "cosine",
            warmup_steps: int = 1000,  # More warmup for larger models
            min_lr_ratio: float = 0.1,
            max_epochs: int = 5,

            # Logging
            log_detailed_metrics: bool = True,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Validate PEER config
        if self.hparams.peer_enabled:
            total_active_experts = self.hparams.peer_heads * self.hparams.peer_num_experts_per_head
            if total_active_experts > self.hparams.peer_num_experts:
                logger.warning(
                    f"Active experts ({total_active_experts}) > total experts ({self.hparams.peer_num_experts})")
                adjusted_experts_per_head = max(1, self.hparams.peer_num_experts // self.hparams.peer_heads)
                logger.info(f"Adjusting experts_per_head to {adjusted_experts_per_head}")
                self.hparams.peer_num_experts_per_head = adjusted_experts_per_head

        # Create model
        self.model = self._create_model()
        self.tokenizer = self._load_tokenizer()

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

    def _load_tokenizer(self):
        """Load tokenizer"""
        tokenizer_name = self.hparams.tokenizer_name or self.hparams.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token=os.getenv("HF_TOKEN")
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _create_model(self):
        """Create the PEER Gemma model"""

        if self.hparams.use_pretrained:
            # Check for pre-built PEER model first
            scratch_dir = os.getenv("SCRATCH_DIR", f"/scratch/users/nus/{os.getenv('USER', 'e0686150')}")
            prebuild_path = f"{scratch_dir}/models/peer_gemma_7b_ready"

            if os.path.exists(f"{prebuild_path}/config.json") and self.hparams.peer_enabled:
                logger.info(f"Loading pre-built PEER model from: {prebuild_path}")
                return PEERGemmaForCausalLM.from_pretrained(
                    prebuild_path,
                    torch_dtype=torch.bfloat16,
                    device_map=None
                )

            logger.info(f"Loading pretrained model: {self.hparams.model_name_or_path}")
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                self.hparams.model_name_or_path,
                token=os.getenv("HF_TOKEN"),
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True,
            )

            if self.hparams.peer_enabled:
                # Only do surgery if no pre-built model exists
                logger.warning("Performing PEER surgery during training - this may cause distributed issues")
                peer_config = {
                    "dim": base_model.config.hidden_size,
                    "heads": self.hparams.peer_heads,
                    "num_experts": self.hparams.peer_num_experts,
                    "num_experts_per_head": self.hparams.peer_num_experts_per_head,
                    "dim_key": self.hparams.peer_dim_key,
                    "pre_rmsnorm": self.hparams.peer_pre_rmsnorm
                }

                model = PEERGemmaForCausalLM.from_pretrained_with_surgery(
                    base_model,
                    replace_layers=self.hparams.replace_layers,
                    peer_config=peer_config
                )
                return model
            else:
                return base_model

        else:
            # Create custom model (original behavior)
            logger.info("Creating custom Gemma model...")
            config = create_custom_gemma_config(
                hidden_size=self.hparams.hidden_size,
                num_layers=self.hparams.num_layers,
                num_heads=self.hparams.num_heads,
                intermediate_size=self.hparams.intermediate_size,
                vocab_size=self.hparams.vocab_size
            )
            config.max_position_embeddings = self.hparams.max_position_embeddings

            if self.hparams.peer_enabled:
                peer_config = {
                    "dim": self.hparams.hidden_size,
                    "heads": self.hparams.peer_heads,
                    "num_experts": self.hparams.peer_num_experts,
                    "num_experts_per_head": self.hparams.peer_num_experts_per_head,
                    "dim_key": self.hparams.peer_dim_key,
                    "pre_rmsnorm": self.hparams.peer_pre_rmsnorm
                }

                model = PEERGemmaForCausalLM(
                    config,
                    replace_layers=self.hparams.replace_layers,
                    peer_config=peer_config
                )
            else:
                from transformers import GemmaForCausalLM
                model = GemmaForCausalLM(config)

            return model

    def _count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass"""
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            return outputs.logits
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            logger.error(f"Input shape: {input_ids.shape}")
            logger.error(f"Input device: {input_ids.device}")
            logger.error(f"Model device: {next(self.model.parameters()).device}")
            raise

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss"""
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for cross-entropy
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return loss

    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute additional metrics"""
        metrics = {}

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Accuracy (only on non-padded tokens)
        predictions = torch.argmax(shift_logits, dim=-1)
        mask = shift_labels != -100
        correct = (predictions == shift_labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        metrics['accuracy'] = accuracy

        # Perplexity
        loss = self._compute_loss(logits, labels)
        perplexity = torch.exp(loss)
        metrics['perplexity'] = perplexity

        return metrics

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        # If batch is list of strings (from your original data.py)
        if isinstance(batch, (list, tuple)) and isinstance(batch[0], str):
            # Tokenize on-the-fly
            tokenized = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.sequence_length,
                return_tensors="pt"
            ).to(self.device)

            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            labels = input_ids.clone()  # For causal LM

        # If batch is already tokenized dict
        elif isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask', None)
            labels = batch.get('labels', input_ids)
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        # Forward pass
        logits = self(input_ids, attention_mask=attention_mask)

        # Compute loss
        loss = self._compute_loss(logits, labels)

        # Rest of your training step...
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        labels = batch.get('labels', input_ids)

        # Forward pass
        logits = self(input_ids, attention_mask=attention_mask)

        # Compute loss
        loss = self._compute_loss(logits, labels)

        # Compute metrics
        metrics = self._compute_metrics(logits, labels)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for name, value in metrics.items():
            self.log(f'val_{name}', value, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""

        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'norm', 'ln', 'layernorm']):
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
        if self.logger and hasattr(self.logger, 'experiment'):
            try:
                # Log model info
                self.logger.experiment.log({
                    "model_parameters": self._count_parameters(),
                    "peer_enabled": self.hparams.peer_enabled,
                    "use_pretrained": self.hparams.use_pretrained,
                    "model_name": self.hparams.model_name_or_path,
                })

                if hasattr(self.model, 'get_surgery_info'):
                    surgery_info = self.model.get_surgery_info()
                    self.logger.experiment.log({
                        "peer_surgery_info": surgery_info
                    })

            except Exception as e:
                logger.warning(f"Could not log model info to wandb: {e}")