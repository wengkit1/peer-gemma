# Updated peer_gemma.py
import torch
from transformers import GemmaForCausalLM, GemmaConfig, AutoModelForCausalLM
from PEER_pytorch import PEER
from loguru import logger
from typing import List, Dict, Any, Union
import json
import sys
import os


class PEERGemmaForCausalLM(GemmaForCausalLM):
    """Gemma model with PEER layers replacing MLP layers"""

    def __init__(self, config, replace_layers="middle", peer_config=None):
        super().__init__(config)
        self.replace_layers = replace_layers
        self.peer_config = peer_config or {}
        self.replaced_layer_indices = []
        self._replace_mlp_layers()
        logger.info(f"Surgery completed. Replaced {len(self.replaced_layer_indices)} MLP layers with PEER")

    def _replace_mlp_layers(self):
        """Replace MLP layers with PEER layers"""
        # Auto-detect Gemma dimensions
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        num_layers = len(self.model.layers)

        logger.info(f"Model info: {num_layers} layers, {hidden_size} hidden, {intermediate_size} intermediate")

        # Determine which layers to replace
        self.replaced_layer_indices = self._get_layers_to_replace(num_layers)

        # Default PEER configuration based on Gemma size
        default_peer_config = {
            "dim": hidden_size,
            "heads": min(16, hidden_size // 128),  # Reasonable default
            "num_experts": 250_000,
            "num_experts_per_head": 16,
            "dim_key": 128,
            "pre_rmsnorm": True
        }

        # Merge with provided config
        final_peer_config = {**default_peer_config, **self.peer_config}

        logger.info(f"PEER config: {json.dumps(final_peer_config, indent=2)}")
        logger.info(f"Replacing layers: {self.replaced_layer_indices}")

        # Track parameter changes
        total_original_params = 0
        total_peer_params = 0

        # Replace MLP layers
        for i in self.replaced_layer_indices:
            original_mlp = self.model.layers[i].mlp

            # Count original parameters
            original_params = sum(p.numel() for p in original_mlp.parameters())
            total_original_params += original_params

            # Create PEER layer
            peer_layer = PEER(**final_peer_config)

            # Count PEER parameters
            peer_params = sum(p.numel() for p in peer_layer.parameters())
            total_peer_params += peer_params

            # Replace the layer
            self.model.layers[i].mlp = peer_layer

            logger.info(f"Layer {i}: MLP({original_params:,}) → PEER({peer_params:,}) params")

        logger.info(f"Total parameter change: {total_original_params:,} → {total_peer_params:,}")
        logger.info(f"Parameter ratio: {total_peer_params / total_original_params:.2f}x")

    def _get_layers_to_replace(self, num_layers: int) -> List[int]:
        """Convert replace_layers specification to list of indices"""
        if isinstance(self.replace_layers, list):
            return self.replace_layers
        elif self.replace_layers == "all":
            return list(range(num_layers))
        elif self.replace_layers == "middle":
            start = num_layers // 4
            end = 3 * num_layers // 4
            return list(range(start, end))
        elif self.replace_layers == "first_half":
            return list(range(num_layers // 2))
        elif self.replace_layers == "last_half":
            return list(range(num_layers // 2, num_layers))
        else:
            raise ValueError(f"Invalid replace_layers: {self.replace_layers}")

    @classmethod
    def from_pretrained_with_surgery_inplace(cls,
                                             model_path: str,
                                             replace_layers="middle",
                                             peer_config=None,
                                             **kwargs):
        """Load and modify model in-place to save memory"""

        logger.info(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=os.getenv("HF_TOKEN"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        # Convert to our class (change __class__)
        model.__class__ = cls

        # Add our attributes
        model.replace_layers = replace_layers
        model.peer_config = peer_config or {}
        model.replaced_layer_indices = []

        # Perform surgery on the existing model
        model._replace_mlp_layers()

        return model

    def get_surgery_info(self) -> Dict[str, Any]:
        """Get detailed information about the surgery performed"""
        total_layers = len(self.model.layers)
        replaced_count = len(self.replaced_layer_indices)

        # Count parameters in different parts
        peer_params = 0
        other_params = 0

        for name, param in self.named_parameters():
            if any(f"layers.{i}.mlp" in name for i in self.replaced_layer_indices):
                peer_params += param.numel()
            else:
                other_params += param.numel()

        info = {
            "model_type": "PEERGemma",
            "base_model": getattr(self.config, '_name_or_path', 'pretrained-gemma'),
            "total_layers": total_layers,
            "replaced_layers": self.replace_layers,
            "replaced_layer_indices": self.replaced_layer_indices,
            "replaced_count": replaced_count,
            "replacement_ratio": replaced_count / total_layers,
            "peer_config": self.peer_config,
            "parameter_counts": {
                "peer_parameters": peer_params,
                "other_parameters": other_params,
                "total_parameters": peer_params + other_params,
                "peer_ratio": peer_params / (peer_params + other_params)
            },
            "model_config": {
                "hidden_size": self.config.hidden_size,
                "intermediate_size": self.config.intermediate_size,
                "num_attention_heads": self.config.num_attention_heads,
                "num_hidden_layers": self.config.num_hidden_layers,
            }
        }
        return info

    def get_peer_layers(self) -> List[PEER]:
        """Get all PEER layers in the model"""
        peer_layers = []
        for i in self.replaced_layer_indices:
            peer_layers.append(self.model.layers[i].mlp)
        return peer_layers

    def freeze_non_peer_parameters(self):
        """Freeze all parameters except PEER layers"""
        frozen_count = 0
        unfrozen_count = 0

        for name, param in self.named_parameters():
            if any(f"layers.{i}.mlp" in name for i in self.replaced_layer_indices):
                param.requires_grad = True
                unfrozen_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        logger.info(f"Froze {frozen_count:,} parameters, kept {unfrozen_count:,} trainable")
        return frozen_count, unfrozen_count

    def unfreeze_all_parameters(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")


# def create_custom_gemma_config(
#         hidden_size: int = 512,
#         num_layers: int = 8,
#         num_heads: int = 8,
#         intermediate_size: int = 1024,
#         vocab_size: int = 1000
# ) -> GemmaConfig:
#     """Create a small custom Gemma config for testing"""
#     config = GemmaConfig(
#         vocab_size=vocab_size,
#         hidden_size=hidden_size,
#         intermediate_size=intermediate_size,
#         num_hidden_layers=num_layers,
#         num_attention_heads=num_heads,
#         num_key_value_heads=num_heads,
#         head_dim=hidden_size // num_heads,
#         max_position_embeddings=2048,
#         rms_norm_eps=1e-6,
#         rope_theta=10000.0,
#         attention_bias=False,
#         attention_dropout=0.0,
#         mlp_bias=False,
#     )
#     return config


# def test_peer_surgery_pretrained():
#     """Test PEER surgery on a pretrained model"""
#
#     logger.remove()
#     logger.add(
#         sys.stderr,
#         format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
#         level="INFO"
#     )
#
#     logger.info("🚀 Starting PEER surgery test on pretrained model")
#
#     try:
#         # Test with small pretrained model first
#         model_name = "google/gemma-2b"  # Smaller model for testing
#
#         logger.info(f"Loading pretrained model: {model_name}")
#
#         # Create PEER model from pretrained
#         peer_model = PEERGemmaForCausalLM.from_pretrained_with_surgery(
#             model_name,
#             replace_layers="middle",
#             peer_config={
#                 "num_experts": 50_000,
#                 "heads": 8,
#                 "num_experts_per_head": 16
#             },
#             torch_dtype=torch.bfloat16,
#             device_map="auto" if torch.cuda.is_available() else None,
#         )
#         scratch_dir = os.getenv("SCRATCH.DIR")
#         peer_model.save_pretrained(f"{scratch_dir}/models/peer_gemma_7b_ready")
#         # Get surgery info
#         surgery_info = peer_model.get_surgery_info()
#
#         # Log results
#         logger.success("✅ PEER surgery successful!")
#         logger.info(f"📊 Replaced layers: {surgery_info['replaced_layer_indices']}")
#         logger.info(f"📈 Total parameters: {surgery_info['parameter_counts']['total_parameters']:,}")
#         logger.info(f"🎯 PEER parameters: {surgery_info['parameter_counts']['peer_parameters']:,}")
#         logger.info(f"📊 PEER ratio: {surgery_info['parameter_counts']['peer_ratio']:.3f}")
#
#         # Test forward pass
#         logger.info("Testing forward pass...")
#         from transformers import AutoTokenizer
#
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#
#         test_text = "The quick brown fox"
#         inputs = tokenizer(test_text, return_tensors="pt")
#
#         if torch.cuda.is_available():
#             inputs = {k: v.cuda() for k, v in inputs.items()}
#             peer_model = peer_model.cuda()
#
#         with torch.no_grad():
#             outputs = peer_model(**inputs)
#             logger.success(f"✅ Forward pass successful! Output shape: {outputs.logits.shape}")
#
#         # Test PEER layer access
#         peer_layers = peer_model.get_peer_layers()
#         logger.info(f"🔍 Found {len(peer_layers)} PEER layers")
#
#         logger.success("🎉 Pretrained PEER surgery test completed!")
#         return True
#
#     except Exception as e:
#         logger.error(f"❌ Test failed: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return False
