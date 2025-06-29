import torch
from transformers import GemmaForCausalLM, GemmaConfig
from PEER_pytorch import PEER
from loguru import logger
from typing import List, Dict, Any
import json
import sys


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
            "heads": 8,
            "num_experts": 10_000,  # Reduced for testing
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
    def from_pretrained_with_surgery(cls,
                                     model_name_or_path,
                                     replace_layers="middle",
                                     peer_config=None,
                                     **kwargs):
        """Load pretrained Gemma and perform surgery"""
        logger.info(f"Loading model from: {model_name_or_path}")

        # Load original model first to get config
        logger.info("Loading original model...")
        original_model = GemmaForCausalLM.from_pretrained(model_name_or_path, **kwargs)

        # Create new model with surgery
        logger.info("Performing surgery...")
        model = cls(
            original_model.config,
            replace_layers=replace_layers,
            peer_config=peer_config
        )

        # Copy weights from original model (except replaced MLP layers)
        logger.info("Transferring weights...")
        model_state = model.state_dict()
        original_state = original_model.state_dict()

        # Count successful transfers
        transferred = 0
        skipped = 0

        for name, param in original_state.items():
            if name in model_state and model_state[name].shape == param.shape:
                model_state[name].copy_(param)
                transferred += 1
            else:
                skipped += 1
                if "mlp" not in name:  # Only log non-MLP skips as warnings
                    logger.warning(f"Skipped parameter: {name}")

        logger.info(f"Weight transfer: {transferred} transferred, {skipped} skipped")
        logger.success("Surgery completed successfully!")

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
            "base_model": getattr(self.config, '_name_or_path', 'custom-gemma'),
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


def create_custom_gemma_config(
        hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        intermediate_size: int = 1024,
        vocab_size: int = 1000
) -> GemmaConfig:
    """Create a small custom Gemma config for testing"""
    config = GemmaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=hidden_size // num_heads,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
    )
    return config


def test_peer_surgery():
    """Test PEER surgery on a small custom Gemma model"""

    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    logger.info("🚀 Starting PEER surgery test")

    # Create small custom Gemma model
    logger.info("Creating custom small Gemma model...")
    config = create_custom_gemma_config(
        hidden_size=256,
        num_layers=6,
        num_heads=4,
        intermediate_size=512,
        vocab_size=1000
    )

    # Create original model
    logger.info("Initializing original Gemma model...")
    original_model = GemmaForCausalLM(config)
    original_params = sum(p.numel() for p in original_model.parameters())
    logger.info(f"Original model parameters: {original_params:,}")

    # Test different replacement strategies
    test_configs = [
        {
            "name": "Middle layers",
            "replace_layers": "middle",
            "peer_config": {
                "num_experts": 2500,
                "heads": 4,
                "num_experts_per_head": 8
            }
        },
        {
            "name": "Specific layers",
            "replace_layers": [2, 3],
            "peer_config": {
                "num_experts": 2500,
                "heads": 2,
                "num_experts_per_head": 16
            }
        },
        {
            "name": "First half",
            "replace_layers": "first_half",
            "peer_config": {
                "num_experts": 2500,
                "heads": 8,
                "num_experts_per_head": 4
            }
        }
    ]

    for i, test_config in enumerate(test_configs):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Test {i + 1}: {test_config['name']}")
        logger.info(f"{'=' * 50}")

        try:
            # Perform surgery
            peer_model = PEERGemmaForCausalLM(
                config,
                replace_layers=test_config["replace_layers"],
                peer_config=test_config["peer_config"]
            )

            # Get surgery info
            surgery_info = peer_model.get_surgery_info()

            # Log results
            logger.success(f"✅ Surgery successful!")
            logger.info(f"📊 Replaced layers: {surgery_info['replaced_layer_indices']}")
            logger.info(f"📈 Total parameters: {surgery_info['parameter_counts']['total_parameters']:,}")
            logger.info(f"🎯 PEER parameters: {surgery_info['parameter_counts']['peer_parameters']:,}")
            logger.info(f"📊 PEER ratio: {surgery_info['parameter_counts']['peer_ratio']:.3f}")

            # Test forward pass
            logger.info("Testing forward pass...")
            test_input = torch.randint(0, config.vocab_size, (1, 10))

            with torch.no_grad():
                output = peer_model(test_input)
                logger.success(f"✅ Forward pass successful! Output shape: {output.logits.shape}")

            # Test PEER layer access
            peer_layers = peer_model.get_peer_layers()
            logger.info(f"🔍 Found {len(peer_layers)} PEER layers")

            # Test parameter freezing
            frozen, unfrozen = peer_model.freeze_non_peer_parameters()
            logger.info(f"❄️  Frozen: {frozen:,}, Unfrozen: {unfrozen:,}")

            peer_model.unfreeze_all_parameters()

        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.success("🎉 All tests completed!")


if __name__ == "__main__":
    test_peer_surgery()