from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModel

from .decoder import MultiScaleDecoder
from .gating_mech import MultiScaledGatingMechanism
from .memory import MemoryModule

MODEL_PATH = "/data2/yijin/MambaVision-L3-512-21K"


class MemVim(nn.Module):
    """
    Memory-augmented Vision Transformer (MemVim) model.
    Combines MambaVision backbone with memory module for enhanced feature representation.
    """

    def __init__(
        self,
        base_model: str = MODEL_PATH,
        num_proto: int = 512,
        temp: float = 1.0,
        input_size: int = 512,
        alpha: float = 0.1,
        beta: float = 0.25,
        lambda_weight: float = 0.5,
        use_stages: List[int] = [1, 2, 3, 4],
        run_dummy_forward: bool = False,
    ):
        super().__init__()
        # load pre-trained MambaVision model
        self.mamba_encoder = AutoModel.from_pretrained(
            base_model, trust_remote_code=True, local_files_only=True
        )
        self.input_size = input_size
        self.use_stages = use_stages

        # Get feature dimensions from model config (official way)
        config = getattr(self.mamba_encoder, "config", None)
        if config is not None:
            in_dim = getattr(config, "in_dim", 64)
            dim = getattr(config, "dim", 256)
            all_feat_dims = [in_dim, dim, dim * 2, dim * 4]
        else:
            # Fallback to default values if config is not available
            all_feat_dims = [64, 256, 512, 1024]

        # Extract feature dimensions for selected stages
        with torch.no_grad():
            self.feat_dims = []
            for stage_idx in use_stages:
                self.feat_dims.append(all_feat_dims[stage_idx - 1])

        # Optionally run dummy forward pass for validation if requested
        if run_dummy_forward:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, input_size, input_size)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                    self.mamba_encoder = self.mamba_encoder.cuda()

                _, feats = self.mamba_encoder(dummy_input)

                # Validate our config-based dimensions match actual feature dimensions
                for i, stage_idx in enumerate(use_stages):
                    actual_dim = feats[stage_idx - 1].shape[1]
                    expected_dim = self.feat_dims[i]
                    if actual_dim != expected_dim:
                        print(
                            f"Warning: Stage {stage_idx} dimension mismatch: "
                            f"config says {expected_dim}, actual is {actual_dim}"
                        )
                        self.feat_dims[i] = actual_dim  # Use actual dimension

        # Memory modules for each selected stage
        self.memories = nn.ModuleList()
        for dim in self.feat_dims:
            memory = MemoryModule(num_proto, dim, temp)
            self.memories.append(memory)

        # Multi-scale gating mechanism
        self.gating = MultiScaledGatingMechanism(self.feat_dims)

        # Multi-scale decoder
        self.decoder = MultiScaleDecoder(
            self.feat_dims, out_channels=3, target_size=input_size
        )

        # Loss weights
        self.alpha = alpha
        self.beta = beta
        self.lambda_weight = lambda_weight

        # For continual learning
        # Track which prototypes to freeze
        self.frozen_proto = 0

        # Freeze the MambaVision encoder to prevent catastrophic forgetting
        for param in self.mamba_encoder.parameters():
            param.requires_grad = False

        # set encoder to eval mode for feature extraction
        self.mamba_encoder.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the MemVim model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Dictionary with reconstructed image, features, attention maps, and anomaly scores.
        """
        with torch.no_grad():
            # Extract features using MambaVision
            _, all_feats = self.mamba_encoder(x)

        # Select features from specified stages
        selected_feats = [all_feats[i - 1] for i in self.use_stages]

        # Query memory for each scale
        retrieved_feats = []
        all_attn_weights = []
        all_entropy_scores = []

        for feat, mem in zip(selected_feats, self.memories):
            retrieved_feat, attn_weights, entropy_scores = mem(feat)
            retrieved_feats.append(retrieved_feat)
            all_attn_weights.append(attn_weights)
            all_entropy_scores.append(entropy_scores)

        # Fuse original and retrieved features at each scale
        fused_feats = self.gating(selected_feats, retrieved_feats)

        # decode to reconstruct the image
        reconstructed = self.decoder(fused_feats)

        return {
            "reconstructed": reconstructed,
            "features": selected_feats,
            "retrieved_features": retrieved_feats,
            "attention_weights": all_attn_weights,
            "entropy_scores": all_entropy_scores,
        }

    def get_param_count(self):
        """
        Get the total number of trainable parameters in the model.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.mamba_encoder.parameters())
        memory_params = sum(p.numel() for p in self.memories.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        other_params = total_params - (encoder_params + memory_params + decoder_params)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "encoder_params": encoder_params,
            "memory_params": memory_params,
            "decoder_params": decoder_params,
            "other_params": other_params,
            "trainable_params_pct": (trainable_params / total_params) * 100,
        }

    def get_trainable_params_cl(self):
        """
        Get parameters that should be trained during continual learning.
        Freeze old prototypes but allows training of new ones
        """

        params = []
        for name, param in self.named_parameters():
            if "mamba_encoder" in name:
                param.requires_grad = False
                continue
            elif "memories" in name and "prototype" in name:
                # Only train new prototypes for continual learning
                if self.frozen_proto > 0:
                    # Split the parametes: Freeze old prototypes, train a new one
                    old_proto = param[: self.frozen_proto]
                    new_proto = param[self.frozen_proto :]

                    # Freeze old prototypes
                    old_proto.requires_grad = False

                    # Add new prototypes to training
                    if new_proto.numel() > 0:
                        params.append(new_proto)
                else:
                    # If no prototypes are frozen, train all
                    params.append(param)
            else:
                # Train all other parameters
                params.append(param)
        return params


def main() -> None:
    # Initialize model
    model = MemVim(
        base_model=MODEL_PATH,
        num_proto=512,
        temp=0.1,
        input_size=512,
        alpha=0.1,
        beta=0.25,
        lambda_weight=0.5,
        use_stages=[2, 3, 4],  # Use stages 2, 3, 4 for multi-scale processing
    )

    param_info = model.get_param_count()
    print(f"Total parameters: {param_info['total_params']}")
    print(
        f"Trainable parameters: {param_info['trainable_params']} ({param_info['trainable_params']:.2f}%)"
    )
    print(f"Encoder parameters: {param_info['encoder_params']}")
    print(f"Memory parameters: {param_info['memory_params']}")
    print(f"Decoder parameters: {param_info['decoder_params']}")
    print(f"Other parameters: {param_info['other_params']}")

    print(f"Memory prototypes per scale: {model.memories[0].num_proto}")
    print(f"Feature dimensions per scale: {model.feat_dims}")


if __name__ == "__main__":
    main()
