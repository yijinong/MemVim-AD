from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .decoder import MultiScaleDecoder
from .gating_mech import MultiScaledGatingMechanism
from .hierarchy_proto import HierarchicalPrototypeMemory
from ..loss_func import ContrastiveLoss, PrototypeAlignmentLoss

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

        # Hierarchical memory modules for each selected stage
        from .hierarchy_proto import HierarchicalPrototypeMemory
        self.k_coarse = 8  # reasonable default for coarse clusters
        self.k_fine = 16   # reasonable default for fine clusters per coarse
        self.memories = nn.ModuleList()
        for dim in self.feat_dims:
            memory = HierarchicalPrototypeMemory(feature_dim=dim, k_coarse=self.k_coarse, k_fine=self.k_fine)
            self.memories.append(memory)

        # Multi-scale gating mechanism
        self.gating = MultiScaledGatingMechanism(self.feat_dims)

        # Multi-scale decoder
        self.decoder = MultiScaleDecoder(
            self.feat_dims, out_channels=3, target_size=input_size
        )

        # Loss weights
        self.alpha = alpha  # weight for contrastive loss
        self.beta = beta    # weight for memory loss
        self.lambda_weight = lambda_weight  # weight for prototype alignment loss

        # Loss functions
        self.contrastive_loss = ContrastiveLoss(temperature=temp)
        self.proto_loss = PrototypeAlignmentLoss(use_vectorized=True)
        
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

        # Initialize hierarchical prototypes if not done
        for feat, mem in zip(selected_feats, self.memories):
            if mem.coarse_prototypes is None:
                mem.initialize_prototypes(feat.detach())

        # For demonstration, anomaly scores from hierarchical memory
        anomaly_scores = [mem.compute_anomaly_score(feat) for feat, mem in zip(selected_feats, self.memories)]

        # For compatibility, set retrieved_feats, attn_weights, entropy_scores to None
        retrieved_feats = [None for _ in selected_feats]
        all_attn_weights = [None for _ in selected_feats]
        all_entropy_scores = [None for _ in selected_feats]

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
            "anomaly_scores": anomaly_scores,
            "memories": self.memories,
        }

    def compute_loss(self, inputs: torch.Tensor, outputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the total loss for training.
        
        Args:
            inputs: The original input images
            outputs: The dictionary of outputs from forward pass
            
        Returns:
            total_loss: The combined loss for optimization
            loss_dict: Dictionary containing individual loss terms
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(outputs["reconstructed"], inputs)
        
        # Contrastive loss on features
        contrast_loss = sum(
            self.contrastive_loss(feat) for feat in outputs["features"]
        ) / len(outputs["features"])
        
        # Prototype alignment loss for each scale (using hierarchical memory)
        proto_loss = 0.0
        for feat, memory_module in zip(outputs["features"], outputs["memories"]):
            proto_loss += self.proto_loss(feat, memory_module)
        proto_loss /= len(outputs["features"])
        # For demonstration, set memory_loss to 0.0 (can be replaced with hierarchical entropy or other metric)
        memory_loss = 0.0
        
        # Combine losses with weights
        total_loss = (
            recon_loss + 
            self.alpha * contrast_loss + 
            self.beta * memory_loss +
            self.lambda_weight * proto_loss
        )
        
        loss_dict = {
            "total": total_loss.item(),
            "reconstruction": recon_loss.item(),
            "contrastive": contrast_loss.item(),
            "memory": memory_loss.item(),
            "prototype": proto_loss.item(),
        }
        
        return total_loss, loss_dict

    def compute_anomaly_score(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pixel-level and image-level anomaly scores.

        Returns:
            pixel_scores: Pixel-level anomaly map (B, H, W)
            image_scores: Image-level anomaly scores (B,)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            reconstructed = outputs["reconstructed"]
            all_entropy_scores = outputs["entropy_scores"]

            # Reconstruction-based anomaly map
            rec_anomaly = torch.mean(torch.abs(x - reconstructed), dim=1)  # (B, H, W)

            # Memory-based anomaly map - combine entropy from all scales
            mem_anomaly = torch.zeros_like(rec_anomaly)

            for entropy_scores in all_entropy_scores:
                if len(entropy_scores.shape) == 3:  # (B, H, W)
                    # Upsample to target size if needed
                    if entropy_scores.shape[-1] != self.input_size:
                        entropy_upsampled = F.interpolate(
                            entropy_scores.unsqueeze(1),
                            size=(self.input_size, self.input_size),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)
                    else:
                        entropy_upsampled = entropy_scores
                else:  # (B, N) - need to reshape to spatial
                    B, N = entropy_scores.shape
                    # Infer spatial dimensions from feature maps
                    spatial_size = int(math.sqrt(N))
                    entropy_spatial = entropy_scores.view(B, spatial_size, spatial_size)
                    entropy_upsampled = F.interpolate(
                        entropy_spatial.unsqueeze(1),
                        size=(self.input_size, self.input_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)

                mem_anomaly += entropy_upsampled

            # Average entropy across scales
            mem_anomaly /= len(all_entropy_scores)

            # Combine anomaly maps
            pixel_scores = (
                1 - self.lambda_weight
            ) * rec_anomaly + self.lambda_weight * mem_anomaly

            # Image-level scores (max pooling)
            image_scores = torch.max(
                pixel_scores.view(pixel_scores.shape[0], -1), dim=1
            )[0]

            return pixel_scores, image_scores

    def expand_for_new_class(self, new_prototypes: int = 128):
        """
        Expand memory for continual learning.
        """
        # Mark current prototypes as frozen
        self.frozen_prototypes = self.memories[0].num_prototypes

        # Expand memory for all scales
        for memory in self.memories:
            memory.expand_memory(new_prototypes)

        print(
            f"Memory expanded for all scales: {self.frozen_prototypes} -> {self.memories[0].num_prototypes} prototypes"
        )


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

    # Create dummy input (batch of 2 RGB images, 512x512)
    batch_size = 2
    input_size = 512
    inputs = torch.randn(batch_size, 3, input_size, input_size)

    # Forward pass
    outputs = model(inputs)

    # Compute loss
    total_loss, loss_dict = model.compute_loss(inputs, outputs)
    print("Loss breakdown:", loss_dict)

    # Print anomaly scores
    print("Anomaly scores (per scale):")
    for i, score in enumerate(outputs["anomaly_scores"]):
        print(f"Scale {i+1}: shape {score.shape}")

if __name__ == "__main__":
    main()
