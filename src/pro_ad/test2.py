import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification
from timm.data.transforms_factory import create_transform

MODEL_PATH = "/data2/yijin/MambaVision-L3-512-21K"

class MemoryModule(nn.Module):
    """
    Learnable memory module that stores prototypes representing normal patterns.
    """

    def __init__(
        self, num_prototypes: int, prototype_dim: int, temperature: float = 1.0
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.temperature = temperature

        # Initialize prototypes with Xavier uniform initialization
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Query the memory with input features.

        Args:
            features: Input features of shape (B, C, H, W) or (B, N, D)

        Returns:
            retrieved_features: Memory-retrieved features same shape as input
            attention_weights: Attention weights (B, spatial_dims, K) where K=num_prototypes
            entropy_scores: Entropy of attention weights (B, spatial_dims)
        """
        original_shape = features.shape

        # Handle different input shapes
        if len(features.shape) == 4:  # (B, C, H, W)
            B, C, H, W = features.shape
            features_flat = features.permute(0, 2, 3, 1).reshape(
                B, H * W, C
            )  # (B, H*W, C)
        else:  # (B, N, D)
            B, N, D = features.shape
            features_flat = features
            C = D

        # Normalize features and prototypes for cosine similarity
        features_norm = F.normalize(features_flat, dim=-1)  # (B, N, C)
        prototypes_norm = F.normalize(self.prototypes, dim=-1)  # (K, C)

        # Compute similarity scores
        similarities = torch.matmul(features_norm, prototypes_norm.t())  # (B, N, K)
        similarities = similarities / self.temperature

        # Compute attention weights
        attention_weights = F.softmax(similarities, dim=-1)  # (B, N, K)

        # Retrieve from memory using attention weights
        retrieved_features_flat = torch.matmul(
            attention_weights, self.prototypes
        )  # (B, N, C)

        # Compute entropy for anomaly detection
        entropy_scores = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-8), dim=-1
        )  # (B, N)

        # Reshape back to original shape
        if len(original_shape) == 4:
            retrieved_features = retrieved_features_flat.reshape(B, H, W, C).permute(
                0, 3, 1, 2
            )
            entropy_scores = entropy_scores.reshape(B, H, W)
        else:
            retrieved_features = retrieved_features_flat

        return retrieved_features, attention_weights, entropy_scores

    def expand_memory(self, new_prototypes: int):
        """
        Expand memory for continual learning by adding new prototype slots.
        """
        old_prototypes = self.prototypes.data
        new_proto_data = torch.empty(new_prototypes, self.prototype_dim)
        nn.init.xavier_uniform_(new_proto_data)

        # Concatenate old and new prototypes
        expanded_prototypes = torch.cat(
            [old_prototypes, new_proto_data.to(old_prototypes.device)], dim=0
        )
        self.prototypes = nn.Parameter(expanded_prototypes)
        self.num_prototypes += new_prototypes


class MultiscaleGatingMechanism(nn.Module):
    """
    Learned gating mechanism to fuse original and memory-retrieved features at multiple scales.
    """

    def __init__(self, feature_dims: List[int]):
        super().__init__()
        self.gates = nn.ModuleList()

        for dim in feature_dims:
            gate_net = nn.Sequential(
                nn.Conv2d(dim * 2, dim // 4, 1),
                nn.ReLU(),
                nn.Conv2d(dim // 4, dim, 1),
                nn.Sigmoid(),
            )
            self.gates.append(gate_net)

    def forward(
        self,
        original_features: List[torch.Tensor],
        retrieved_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Fuse original and retrieved features using learned gating at each scale.
        """
        fused_features = []

        for orig_feat, retr_feat, gate_net in zip(
            original_features, retrieved_features, self.gates
        ):
            # Concatenate features for gate computation
            concat_features = torch.cat([orig_feat, retr_feat], dim=1)
            gate = gate_net(concat_features)

            # Apply gating
            fused_feat = gate * orig_feat + (1 - gate) * retr_feat
            fused_features.append(fused_feat)

        return fused_features


class MultiscaleDecoder(nn.Module):
    """
    Multi-scale decoder to reconstruct images from fused features.
    """

    def __init__(
        self, feature_dims: List[int], output_channels: int = 3, target_size: int = 512
    ):
        super().__init__()
        self.target_size = target_size
        self.output_channels = output_channels

        # Decoder blocks for each scale
        self.decoders = nn.ModuleList()

        for i, dim in enumerate(feature_dims):
            # Calculate upsampling factor based on stage
            if i == 0:  # Stage 1: 128x128 -> 512x512 (4x)
                upsample_factor = 4
            elif i == 1:  # Stage 2: 64x64 -> 512x512 (8x)
                upsample_factor = 8
            elif i == 2:  # Stage 3: 32x32 -> 512x512 (16x)
                upsample_factor = 16
            else:  # Stage 4: 16x16 -> 512x512 (32x)
                upsample_factor = 32

            decoder = nn.Sequential(
                nn.Conv2d(dim, dim // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim // 2, dim // 4, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(
                    scale_factor=upsample_factor, mode="bilinear", align_corners=False
                ),
                nn.Conv2d(dim // 4, output_channels, 3, padding=1),
            )
            self.decoders.append(decoder)

        # Final fusion layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                output_channels * len(feature_dims), output_channels * 2, 3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(output_channels * 2, output_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, fused_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode multi-scale fused features back to image.
        """
        decoded_features = []

        for feat, decoder in zip(fused_features, self.decoders):
            decoded = decoder(feat)
            # Ensure all decoded features have the same spatial size
            if decoded.shape[-1] != self.target_size:
                decoded = F.interpolate(
                    decoded,
                    size=(self.target_size, self.target_size),
                    mode="bilinear",
                    align_corners=False,
                )
            decoded_features.append(decoded)

        # Concatenate all decoded features
        concat_decoded = torch.cat(decoded_features, dim=1)

        # Final reconstruction
        reconstructed = self.final_conv(concat_decoded)

        return reconstructed


class MemVimMamba(nn.Module):
    """
    Memory-Augmented MambaVision for Continual Anomaly Detection.
    """

    def __init__(
        self,
        mamba_model_name: str = MODEL_PATH,
        num_prototypes: int = 512,
        temperature: float = 1.0,
        input_size: int = 512,
        alpha: float = 0.1,  # Sparsity loss weight
        beta: float = 0.25,  # Commitment loss weight
        lambda_weight: float = 0.5,  # Memory vs reconstruction weight
        use_stages: List[int] = [1, 2, 3, 4],  # Which stages to use (1-4)
    ):
        super().__init__()

        # Load pre-trained MambaVision model
        self.mamba_encoder = AutoModel.from_pretrained(
            mamba_model_name, trust_remote_code=True, local_files_only=True
        )
        self.input_size = input_size
        self.use_stages = use_stages

        # Create transform for preprocessing
        input_resolution = (3, input_size, input_size)
        self.transform = create_transform(
            input_size=input_resolution,
            is_training=False,
            mean=self.mamba_encoder.config.mean,
            std=self.mamba_encoder.config.std,
            crop_mode=self.mamba_encoder.config.crop_mode,
            crop_pct=self.mamba_encoder.config.crop_pct,
        )

        # Get feature dimensions by running a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size, input_size)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                self.mamba_encoder = self.mamba_encoder.cuda()

            _, features = self.mamba_encoder(dummy_input)

            # Extract dimensions for selected stages
            self.feature_dims = []
            for stage_idx in use_stages:
                self.feature_dims.append(
                    features[stage_idx - 1].shape[1]
                )  # Channel dimension

        # Memory modules for each selected stage
        self.memories = nn.ModuleList()
        for dim in self.feature_dims:
            memory = MemoryModule(num_prototypes, dim, temperature)
            self.memories.append(memory)

        # Multi-scale gating mechanism
        self.gating = MultiscaleGatingMechanism(self.feature_dims)

        # Multi-scale decoder
        self.decoder = MultiscaleDecoder(
            self.feature_dims, output_channels=3, target_size=input_size
        )

        # Loss weights
        self.alpha = alpha
        self.beta = beta
        self.lambda_weight = lambda_weight

        # For continual learning - track which prototypes to freeze
        self.frozen_prototypes = 0

        # Freeze the MambaVision encoder to prevent catastrophic forgetting
        for param in self.mamba_encoder.parameters():
            param.requires_grad = False

        # Set encoder to eval mode for feature extraction
        self.mamba_encoder.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Mem-Vim-Mamba model.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Dictionary containing reconstructed images, features, attention weights, etc.
        """
        with torch.no_grad():
            # Extract features using MambaVision
            _, all_features = self.mamba_encoder(x)

        # Select features from specified stages
        selected_features = [all_features[i - 1] for i in self.use_stages]

        # Query memory for each scale
        retrieved_features = []
        all_attention_weights = []
        all_entropy_scores = []

        for feat, memory in zip(selected_features, self.memories):
            retrieved_feat, attention_weights, entropy_scores = memory(feat)
            retrieved_features.append(retrieved_feat)
            all_attention_weights.append(attention_weights)
            all_entropy_scores.append(entropy_scores)

        # Fuse original and retrieved features at each scale
        fused_features = self.gating(selected_features, retrieved_features)

        # Decode to reconstruct image
        reconstructed = self.decoder(fused_features)

        return {
            "reconstructed": reconstructed,
            "original_features": selected_features,
            "retrieved_features": retrieved_features,
            "fused_features": fused_features,
            "attention_weights": all_attention_weights,
            "entropy_scores": all_entropy_scores,
        }

    def compute_loss(
        self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the multi-part loss function.
        """
        reconstructed = outputs["reconstructed"]
        original_features = outputs["original_features"]
        retrieved_features = outputs["retrieved_features"]
        all_attention_weights = outputs["attention_weights"]

        # 1. Reconstruction loss (L1)
        rec_loss = F.l1_loss(reconstructed, x)

        # 2. Sparsity loss (entropy minimization) - averaged across all scales
        sparse_loss = 0
        for attention_weights in all_attention_weights:
            if len(attention_weights.shape) == 4:  # (B, H, W, K)
                entropy = -torch.sum(
                    attention_weights * torch.log(attention_weights + 1e-8), dim=-1
                )
            else:  # (B, N, K)
                entropy = -torch.sum(
                    attention_weights * torch.log(attention_weights + 1e-8), dim=-1
                )
            sparse_loss += torch.mean(entropy)
        sparse_loss /= len(all_attention_weights)

        # 3. Commitment loss (VQ-VAE style) - averaged across all scales
        commit_loss = 0
        for orig_feat, retr_feat in zip(original_features, retrieved_features):
            commit_loss += F.mse_loss(orig_feat.detach(), retr_feat)
        commit_loss /= len(original_features)

        # Total loss
        total_loss = rec_loss + self.alpha * sparse_loss + self.beta * commit_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": rec_loss,
            "sparsity_loss": sparse_loss,
            "commitment_loss": commit_loss,
        }

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

    def get_parameter_count(self):
        """
        Get detailed parameter count information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.mamba_encoder.parameters())
        memory_params = sum(
            p.numel() for memory in self.memories for p in memory.parameters()
        )
        other_params = trainable_params - memory_params

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "encoder_parameters": encoder_params,
            "memory_parameters": memory_params,
            "other_trainable_parameters": other_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
        }

    def get_trainable_params_for_continual_learning(self):
        """
        Get parameters that should be trained during continual learning.
        Freezes old prototypes but allows training of new ones.
        """
        params = []

        # Add all parameters except encoder
        for name, param in self.named_parameters():
            if "mamba_encoder" in name:
                # Keep encoder frozen
                param.requires_grad = False
                continue
            elif "memories" in name and "prototypes" in name:
                # Only train new prototypes for continual learning
                if self.frozen_prototypes > 0:
                    # Split the parameter: freeze old prototypes, train new ones
                    old_prototypes = param[: self.frozen_prototypes]
                    new_prototypes = param[self.frozen_prototypes :]

                    # Freeze old prototypes
                    old_prototypes.requires_grad = False

                    # Add new prototypes to training
                    if new_prototypes.numel() > 0:
                        params.append(new_prototypes)
                else:
                    params.append(param)
            else:
                params.append(param)

        return params


# Training utilities
class MemVimMambaTrainer:
    """
    Training utilities for Mem-Vim-Mamba model.
    """

    def __init__(self, model: MemVimMamba, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device

    def train_on_normal_class(
        self,
        dataloader,
        epochs: int = 100,
        lr: float = 1e-4,
        continual_learning: bool = False,
        exemplar_loader=None,
    ):
        """
        Train the model on normal samples from a class.
        """
        if continual_learning:
            # Get parameters for continual learning
            optimizer = torch.optim.Adam(
                self.model.get_trainable_params_for_continual_learning(), lr=lr
            )
        else:
            optimizer = torch.optim.Adam(
                [p for p in self.model.parameters() if p.requires_grad], lr=lr
            )

        # Set training mode for trainable parts only
        for name, module in self.model.named_modules():
            if "mamba_encoder" in name:
                module.eval()  # Keep encoder in eval mode
            else:
                module.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device)

                # Mix with exemplars if doing continual learning
                if continual_learning and exemplar_loader is not None:
                    try:
                        exemplar_images, _ = next(iter(exemplar_loader))
                        exemplar_images = exemplar_images.to(self.device)

                        # Mix current batch with exemplars
                        batch_size = min(images.shape[0], exemplar_images.shape[0])
                        mixed_images = torch.cat(
                            [
                                images[: batch_size // 2],
                                exemplar_images[: batch_size // 2],
                            ],
                            dim=0,
                        )
                        images = mixed_images
                    except StopIteration:
                        pass

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)

                # Compute loss
                losses = self.model.compute_loss(images, outputs)

                # Backward pass
                losses["total_loss"].backward()
                optimizer.step()

                total_loss += losses["total_loss"].item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}")

    def evaluate_anomaly_detection(self, test_loader, normal_loader=None):
        """
        Evaluate anomaly detection performance.
        """
        self.model.eval()

        all_scores = []
        all_labels = []

        # Test on normal samples
        if normal_loader:
            for images, _ in normal_loader:
                images = images.to(self.device)
                _, image_scores = self.model.compute_anomaly_score(images)

                all_scores.extend(image_scores.cpu().numpy())
                all_labels.extend([0] * len(image_scores))  # 0 for normal

        # Test on anomalous samples
        for images, labels in test_loader:
            images = images.to(self.device)
            _, image_scores = self.model.compute_anomaly_score(images)

            all_scores.extend(image_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return np.array(all_scores), np.array(all_labels)


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = MemVimMamba(
        mamba_model_name=MODEL_PATH,
        num_prototypes=512,
        temperature=0.1,
        input_size=512,
        alpha=0.1,
        beta=0.25,
        lambda_weight=0.5,
        use_stages=[2, 3, 4],  # Use stages 2, 3, 4 for multi-scale processing
    )

    # Create trainer
    trainer = MemVimMambaTrainer(
        model, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Mem-Vim-Mamba model initialized successfully!")

    # Get detailed parameter information
    param_info = model.get_parameter_count()
    print(f"Total parameters: {param_info['total_parameters']:,}")
    print(
        f"Trainable parameters: {param_info['trainable_parameters']:,} ({param_info['trainable_percentage']:.1f}%)"
    )
    print(f"Encoder parameters (frozen): {param_info['encoder_parameters']:,}")
    print(f"Memory parameters: {param_info['memory_parameters']:,}")
    print(f"Other trainable parameters: {param_info['other_trainable_parameters']:,}")
    print(f"Memory prototypes per scale: {model.memories[0].num_prototypes}")
    print(f"Feature dimensions: {model.feature_dims}")

    # Example: Test with a single image
    dummy_input = torch.randn(2, 3, 512, 512)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()

    with torch.no_grad():
        outputs = model(dummy_input)
        pixel_scores, image_scores = model.compute_anomaly_score(dummy_input)

    print(f"Reconstruction shape: {outputs['reconstructed'].shape}")
    print(f"Pixel anomaly scores shape: {pixel_scores.shape}")
    print(f"Image anomaly scores shape: {image_scores.shape}")
