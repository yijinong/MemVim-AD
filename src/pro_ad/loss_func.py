from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model.hierarchy_proto import HierarchicalPrototypeMemory


class ContrastiveLoss(nn.Module):
    """Contrastive loss for feature learning"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, features: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = features.shape[0]

        if labels is None:
            # Self-supervised contrastive loss (SimCLR style)
            # Assume features come in pairs (original, augmented)
            assert batch_size % 2 == 0, (
                "Batch size must be even for self-supervised learning"
            )

            # Normalize features
            features = F.normalize(features, dim=1)

            # Compute similarity matrix
            similarity_matrix = torch.mm(features, features.t()) / self.temperature

            # Create positive pairs mask
            batch_size_half = batch_size // 2
            labels_idx = torch.arange(batch_size_half, device=features.device).repeat(2)
            labels_idx[1::2] = labels_idx[::2]  # [0,0,1,1,2,2,...]

            # Create mask for positive pairs (same augmented pair)
            mask = torch.eq(labels_idx.unsqueeze(1), labels_idx.unsqueeze(0)).float()
            mask = mask - torch.eye(
                batch_size, device=features.device
            )  # Remove diagonal

            # Compute InfoNCE loss
            exp_sim = torch.exp(similarity_matrix)

            # For numerical stability, use log-sum-exp trick
            max_sim = torch.max(similarity_matrix, dim=1, keepdim=True)[0]
            exp_sim_stable = torch.exp(similarity_matrix - max_sim)

            # Positive similarities
            pos_sim = (mask * exp_sim_stable).sum(dim=1)

            # All similarities (excluding self)
            mask_diag = torch.eye(batch_size, device=features.device, dtype=torch.bool)
            all_sim = exp_sim_stable.masked_fill(mask_diag, 0).sum(dim=1)

            # InfoNCE loss
            loss = -torch.log(
                pos_sim / (all_sim + 1e-8)
            )  # Add small epsilon for stability
            return loss.mean()
        else:
            # Supervised contrastive loss
            features = F.normalize(features, dim=1)
            similarity_matrix = torch.mm(features, features.t()) / self.temperature

            # Create mask for same class samples
            labels = labels.unsqueeze(1)
            mask = torch.eq(labels, labels.t()).float()
            mask = mask - torch.eye(batch_size, device=features.device)

            exp_sim = torch.exp(similarity_matrix)
            log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

            mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(
                min=1
            )
            loss = -mean_log_prob_pos.mean()

            return loss


class PrototypeAlignmentLoss(nn.Module):
    """
    Loss to align features with prototypes

    This loss encourages features to be close to their assigned prototypes
    in the hierarchical prototype memory structure.
    """

    def __init__(self, use_vectorized: bool = True):
        super().__init__()
        self.use_vectorized = use_vectorized

    def forward(
        self, features: torch.Tensor, prototype_memory: HierarchicalPrototypeMemory
    ) -> torch.Tensor:
        if prototype_memory.coarse_prototypes is None:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        if self.use_vectorized:
            return self._vectorized_forward(features, prototype_memory)
        else:
            return self._loop_forward(features, prototype_memory)

    def _vectorized_forward(
        self, features: torch.Tensor, prototype_memory: HierarchicalPrototypeMemory
    ) -> torch.Tensor:
        """Fully vectorized implementation for better performance"""
        device = features.device
        batch_size = features.shape[0]

        # Type guard to ensure coarse_prototypes is not None
        assert prototype_memory.coarse_prototypes is not None, (
            "Coarse prototypes must be initialized"
        )

        # Find closest coarse prototypes for all features
        coarse_distances = torch.cdist(features, prototype_memory.coarse_prototypes)
        closest_coarse_indices = torch.argmin(coarse_distances, dim=1)

        # Initialize loss tensor
        losses = torch.zeros(batch_size, device=device)

        # Process each unique coarse cluster
        unique_coarse_indices = torch.unique(closest_coarse_indices)

        for coarse_idx in unique_coarse_indices:
            coarse_idx_int = coarse_idx.item()

            # Get mask for features assigned to this coarse prototype
            mask = closest_coarse_indices == coarse_idx
            batch_features = features[mask]

            if coarse_idx_int in prototype_memory.fine_prototypes:
                fine_prototypes = prototype_memory.fine_prototypes[coarse_idx_int]
                if fine_prototypes.shape[0] > 0:
                    # Compute distances to all fine prototypes for this cluster
                    fine_distances = torch.cdist(batch_features, fine_prototypes)
                    min_distances = torch.min(fine_distances, dim=1)[0]
                    losses[mask] = min_distances
                else:
                    # No fine prototypes, use coarse distance
                    losses[mask] = coarse_distances[mask, coarse_idx]
            else:
                # No fine prototypes for this coarse cluster
                losses[mask] = coarse_distances[mask, coarse_idx]

        return losses.mean()

    def _loop_forward(
        self, features: torch.Tensor, prototype_memory: HierarchicalPrototypeMemory
    ) -> torch.Tensor:
        """Original loop-based implementation"""
        batch_size = features.shape[0]
        device = features.device

        # Type guard to ensure coarse_prototypes is not None
        assert prototype_memory.coarse_prototypes is not None, (
            "Coarse prototypes must be initialized"
        )

        # Initialize total loss as tensor
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Vectorized approach for better efficiency
        # Find closest coarse prototypes for all features at once
        coarse_distances = torch.cdist(
            features, prototype_memory.coarse_prototypes
        )  # (B, K_coarse)
        closest_coarse_indices = torch.argmin(coarse_distances, dim=1)  # (B,)

        for i in range(batch_size):
            feature = features[i : i + 1]  # (1, feature_dim)
            closest_coarse_idx = closest_coarse_indices[i].item()

            # Compute distance to closest fine prototype
            if closest_coarse_idx in prototype_memory.fine_prototypes:
                fine_prototypes = prototype_memory.fine_prototypes[closest_coarse_idx]
                if fine_prototypes.shape[0] > 0:  # Check if fine prototypes exist
                    fine_distances = torch.cdist(feature, fine_prototypes)
                    min_fine_distance = torch.min(fine_distances)
                    total_loss = total_loss + min_fine_distance
            else:
                # If no fine prototypes, use distance to coarse prototype
                coarse_distance = coarse_distances[i, int(closest_coarse_idx)]
                total_loss = total_loss + coarse_distance

        return total_loss / batch_size
