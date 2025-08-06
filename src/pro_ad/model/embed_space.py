from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

# from .momentum_update import PrototypeMemoryBank


class LossFunction(nn.Module):
    def __init__(
        self, temp: float = 0.1, lambda_decor: float = 1.8, lambda_contrast: float = 1.0
    ):
        """
        Args:
            temp: Temperature for contrastive loss
            lambda_decor: Weight for decorative loss
            lambda_contrast: Weight for contrastive loss
        """
        super().__init__()
        self.temp = temp
        self.lambda_decor = lambda_decor
        self.lambda_contrast = lambda_contrast

    def forward(
        self, feats: torch.Tensor, proto: torch.Tensor, assignments: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combines structuring loss

        Args:
            feats: Patch features [N, C]
            proto: Current prototypes [K, C]
            assignments: Cluster assignments [N]

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Contrastive loss (pull assigned prototypes, push others)
        contrast_loss = self._contrastive_loss(feats, proto, assignments)

        # Decorrelation loss
        decor_loss = self._decorrelation_loss(proto)

        # Combined loss
        total_loss = (
            self.lambda_contrast * contrast_loss + self.lambda_decor * decor_loss
        )

        return total_loss, {
            "contrast_loss": contrast_loss.item(),
            "decor_loss": decor_loss.item(),
            "total_loss": total_loss.item(),
        }

    def _contrastive_loss(self, feats, proto, assignments):
        """
        Pull assigned prototypes, push other prototypes
        """
        # Normalize features and prototypes
        feat_norm = F.normalize(feats, p=2, dim=1)
        proto_norm = F.normalize(proto, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(feat_norm, proto_norm.T) / self.temp

        # Create target (1 for assigned prototypes, 0 for others)
        targets = torch.zeros_like(sim_matrix)
        targets[torch.arange(len(assignments)), assignments] = 1

        # for cosine similarities
        # loss = -torch.sum(targets * F.log_softmax(sim_matrix, dim=1)) / len(feats)
        # return loss

        return F.binary_cross_entropy_with_logits(sim_matrix, targets)

    def _decorrelation_loss(self, proto):
        """
        Penalize cosine similarity between prototypes
        """
        # Normalize prototypes
        proto_norm = F.normalize(proto, p=2, dim=1)

        # Compute correlation matrix (cosine similarity)
        corr_matrix = torch.matmul(proto_norm, proto_norm.T)

        # Remove diagonal (self-similarity)
        corr_matrix = corr_matrix - torch.diag(torch.diag(corr_matrix))

        # squared Frobenius norm (sum of squared elements)
        return torch.norm(corr_matrix, p="fro") ** 2


class StructuredPrototypeMemoryBank(PrototypeMemoryBank):
    def __init__(
        self, initial_proto: torch.Tensor, alpha: float = 0.9, loss_weights: dict = None
    ):
        """
        Extended memory bank with embedding structuring

        Args:
            loss_weights: Dict with keys "decor" and "contrast"
        """
        super().__init__(initial_proto, alpha)
        loss_weights = loss_weights or {}

        self.alpha = alpha
        self.shadow_proto = self.prototypes.data.clone()
        self.update_counts = torch.zeros(len(initial_proto))
        self.mature_thres = 10
        self.loss_fn = LossFunction(
            lambda_decor=loss_weights.get("decor", 1.0),
            lambda_contrast=loss_weights.get("contrast", 1.0),
        )

    def update(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(feats) == 0:
            return self.prototypes, torch.tensor([], device=feats.device)

        device = feats.device
        feats = feats.to(device)
        self.prototypes.data = self.prototypes.data.to(device)
        self.shadow_proto = self.shadow_proto.to(device)
        self.update_counts = self.update_counts.to(device)

        dist = torch.cdist(feats, self.shadow_proto, p=2)
        assignments = torch.argmin(dist, dim=1)

        for k in range(len(self.prototypes)):
            mask = assignments == k
            if mask.any():
                assigned_feats = feats[mask]
                mean_feat = assigned_feats.mean(dim=0)
                self.update_counts[k] += len(assigned_feats)

                if self.update_counts[k] < self.mature_thres:
                    warmup_alpha = self.alpha * (
                        self.update_counts[k] / self.mature_thres
                    )
                    self.prototypes.data[k] = (
                        warmup_alpha * self.prototypes.data[k]
                        + (1 - warmup_alpha) * mean_feat
                    )
                else:
                    self.prototypes.data[k] = (
                        self.alpha * self.prototypes.data[k]
                        + (1 - self.alpha) * mean_feat
                    )

        if torch.rand(1).item() < 0.1:
            self.shadow_proto = self.prototypes.data.clone().detach()

        return self.prototypes, assignments

    def structured_update(
        self, feats: torch.Tensor, optim: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, dict]:
        """
        Update prototypes with structure-preserving loss

        Args:
            optim: Optimizer for prototype refinement

        Returns:
            loss_dict: Dictionary of loss values
        """
        updated_proto, assignments = self.update(feats)

        # Refine with structure losses
        optim.zero_grad()
        total_loss, loss_dict = self.loss_fn(feats, self.prototypes, assignments)
        total_loss.backward()
        optim.step()
        return updated_proto, loss_dict

    def get_mature_proto(self) -> torch.Tensor:
        mature_mask = self.update_counts >= self.mature_thres
        return self.prototypes.data[mature_mask]

    def get_proto_weights(self) -> torch.Tensor:
        return self.update_counts / self.update_counts.sum()


def main() -> None:
    prototypes = torch.randn(10, 128, requires_grad=True)
    features = torch.randn(100, 128)

    memory_bank = StructuredPrototypeMemoryBank(
        prototypes, alpha=0.9, loss_weights={"decor": 1.0, "contrast": 0.5}
    )

    optimizer = torch.optim.SGD([prototypes], lr=0.01)

    updated_protos, loss_dict = memory_bank.structured_update(features, optimizer)

    print("Updated prototypes shape:", updated_protos.shape)
    print("Losses:", loss_dict)


if __name__ == "__main__":
    main()
