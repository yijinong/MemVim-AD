from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModule(nn.Module):
    """
    Learnable memory module that stores prototypes and representing normal patterns
    """

    def __init__(self, num_proto: int, proto_dim: int, temp: float = 1.0):
        super().__init__()
        self.num_proto = num_proto
        self.proto_dim = proto_dim
        self.temp = temp

        # initialize prototypes with Xavier uniform initialization
        self.protos = nn.Parameter(torch.empty(num_proto, proto_dim))

        nn.init.xavier_uniform_(self.protos)

    def forward(
        self, feats: torch.Tensor
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
        ori_shape = feats.shape

        if len(feats.shape) == 4:
            B, C, H, W = feats.shape
            feats_flat = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:
            B, N, D = feats.shape
            feats_flat = feats
            C = D

        # Normalize features and prototypes for cosine similarity
        feats_norm = F.normalize(feats_flat, dim=-1)  # (B, H, C)
        protos_norm = F.normalize(self.protos, dim=-1)  # (K, C)

        sims = torch.matmul(feats_norm, protos_norm.t())
        sims = sims / self.temp

        # Compute attention weights
        attn_weights = F.softmax(sims, dim=-1)  # (B, H, K)

        # Retrieve from memory using attention weights
        retrieved_feats_flat = torch.matmul(attn_weights, self.protos)  # (B, H, C)

        # Compute entropy for anomaly detection
        entropy_scores = -torch.sum(
            attn_weights * torch.log(attn_weights + 1e-8), dim=-1
        )  # (B, H)

        # Reshape back to original input shape
        if len(ori_shape) == 4:
            retrieved_feats = retrieved_feats_flat.reshape(B, H, W, C).permute(
                0, 3, 1, 2
            )
            entropy_scores = entropy_scores.reshape(B, H, W)
        else:
            retrieved_feats = retrieved_feats_flat

        return retrieved_feats, attn_weights, entropy_scores

    def expand_mem(self, new_protos: int):
        """
        Expand memory for continual learning by adding new prototypes slots
        """
        old_proto = self.protos.data
        new_proto_data = torch.empty(new_protos, self.proto_dim)
        nn.init.xavier_uniform_(new_proto_data)

        # concatenate old and new prototypes
        expanded_protos = torch.cat(
            [old_proto, new_proto_data.to(old_proto.device)], dim=0
        )
        self.protos = nn.Parameter(expanded_protos)
        self.num_proto += new_protos
