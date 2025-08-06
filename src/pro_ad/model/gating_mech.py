from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaledGatingMechanism(nn.Module):
    """
    Learned gating mechanism to fuse original and memory-retrived features at multiscale
    """

    def __init__(self, feats_dim: List[int]):
        super().__init__()
        self.gates = nn.ModuleList()

        for dim in feats_dim:
            gate_net = nn.Sequential(
                nn.Conv2d(dim * 2, dim // 4, 1),
                nn.ReLU(),
                nn.Conv2d(dim // 4, dim, 1),
                nn.Sigmoid(),
            )
            self.gates.append(gate_net)

    def forward(
        self, ori_feats: List[torch.Tensor], retrieved_feats: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Fuse original and memory-retrieved features using learned gates at each scale.

        Args:
            ori_feats: Original features at different scales
            retrieved_feats: Memory-retrieved features at the same scales

        Returns:
            gated_feats: Fused features at each scale
        """
        gated_feats = []
        for ori, ret, gate in zip(ori_feats, retrieved_feats, self.gates):
            # Concatenate along channel dimension
            combined = torch.cat([ori, ret], dim=1)
            gate_weights = gate(combined)

            # Apply gating
            gated_feat = ori * gate_weights + ret * (1 - gate_weights)
            gated_feats.append(gated_feat)

        return gated_feats
