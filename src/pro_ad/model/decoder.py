from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder to reconstruct images from fused features
    """

    def __init__(
        self, feat_dims: List[int], out_channels: int = 3, target_size: int = 512
    ):
        super().__init__()
        self.target_size = target_size
        self.out_channels = out_channels

        # Decoder blocks for each scale
        self.decoders = nn.ModuleList()

        for i, dim in enumerate(feat_dims):
            # calculate upsampling factor based on stage
            if i == 0:
                # Stage 1: 128x128 -> 512x512 (4x)
                upsample_factor = 4
            elif i == 1:
                # Stage 2: 64x64 -> 512x512 (8x)
                upsample_factor = 8
            elif i == 2:
                # Stage 3: 32x32 -> 512x512 (16x)
                upsample_factor = 16
            else:
                # Stage 4: 16x16 -> 512x512 (32x)
                upsample_factor = 32

            decoder = nn.Sequential(
                nn.Conv2d(dim, dim // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim // 2, dim // 4, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(
                    scale_factor=upsample_factor, mode="bilinear", align_corners=False
                ),
                nn.Conv2d(dim // 4, out_channels, 3, padding=1),
            )

            self.decoders.append(decoder)

        # Final fusion layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(feat_dims), out_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, fused_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode multi-scale features into a single image
        Args:
            fused_feats: List of features from different scales

        Returns:
            Reconstructed image of shape (B, C, target_size, target_size)
        """
        decoded_feats = []
        for feat, decoder in zip(fused_feats, self.decoders):
            decoded = decoder(feat)
            # Ensure all decoded features have the same spatial size
            if decoded.shape[-1] != self.target_size:
                decoded = F.interpolate(
                    decoded,
                    size=(self.target_size, self.target_size),
                    mode="bilinear",
                    align_corners=False,
                )
            decoded_feats.append(decoded)

        # Concatenate all decoded features
        concat_decoded = torch.cat(decoded_feats, dim=1)
        reconstructed = self.final_conv(concat_decoded)

        return reconstructed
