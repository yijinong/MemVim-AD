from typing import Optional

import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from timm.models.layers import DropPath, to_2tuple

__all__ = ["vim_base_patch16_224"]


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        stride: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module = None,
        flatten: bool = True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride + 1,
            (img_size[1] - patch_size[1]) // stride + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}, {W}) does not match expected size {self.img_size}"
        )

        x = self.proj(x)
        if self.flatten:
            # BCHW -> BNC
            x = x.flatten(2).transpose(1, 2)

        x = self.norm(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        drop_path: float = 0.0,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm,
                (nn.LayerNorm, RMSNorm),
                "Only LayerNorm and RMSNorm are supported for fused_add_norm",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> torch.Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))

        """


class MambaBlock(nn.Module):
    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        d_inner = int(self.expand * d_model)

        # input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=d_inner,
            padding=d_conv - 1,
        )

        # State space parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        # Output projector
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.act = nn.SiLU()
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape

        # input projection
        # (batch, length, 2 * d_inner)
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=x_and_res.shape[-1] // 2, dim=-1)

        # Convolution
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = self.act(x)

        # Simplified state space computation
        dt = self.dt_proj(x)
        dt = nn.Softplus()(dt)

        # Simplified selective scan (not fully accurate to Mamba)
        # This is a placeholder - full Mamba would require custom CUDA kernels
        y = x * torch.sigmoid(dt)

        # Gating
        y = y * self.act(res)

        # Output projection
        output = self.out_proj(y)

        return output


class MambaFeatureExtractor(nn.Module):
    """
    Mamba-based feature extractor for images

    This implementation provides a Vision-Mamba hybrid that:
    1. Converts images to patches (similar to Vision Transformer)
    2. Applies Mamba blocks for sequence modeling
    3. Outputs global image features for anomaly detection

    Args:
        input_dim: Number of input channels (3 for RGB)
        d_model: Model dimension
        n_layers: Number of Mamba layers
        patch_size: Size of image patches
        img_size: Input image size (assumed square)
    """

    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 256,
        n_layers: int = 4,
        patch_size: int = 16,
        img_size: int = 224,
    ):
        super().__init__()

        # Validation
        assert img_size % patch_size == 0, (
            f"Image size {img_size} must be divisible by patch size {patch_size}"
        )
        assert d_model > 0, "d_model must be positive"
        assert n_layers > 0, "n_layers must be positive"

        self.d_model = d_model  # Store d_model as an attribute
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            input_dim, d_model, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))

        # Mamba layers
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got {x.shape}")

        _ = x.shape  # We don't actually need to unpack these

        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, d_model)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply Mamba layers
        for layer in self.layers:
            x = layer(x) + x  # Residual connection

        x = self.norm(x)

        # Global pooling
        x = x.transpose(1, 2)  # (B, d_model, n_patches)
        x = self.pool(x).squeeze(-1)  # (B, d_model)

        return x
