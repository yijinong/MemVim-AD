from typing import Tuple

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.data.transforms_factory import create_transform
from transformers import AutoModel, AutoModelForImageClassification

MODEL_PATH = "/data2/yijin/MambaVision-L3-512-21K"


class MemoryAugmentedLayer(nn.Module):
    def __init__(self, dim: int, num_proto: int):
        super().__init__()
        self.dim = dim
        self.num_proto = num_proto
        self.memory = nn.Parameter(torch.randn(num_proto, dim))
        self.query_proj = nn.Linear(dim, dim)
        self.scale = dim**-0.5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query_proj(x)
        attn_logits = torch.matmul(q, self.memory.t()) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        recon = torch.matmul(attn_weights, self.memory)

        return recon, attn_weights


class MemoryAugmentedMamba(nn.Module):
    def __init__(self, base_mamba: nn.Module, mem_layers: list, num_proto: int):
        super().__init__()
        self.base = base_mamba
        self.mem_layers = mem_layers

        # insert memory modules
        self.mem_modules = nn.ModuleDict()
        for i in mem_layers:
            block_dim = base_mamba.model.levels[i].blocks[0].mlp.fc1.in_features
            # block_dim = base_mamba.config.hidden_size
            self.mem_modules[f"mem_{i}"] = MemoryAugmentedLayer(block_dim, num_proto)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        attns = {}
        for i, blk in enumerate(self.base.model.levels):
            x = blk(x)
            if str(i) in self.mem_modules:
                x, attn = self.mem_modules[str(i)](x)
                attns[f"layer_{i}"] = attn
        return x, attns


class MemoryBank(nn.Module):
    def __init__(self, num_prototypes=100, prototype_dim=1024):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, prototype_dim))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        """
        Args:
            z: Input features [B, h*w, D] (flattened spatial dims)
        Returns:
            z_hat: Reconstructed features [B, h*w, D]
            attn: Attention weights [B, h*w, K]
        """
        # Cosine similarity between z and prototypes
        sim = F.cosine_similarity(
            z.unsqueeze(2), self.prototypes.unsqueeze(0), dim=-1
        )  # [B, h*w, K]
        attn = self.softmax(sim)
        z_hat = torch.einsum("bik,kd->bid", attn, self.prototypes)  # [B, h*w, D]
        return z_hat, attn


class GatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * feature_dim, feature_dim), nn.Sigmoid())

    def forward(self, z, z_hat):
        """
        Args:
            z: Original features [B, h*w, D]
            z_hat: Memory-retrieved features [B, h*w, D]
        """
        gate = self.gate(torch.cat([z, z_hat], dim=-1))  # [B, h*w, D]
        return gate * z + (1 - gate) * z_hat


class Decoder(nn.Module):
    def __init__(self, in_dim=1024, out_channels=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 7 * 7 * 512),  # Example upsampling
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z_fused):
        return self.layers(z_fused)


class MemVim(nn.Module):
    def __init__(self, backbone, num_prototypes=100):
        super().__init__()
        self.backbone = backbone  # MambaVision model
        self.memory = MemoryBank(
            num_prototypes, prototype_dim=1024
        )  # Match level[2] dim
        self.fusion = GatedFusion(feature_dim=1024)
        self.decoder = Decoder(in_dim=1024)

    def forward(self, x):
        # Step 1: Extract features from MambaVision (e.g., level[2] output)
        x = self.backbone.patch_embed(x)
        for i, level in enumerate(self.backbone.levels):
            x = level(x)
            if i == 2:  # Extract features from level 2 (shape [B, 1024, h, w])
                z = x.flatten(2).transpose(1, 2)  # [B, h*w, 1024]

        # Step 2: Memory interaction
        z_hat, attn = self.memory(z)  # z_hat: [B, h*w, 1024]
        z_fused = self.fusion(z, z_hat)  # [B, h*w, 1024]

        # Step 3: Reconstruct input
        z_fused = z_fused.transpose(1, 2).view_as(x)  # [B, 1024, h, w]
        x_recon = self.decoder(z_fused)  # [B, 3, H, W]

        return x_recon, attn


def mem_vim_loss(
    x, x_recon, attn, lambda_rec=1.0, lambda_sparse=0.1, lambda_commit=0.1
):
    # Reconstruction loss (L1)
    loss_rec = F.l1_loss(x_recon, x)

    # Sparsity loss (minimize entropy of attention)
    entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1)  # [B, h*w]
    loss_sparse = entropy.mean()

    # Commitment loss (pull prototypes closer to encoder outputs)
    loss_commit = F.mse_loss(z.detach(), z_hat)  # Stop gradient for z

    total_loss = (
        lambda_rec * loss_rec
        + lambda_sparse * loss_sparse
        + lambda_commit * loss_commit
    )
    return total_loss


def expand_memory(memory_module, new_prototypes=10):
    old_prototypes = memory_module.prototypes.data
    new_prototypes = torch.randn(new_prototypes, old_prototypes.shape[1])
    memory_module.prototypes = nn.Parameter(torch.cat([old_prototypes, new_prototypes]))


def main() -> None:
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype="auto", local_files_only=True
    )
    model.cuda().eval()

    # get the information the layers in encoder
    # for i in range(len(model.model.levels)):
    #     for j in range(len(model.model.levels[i].blocks)):
    #         print(
    #             f"Layer {i}, Block {j} in Mamba encoder: {model.model.levels[i].blocks[j]}"
    #         )

    model.memory = expand_memory(model.memory, new_prototypes=10)

    x = torch.randn(1, 3, 224, 224)
    x_recon, attn = model(x)
    print("Recon shape:", x_recon.shape)  # Should match input
    print("Attn shape:", attn.shape)  # [1, h*w, num_prototypes]
    # Store exemplars from old classes

    exemplars = []  # List of (x_old, y_old) tuples

    def train_step(x_new, optimizer):
        # Mixed batch with exemplars
        x_exemplar = torch.stack([ex[0] for ex in exemplars])  # Random subset
        x_mixed = torch.cat([x_new, x_exemplar])

        # Forward pass
        x_recon, attn = model(x_mixed)
        loss = mem_vim_loss(x_mixed, x_recon, attn)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Freeze old prototypes (gradient mask)
        model.memory.prototypes.grad[:num_old_prototypes] = 0

    def anomaly_score(x):
        x_recon, attn = model(x)

        # Reconstruction-based anomaly map
        a_rec = torch.abs(x - x_recon).mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Memory-based anomaly (entropy of attention)
        entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1)  # [B, h*w]
        a_mem = entropy.view(x.shape[0], 1, *x.shape[2:])  # Reshape to spatial dims

        # Combined score
        score = 0.7 * a_rec + 0.3 * a_mem  # Weighted sum
        return score

    # print("Model configuration:", model.config.name)
    # aug_model = MemoryAugmentedMamba(base_mamba=model, mem_layers=[2, 3], num_proto=64)

    # image = Image.open(
    #     "/home/yijin/projects/pro-ad/data/mvtec-ad/carpet/train/good/001.png"
    # )
    # input_resolution = (3, 512, 512)

    # transform = create_transform(
    #     input_size=input_resolution,
    #     is_training=False,
    #     mean=model.config.mean,
    #     std=model.config.std,
    #     crop_mode=model.config.crop_mode,
    #     crop_pct=model.config.crop_pct,
    # )

    # inputs = transform(image).unsqueeze(0).cuda()

    # output, attn_maps = aug_model(inputs)
    # classification
    # outputs = model(inputs)
    # logits = outputs["logits"]
    # predicted_class_idx = logits.argmax(-1).item()
    # print("Predicted class:", model.config.id2label[predicted_class_idx])

    # get the layers in mamba model

    # Feature extraction
    # out_avg_pool, features = model(inputs)
    # print(
    #     "Size of the averaged pool features:", out_avg_pool.size()
    # )  # torch.Size([1, 640])
    # print("Number of stages in extracted features:", len(features))  # 4 stages
    # print(
    #     "Size of extracted features in stage 1:", features[0].size()
    # )  # torch.Size([1, 80, 56, 56])
    # print(
    #     "Size of extracted features in stage 4:", features[3].size()
    # )  # torch.Size([1, 640, 7, 7])


if __name__ == "__main__":
    main()
