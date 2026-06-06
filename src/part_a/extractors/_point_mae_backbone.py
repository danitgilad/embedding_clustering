"""Self-contained, CPU-friendly Point-MAE encoder for 3D shape feature extraction.

WHY THIS FILE EXISTS
The upstream Point-MAE repo couples its point grouping to CUDA ops (``knn_cuda`` and
pointnet2 furthest-point-sampling) and its module imports pull in CUDA-only extensions
(``extensions.chamfer_dist``) at import time, so it cannot run CPU-only as published. The
transformer encoder itself is plain PyTorch, however. So we reimplement *only* the encoder
forward path here with a pure-torch FPS + KNN grouping, and load the official pretrained
encoder weights (the ``module.MAE_encoder.*`` tensors from Point-MAE's ``pretrain.pth``)
into it. This keeps Part A's 3D feature CPU-only and removes any CUDA-extension build from
setup, which makes the project far more reproducible.

The submodule names here (``encoder.first_conv``/``second_conv``, ``pos_embed``,
``blocks.blocks.N``, ``norm``) intentionally mirror the upstream ``MaskTransformer`` so the
pretrained weights load by name without renaming. The architecture/constants match
Point-MAE's ShapeNet pretrain config (trans_dim=384, depth=12, num_heads=6, group_size=32,
num_group=64).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# ----------------------------- grouping (pure torch) -----------------------------


def farthest_point_sample(xyz: torch.Tensor, n_sample: int) -> torch.Tensor:
    """Iterative farthest-point sampling. ``xyz`` (B, N, 3) -> centroid indices (B, n_sample).

    Deterministic: starts from point index 0, so the resulting embedding is reproducible.
    """
    b, n, _ = xyz.shape
    centroids = torch.zeros(b, n_sample, dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), 1e10, device=xyz.device)
    farthest = torch.zeros(b, dtype=torch.long, device=xyz.device)
    batch = torch.arange(b, device=xyz.device)
    for i in range(n_sample):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)        # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)              # (B, N)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(-1)[1]
    return centroids


def knn_group(xyz: torch.Tensor, center: torch.Tensor, k: int) -> torch.Tensor:
    """k nearest neighbours of each center among ``xyz``. Returns indices (B, G, k)."""
    dist = torch.cdist(center, xyz)                          # (B, G, N)
    return dist.topk(k, dim=-1, largest=False)[1]


class Group(nn.Module):
    """FPS centers + KNN neighbourhoods, each neighbourhood recentred (pure torch)."""

    def __init__(self, num_group: int, group_size: int) -> None:
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``xyz`` (B, N, 3) -> (neighborhood (B, G, M, 3), center (B, G, 3))."""
        b, n, _ = xyz.shape
        center_idx = farthest_point_sample(xyz, self.num_group)            # (B, G)
        center = torch.gather(xyz, 1, center_idx.unsqueeze(-1).expand(-1, -1, 3))
        idx = knn_group(xyz, center, self.group_size)                      # (B, G, M)
        idx_base = torch.arange(b, device=xyz.device).view(-1, 1, 1) * n
        flat = (idx + idx_base).view(-1)
        neighborhood = xyz.reshape(b * n, 3)[flat].view(b, self.num_group, self.group_size, 3)
        neighborhood = neighborhood - center.unsqueeze(2)                  # recenter
        return neighborhood, center


# --------------------- encoder modules (names mirror upstream) ---------------------


class Encoder(nn.Module):
    """PointNet-style per-group encoder: (B, G, M, 3) -> (B, G, C)."""

    def __init__(self, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        bs, g, n, _ = point_groups.shape
        pg = point_groups.reshape(bs * g, n, 3)
        feat = self.first_conv(pg.transpose(2, 1))                         # (BG, 256, n)
        feat_global = torch.max(feat, dim=2, keepdim=True)[0]
        feat = torch.cat([feat_global.expand(-1, -1, n), feat], dim=1)     # (BG, 512, n)
        feat = self.second_conv(feat)                                      # (BG, C, n)
        feat_global = torch.max(feat, dim=2, keepdim=False)[0]             # (BG, C)
        return feat_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None,
                 out_features: int | None = None, act_layer=nn.GELU, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, drop: float = 0.0, attn_drop: float = 0.0,
                 norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = nn.Identity()          # stochastic depth is a no-op at inference
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0, qkv_bias: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos)
        return x


class PointMAEEncoder(nn.Module):
    """Pure-torch Point-MAE encoder producing one global shape embedding per cloud.

    Mirrors upstream ``MaskTransformer`` submodule names so the pretrained
    ``module.MAE_encoder.*`` weights load directly. Runs with all groups visible (no
    masking) and pools the per-group tokens (max ++ mean) into a ``2 * trans_dim`` vector.
    """

    def __init__(self, trans_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 encoder_dims: int = 384, group_size: int = 32, num_group: int = 64) -> None:
        super().__init__()
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.encoder = Encoder(encoder_channel=encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, trans_dim),
        )
        self.blocks = TransformerEncoder(embed_dim=trans_dim, depth=depth, num_heads=num_heads)
        self.norm = nn.LayerNorm(trans_dim)

    @torch.no_grad()
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """``pts`` (B, N, 3) -> global embedding (B, 2 * trans_dim)."""
        neighborhood, center = self.group_divider(pts)
        tokens = self.encoder(neighborhood)                # (B, G, C)
        pos = self.pos_embed(center)                       # (B, G, C)
        x = self.norm(self.blocks(tokens, pos))            # (B, G, C)
        return torch.cat([x.max(dim=1)[0], x.mean(dim=1)], dim=-1)


def load_point_mae_encoder(checkpoint: str, trans_dim: int = 384, depth: int = 12,
                           num_heads: int = 6, encoder_dims: int = 384,
                           group_size: int = 32, num_group: int = 64) -> PointMAEEncoder:
    """Build the encoder and load upstream pretrained weights (``module.MAE_encoder.*`` keys).

    Raises ValueError if no encoder weights were found in the checkpoint (wrong file), so a
    silently-untrained encoder can never reach the clustering stage.
    """
    model = PointMAEEncoder(trans_dim=trans_dim, depth=depth, num_heads=num_heads,
                            encoder_dims=encoder_dims, group_size=group_size, num_group=num_group)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("base_model", ckpt) if isinstance(ckpt, dict) else ckpt
    prefix = "module.MAE_encoder."
    enc_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not enc_state:
        raise ValueError(
            f"no 'module.MAE_encoder.*' weights found in {checkpoint!r}; wrong checkpoint?"
        )
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    # group_divider holds no parameters, so a clean load reports no missing/unexpected keys.
    if missing or unexpected:
        raise ValueError(f"Point-MAE weight load mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval()
