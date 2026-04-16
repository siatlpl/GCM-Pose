import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from gcmpose.model.dino_layers import (MemEffSelfAttention, MemEffCrossAttention,
                          MemEffEncoderLayer, MemEffDecoderLayer)
from gcmpose.model.pointnet_utils import PointNetMultiScaleEncoder


class DeformableOffsetNet(nn.Module):
    """Predicts sampling offsets for deformable attention on 2D feature maps."""
    def __init__(self, feat_dim, num_points=4, num_heads=8):
        super().__init__()
        self.num_points = num_points
        self.num_heads = num_heads
        self.offset_proj = nn.Linear(feat_dim, num_heads * num_points * 2)
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)

    def forward(self, query_feat):
        """
        query_feat: (B, N, C)
        Returns offsets: (B, N, num_heads, num_points, 2)
        """
        B, N, C = query_feat.shape
        offsets = self.offset_proj(query_feat)
        offsets = offsets.view(B, N, self.num_heads, self.num_points, 2)
        return offsets.tanh() * 0.5


class DeformableCrossAttention2D(nn.Module):
    """
    Deformable cross-attention: query tokens attend to offset-sampled
    locations on a 2D feature map.
    """
    def __init__(self, feat_dim, num_heads=8, num_points=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = feat_dim // num_heads

        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.offset_net = DeformableOffsetNet(feat_dim, num_points, num_heads)
        self.attn_weights = nn.Linear(feat_dim, num_heads * num_points)
        nn.init.zeros_(self.attn_weights.bias)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, query, feat_map, ref_pos=None):
        """
        query:    (B, N, C)
        feat_map: (B, C, H, W)
        ref_pos:  (B, N, 2) normalized positions in [-1,1], or None

        Returns: (B, N, C)
        """
        B, N, C = query.shape
        _, _, H, W = feat_map.shape

        offsets = self.offset_net(query)  # (B, N, num_heads, num_points, 2)

        if ref_pos is None:
            ref_pos = torch.zeros(B, N, 2, device=query.device)

        ref_pos_exp = ref_pos.unsqueeze(2).unsqueeze(3)  # (B, N, 1, 1, 2)
        sample_locs = ref_pos_exp + offsets              # (B, N, num_heads, num_points, 2)

        # Sample from feature map per head
        feat_per_head = feat_map.view(B, self.num_heads, self.head_dim, H, W)
        sampled_values = []
        for h in range(self.num_heads):
            locs_h = sample_locs[:, :, h, :, :]  # (B, N, num_points, 2)
            locs_h = locs_h.reshape(B, N * self.num_points, 1, 2)
            feat_h = feat_per_head[:, h]  # (B, head_dim, H, W)
            sampled = F.grid_sample(feat_h, locs_h, mode='bilinear',
                                    padding_mode='zeros', align_corners=True)
            # (B, head_dim, N*num_points, 1) -> (B, N, num_points, head_dim)
            sampled = sampled.squeeze(-1).permute(0, 2, 1).view(B, N, self.num_points, self.head_dim)
            sampled_values.append(sampled)

        # (B, N, num_heads, num_points, head_dim)
        sampled_values = torch.stack(sampled_values, dim=2)

        # Attention weights over sampling points
        attn_w = self.attn_weights(query)  # (B, N, num_heads*num_points)
        attn_w = attn_w.view(B, N, self.num_heads, self.num_points)
        attn_w = attn_w.softmax(dim=-1).unsqueeze(-1)  # (B, N, num_heads, num_points, 1)

        out = (attn_w * sampled_values).sum(dim=3)  # (B, N, num_heads, head_dim)
        out = out.reshape(B, N, C)
        out = self.out_proj(out)
        return self.norm(query + out)


class ScaleFusionBlock(nn.Module):
    """
    Per-scale CMMDA fusion block:
    1. Self-attention on 2D tokens (intra-modal context)
    2. Self-attention on 3D tokens (intra-modal context)
    3. Deformable cross-attention: 3D tokens query 2D feature map
    4. Standard cross-attention: 2D tokens query 3D tokens
    """
    def __init__(self, feat_dim, num_heads=8, num_layers=2, num_deform_points=4):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)

        self.self_attn_2d = nn.ModuleList([
            MemEffEncoderLayer(
                attn_class=MemEffSelfAttention, rope=None,
                dim=feat_dim, num_heads=num_heads, mlp_ratio=4, norm_layer=norm,
            ) for _ in range(num_layers)
        ])
        self.self_attn_3d = nn.ModuleList([
            MemEffEncoderLayer(
                attn_class=MemEffSelfAttention, rope=None,
                dim=feat_dim, num_heads=num_heads, mlp_ratio=4, norm_layer=norm,
            ) for _ in range(num_layers)
        ])
        self.deform_cross_attn = DeformableCrossAttention2D(
            feat_dim=feat_dim, num_heads=num_heads, num_points=num_deform_points
        )
        self.cross_attn_2d_to_3d = MemEffDecoderLayer(
            attn_class=MemEffCrossAttention, rope=None,
            dim=feat_dim, num_heads=num_heads, mlp_ratio=4, norm_layer=norm,
        )

    def forward(self, feat_2d_tokens, feat_3d_tokens, feat_2d_map, ref_pos_2d=None):
        """
        feat_2d_tokens: (B, N2, C)
        feat_3d_tokens: (B, N3, C)
        feat_2d_map:    (B, C, H, W)
        ref_pos_2d:     (B, N3, 2) normalized 2D positions for 3D queries

        Returns: feat_2d_tokens (B,N2,C), feat_3d_tokens (B,N3,C)
        """
        for layer in self.self_attn_2d:
            feat_2d_tokens = layer(x=feat_2d_tokens)
        for layer in self.self_attn_3d:
            feat_3d_tokens = layer(x=feat_3d_tokens)

        feat_3d_tokens = self.deform_cross_attn(feat_3d_tokens, feat_2d_map, ref_pos_2d)
        feat_2d_tokens = self.cross_attn_2d_to_3d(x=feat_2d_tokens, y=feat_3d_tokens)

        return feat_2d_tokens, feat_3d_tokens


class CMMDA(nn.Module):
    """
    Cross-Modal Multi-scale Deformable Attention network.

    Takes DINOv2 2D features and a 3D point cloud, produces matched 2D and 3D
    feature descriptors for pose estimation via PnP+RANSAC.

    Architecture:
    - 2D branch: DINOv2 features at 3 scales (32x32, 16x16, 8x8)
    - 3D branch: PointNet++ at 3 scales (512, 128, 32 points)
    - Per-scale ScaleFusionBlock (self-attn + deformable cross-attn)
    - 2D reconstruction head for photometric loss
    - 3D reconstruction head for chamfer loss
    """
    POINTNET_CHANNELS = [128, 256, 512]  # per scale

    def __init__(self, dino_feat_dim=768, feat_dim=256, num_heads=8,
                 num_layers=2, num_scales=3, num_deform_points=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_scales = num_scales

        self.pointnet = PointNetMultiScaleEncoder(in_channel=3)

        # 2D projection per scale
        self.proj_2d = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dino_feat_dim, feat_dim, 1),
                nn.GroupNorm(1, feat_dim),
                nn.GELU(),
            ) for _ in range(num_scales)
        ])

        # 3D projection per scale
        self.proj_3d = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.POINTNET_CHANNELS[i], feat_dim),
                nn.LayerNorm(feat_dim),
                nn.GELU(),
            ) for i in range(num_scales)
        ])

        # Per-scale fusion blocks
        self.fusion_blocks = nn.ModuleList([
            ScaleFusionBlock(feat_dim, num_heads, num_layers, num_deform_points)
            for _ in range(num_scales)
        ])

        self.norm_2d = nn.LayerNorm(feat_dim)
        self.norm_3d = nn.LayerNorm(feat_dim)

        # 2D reconstruction head: (B, feat_dim, H, W) -> (B, 3, H, W)
        self.recon_head_2d = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, 1, 1),
            nn.GroupNorm(1, feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim // 2, 3, 1, 1),
            nn.GroupNorm(1, feat_dim // 2),
            nn.GELU(),
            nn.Conv2d(feat_dim // 2, 3, 1),
            nn.Sigmoid(),
        )

        # 3D reconstruction head: (B, N3, feat_dim) -> (B, N3, 3)
        self.recon_head_3d = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 3),
        )

    def forward(self, dino_feat, point_cloud, return_reconstructions=False):
        """
        dino_feat:   (B, 768, 32, 32) from model_arch.extract_DINOv2_feature()
        point_cloud: (B, N, 3)

        Returns dict:
            'feat_2d':  (B, 1024, feat_dim) finest-scale 2D descriptors
            'feat_3d':  (B, 512, feat_dim)  finest-scale 3D descriptors
            'feat_2d_scales': list of (B, N2_s, feat_dim) per scale
            'feat_3d_scales': list of (B, N3_s, feat_dim) per scale
            'recon_2d': (B, 3, 32, 32) if return_reconstructions
            'recon_3d': (B, 512, 3)    if return_reconstructions
        """
        B = dino_feat.shape[0]

        # 3D multi-scale features
        pn_scales = self.pointnet(point_cloud)  # [(xyz1,f1),(xyz2,f2),(xyz3,f3)]

        # 2D multi-scale: 32x32, 16x16, 8x8
        dino_scales = []
        feat_map = dino_feat
        for s in range(self.num_scales):
            if s > 0:
                feat_map = F.avg_pool2d(feat_map, kernel_size=2, stride=2)
            dino_scales.append(feat_map)

        # Process coarse-to-fine (scale 2 -> 1 -> 0)
        feat_2d_scales = []
        feat_3d_scales = []
        prev_feat_3d = None

        for s in range(self.num_scales - 1, -1, -1):
            # 2D: project and flatten
            feat_2d_map = self.proj_2d[s](dino_scales[s])  # (B, feat_dim, H_s, W_s)
            H_s, W_s = feat_2d_map.shape[2], feat_2d_map.shape[3]
            feat_2d_tokens = feat_2d_map.flatten(2).transpose(1, 2)  # (B, H_s*W_s, feat_dim)

            # 3D: project
            xyz_s, raw_feat_s = pn_scales[s]
            feat_3d_tokens = self.proj_3d[s](raw_feat_s)  # (B, N_s, feat_dim)

            # Coarse-to-fine residual for 3D
            if prev_feat_3d is not None:
                N_curr = feat_3d_tokens.shape[1]
                N_prev = prev_feat_3d.shape[1]
                if N_curr != N_prev:
                    # Upsample coarse features to current scale
                    upsampled = F.interpolate(
                        prev_feat_3d.transpose(1, 2),  # (B, feat_dim, N_prev)
                        size=N_curr, mode='nearest'
                    ).transpose(1, 2)  # (B, N_curr, feat_dim)
                    feat_3d_tokens = feat_3d_tokens + upsampled
                else:
                    feat_3d_tokens = feat_3d_tokens + prev_feat_3d

            # Compute reference 2D positions for 3D tokens (normalized to [-1,1])
            xyz_norm = xyz_s[..., :2]  # use x,y as proxy 2D positions
            xyz_min = xyz_norm.min(dim=1, keepdim=True)[0]
            xyz_max = xyz_norm.max(dim=1, keepdim=True)[0]
            ref_pos_2d = 2.0 * (xyz_norm - xyz_min) / (xyz_max - xyz_min + 1e-8) - 1.0

            # Fusion block
            feat_2d_tokens, feat_3d_tokens = self.fusion_blocks[self.num_scales - 1 - s](
                feat_2d_tokens, feat_3d_tokens, feat_2d_map, ref_pos_2d
            )

            feat_2d_scales.append((feat_2d_tokens, feat_2d_map, H_s, W_s))
            feat_3d_scales.append(feat_3d_tokens)
            prev_feat_3d = feat_3d_tokens

        # Finest scale outputs (last processed = scale 0)
        feat_2d_final = self.norm_2d(feat_2d_scales[-1][0])  # (B, 32*32, feat_dim)
        feat_3d_final = self.norm_3d(feat_3d_scales[-1])     # (B, 512, feat_dim)

        out = {
            'feat_2d': feat_2d_final,
            'feat_3d': feat_3d_final,
            'feat_2d_scales': [self.norm_2d(x[0]) for x in feat_2d_scales],
            'feat_3d_scales': [self.norm_3d(x) for x in feat_3d_scales],
        }

        if return_reconstructions:
            # 2D reconstruction from finest-scale feature map
            feat_map_finest = feat_2d_scales[-1][1]  # (B, feat_dim, 32, 32)
            out['recon_2d'] = self.recon_head_2d(feat_map_finest)

            # 3D reconstruction from finest-scale 3D features
            out['recon_3d'] = self.recon_head_3d(feat_3d_final)

        return out
