import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import ops as py3d_ops


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ SetAbstraction layer.
    FPS sampling -> ball query grouping -> shared MLP -> max pooling.
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        last_ch = in_channel
        for out_ch in mlp_channels:
            layers += [
                nn.Conv2d(last_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)
        self.out_channels = last_ch

    def forward(self, xyz, features=None):
        """
        xyz:      (B, N, 3)
        features: (B, N, C) or None

        Returns:
            new_xyz:      (B, npoint, 3)
            new_features: (B, npoint, out_channels)
        """
        B, N, _ = xyz.shape

        # FPS to select centroids
        _, fps_idx = py3d_ops.sample_farthest_points(xyz, K=self.npoint)  # (B, npoint)

        # Gather centroid coordinates
        fps_idx_exp = fps_idx.unsqueeze(-1).expand(-1, -1, 3)  # (B, npoint, 3)
        new_xyz = torch.gather(xyz, 1, fps_idx_exp)  # (B, npoint, 3)

        # Ball query: find neighbors within radius
        # Returns (B, npoint, nsample) indices
        ball_idx, _ = py3d_ops.ball_query(
            p1=new_xyz, p2=xyz,
            K=self.nsample, radius=self.radius, return_nn=False
        )  # (B, npoint, nsample)

        # Gather grouped xyz
        B, npoint, nsample = ball_idx.shape
        ball_idx_flat = ball_idx.view(B, -1)  # (B, npoint*nsample)
        ball_idx_exp = ball_idx_flat.unsqueeze(-1).expand(-1, -1, 3)
        grouped_xyz = torch.gather(xyz, 1, ball_idx_exp)  # (B, npoint*nsample, 3)
        grouped_xyz = grouped_xyz.view(B, npoint, nsample, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)  # normalize to centroid

        if features is not None:
            C = features.shape[-1]
            feat_idx_exp = ball_idx_flat.unsqueeze(-1).expand(-1, -1, C)
            grouped_feat = torch.gather(features, 1, feat_idx_exp)  # (B, npoint*nsample, C)
            grouped_feat = grouped_feat.view(B, npoint, nsample, C)
            grouped_input = torch.cat([grouped_xyz, grouped_feat], dim=-1)  # (B, npoint, nsample, 3+C)
        else:
            grouped_input = grouped_xyz  # (B, npoint, nsample, 3)

        # MLP: (B, C_in, npoint, nsample)
        grouped_input = grouped_input.permute(0, 3, 1, 2).contiguous()
        grouped_input = self.mlp(grouped_input)   # (B, C_out, npoint, nsample)
        new_features = grouped_input.max(dim=-1)[0]  # (B, C_out, npoint)
        new_features = new_features.transpose(1, 2).contiguous()  # (B, npoint, C_out)

        return new_xyz, new_features


class PointNetMultiScaleEncoder(nn.Module):
    """
    3-scale PointNet++ encoder.
    Returns features at 3 resolutions for multi-scale CMMDA.

    Scale 0 (fine):   512 points, 128 channels
    Scale 1 (medium): 128 points, 256 channels
    Scale 2 (coarse):  32 points, 512 channels
    """
    def __init__(self, in_channel=3):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.1, nsample=32,
            in_channel=in_channel,
            mlp_channels=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.2, nsample=64,
            in_channel=3 + 128,
            mlp_channels=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=32, radius=0.4, nsample=128,
            in_channel=3 + 256,
            mlp_channels=[256, 256, 512]
        )

    def forward(self, xyz):
        """
        xyz: (B, N, 3)

        Returns list of (xyz_i, feat_i) at 3 scales:
            [(B,512,3), (B,512,128)],
            [(B,128,3), (B,128,256)],
            [(B,32,3),  (B,32,512)]
        """
        xyz1, feat1 = self.sa1(xyz)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)
        return [(xyz1, feat1), (xyz2, feat2), (xyz3, feat3)]
