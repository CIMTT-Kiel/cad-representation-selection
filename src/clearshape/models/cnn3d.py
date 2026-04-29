import torch
import torch.nn as nn


def _conv3d_block(in_ch, out_ch, stride=2):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class VoxelCNN3D(nn.Module):
    """3D CNN backbone for 128^3 binary voxel grids.

    Spatial reduction: 128 → 64 → 32 → 16 → 8 → 4 → 1 (AdaptiveAvgPool)
    Channels:          1  → 32 → 64 → 128 → 256 → 512
    Output: feature vector [B, 512]
    """

    out_dim = 512

    def __init__(self, dropout: float = 0.1):
        super().__init__()

        self.features = nn.Sequential(
            _conv3d_block(1,   32,  stride=2),
            _conv3d_block(32,  64,  stride=2),
            _conv3d_block(64,  128, stride=2),
            _conv3d_block(128, 256, stride=2),
            _conv3d_block(256, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.dropout(x)


class VoxelClassifierCNN3D(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = VoxelCNN3D(dropout=dropout)
        self.head = nn.Linear(VoxelCNN3D.out_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.head.weight, nonlinearity='relu')
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        return self.head(self.backbone(x))


class VoxelRegressorCNN3D(nn.Module):
    """3D CNN for multi-target regression with per-target weighted loss."""

    def __init__(self,
                 n_targets: int = 4,
                 dropout: float = 0.1,
                 use_target_specific_heads: bool = True,
                 target_weights=None):
        super().__init__()
        self.n_targets = n_targets
        self.use_target_specific_heads = use_target_specific_heads

        self.backbone = VoxelCNN3D(dropout=dropout)
        feat_dim = VoxelCNN3D.out_dim

        if use_target_specific_heads:
            self.target_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feat_dim, feat_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(feat_dim // 4, 1),
                )
                for _ in range(n_targets)
            ])
        else:
            self.output_layer = nn.Linear(feat_dim, n_targets)

        if target_weights is None:
            # Use inverse of target std dev as weights (higher weight for targets with lower variance)
            w = torch.tensor([1 / 2.89, 1 / 1.06, 1 / 1.28, 1 / 1.28])
            self.target_weights = w / w.sum() * n_targets
        else:
            self.target_weights = torch.tensor(target_weights, dtype=torch.float32)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        if self.use_target_specific_heads:
            return torch.cat([h(features) for h in self.target_heads], dim=1)
        return self.output_layer(features)

    def compute_weighted_loss(self, predictions, targets):
        loss_fn = nn.MSELoss(reduction='none')
        per_target = loss_fn(predictions, targets)
        weights = self.target_weights.to(predictions.device)
        total_loss = (per_target * weights.unsqueeze(0)).mean()
        individual_losses = per_target.mean(dim=0)
        return total_loss, individual_losses
