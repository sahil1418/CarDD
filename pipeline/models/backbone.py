"""
DINOv2 Backbone + Feature Pyramid Network Adapter.

- Loads DINOv2 ViT-B/14 (frozen by default).
- Extracts multi-scale features from intermediate transformer layers.
- Projects them through a lightweight FPN to produce 4-scale feature maps
  that the Mask2Former decoders consume.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.config import (
    DINO_MODEL, DINO_EMBED_DIM, DINO_PATCH,
    FPN_CHANNELS, FREEZE_BACKBONE, IMG_SIZE,
)


class FPNAdapter(nn.Module):
    """
    Takes raw ViT token features at 4 intermediate layers and produces
    a classic 4-scale FPN: {1/4, 1/8, 1/16, 1/32} of input resolution.
    """

    def __init__(self, in_dim: int = DINO_EMBED_DIM,
                 out_dim: int = FPN_CHANNELS):
        super().__init__()
        # Lateral 1×1 projections (one per extracted layer)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim, 1) for _ in range(4)
        ])
        # Smooth 3×3 convs after top-down merge
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_dim, out_dim, 3, padding=1) for _ in range(4)
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            features: list of 4 tensors, each (B, C, h, w) from the
                      reshaped ViT intermediate outputs, ordered from
                      shallowest to deepest.

        Returns:
            list of 4 FPN feature maps at decreasing resolutions.
        """
        assert len(features) == 4

        # Lateral connections
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, features)
        ]

        # Top-down pathway (deepest → shallowest)
        for i in range(2, -1, -1):
            h, w = laterals[i].shape[2:]
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=(h, w), mode="bilinear",
                align_corners=False
            )

        # Smooth
        outs = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]

        return outs


class DINOv2Backbone(nn.Module):
    """
    Frozen (or LoRA-tunable) DINOv2 ViT-B/14 backbone with FPN adapter.
    """

    def __init__(self, freeze: bool = FREEZE_BACKBONE):
        super().__init__()

        # Load DINOv2 from torch hub
        self.encoder = torch.hub.load(
            "facebookresearch/dinov2", DINO_MODEL, pretrained=True
        )
        self.encoder.eval()

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.fpn = FPNAdapter(DINO_EMBED_DIM, FPN_CHANNELS)

        # Number of intermediate blocks to tap (last 4 of 12)
        self.n_layers = 4

        # Spatial size after ViT patching for 512×512 input
        self._feat_h = IMG_SIZE // DINO_PATCH
        self._feat_w = IMG_SIZE // DINO_PATCH

    def _extract_intermediate(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward through DINOv2 and collect intermediate token outputs.

        Uses n=4 to get the last 4 transformer block outputs.
        Returns list of (B, embed_dim, h, w) tensors.
        """
        B = x.shape[0]

        # get_intermediate_layers(x, n=4) returns outputs from the
        # last 4 blocks (blocks 8, 9, 10, 11 for ViT-B with 12 blocks).
        # reshape=True converts (B, N_tokens, C) → (B, C, h, w)
        intermediate = self.encoder.get_intermediate_layers(
            x, n=4, reshape=True
        )

        # intermediate is a tuple of tensors (B, C, h, w)
        return list(intermediate)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, 3, 512, 512) normalised image tensor.

        Returns:
            List of 4 FPN feature maps at scales ~{1/4, 1/8, 1/16, 1/32}.
        """
        # Extract from frozen DINOv2
        with torch.no_grad() if FREEZE_BACKBONE else torch.enable_grad():
            raw_features = self._extract_intermediate(x)

        # Build multi-scale FPN
        # DINOv2 outputs are all same resolution (patch_size stride).
        # We create multi-scale by downscaling the deeper layers.
        multi_scale = []
        base_h, base_w = raw_features[0].shape[2:]

        for i, feat in enumerate(raw_features):
            # Scale 0: original, Scale 1: 1/2, Scale 2: 1/4, Scale 3: 1/8
            if i == 0:
                multi_scale.append(feat)
            else:
                scale = 2 ** i
                target_h = max(base_h // scale, 1)
                target_w = max(base_w // scale, 1)
                scaled = F.interpolate(
                    feat, size=(target_h, target_w),
                    mode="bilinear", align_corners=False
                )
                multi_scale.append(scaled)

        return self.fpn(multi_scale)
