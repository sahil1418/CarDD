"""
Severity Classification Model — Decoupled EfficientNet-B0 classifier.

Trained separately on the severity dataset (minor / moderate / severe).
Used at inference by the Severity Engine for fusion with geometric features.
"""
import torch
import torch.nn as nn
from torchvision import models

from pipeline.config import NUM_SEVERITY


class SeverityClassifier(nn.Module):
    """EfficientNet-B0 fine-tuned for 3-class severity classification."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        # Replace final classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, NUM_SEVERITY),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) normalised image tensor.

        Returns:
            (B, 3) logits for minor/moderate/severe.
        """
        return self.backbone(x)

    def predict(self, x: torch.Tensor):
        """Returns predicted class index and confidence."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            conf, cls_idx = probs.max(dim=-1)
        return cls_idx, conf
