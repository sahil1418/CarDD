"""
Mask2Former — Dual-head decoder for damage and anatomy segmentation.

Architecture:
  - Shared FPN input from DINOv2Backbone
  - Two independent Mask2Former decoder heads
  - Head A: Damage segmentation (5 classes)
  - Head B: Anatomy segmentation (8 classes)
  - Each head uses 100 learnable queries, 6 transformer decoder layers,
    and masked cross-attention over multi-scale FPN features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pipeline.config import (
    NUM_QUERIES, DECODER_LAYERS, DECODER_DIM, DECODER_HEADS,
    DECODER_DROPOUT, NUM_DAMAGE_CLASSES, NUM_ANATOMY_CLASSES,
    FPN_CHANNELS, IGNORE_INDEX, IMG_SIZE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ──────────────────────────────────────────────────────────────────────────────

class PositionEmbeddingSine(nn.Module):
    """2-D sinusoidal position encoding for spatial feature maps."""

    def __init__(self, num_features: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_features = num_features
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, num_features*2, H, W)"""
        B, _, H, W = x.shape
        device = x.device

        y_embed = torch.arange(H, device=device).unsqueeze(1).expand(H, W).float()
        x_embed = torch.arange(W, device=device).unsqueeze(0).expand(H, W).float()

        y_embed = y_embed / H
        x_embed = x_embed / W

        dim_t = torch.arange(self.num_features, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_features)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1)          # (H, W, num_features*2)
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        return pos


class MaskedCrossAttention(nn.Module):
    """
    Cross-attention where queries attend to spatial features,
    with optional predicted-mask guidance.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, memory, memory_pos=None, attn_mask=None):
        """
        queries: (B, N_q, D)
        memory:  (B, L, D)   — flattened spatial features
        memory_pos: (B, L, D)
        attn_mask:  (B, heads, N_q, L) — predicted mask logits
        """
        B, N_q, D = queries.shape
        _, L, _ = memory.shape

        q = self.q_proj(queries)
        k = self.k_proj(memory + memory_pos if memory_pos is not None else memory)
        v = self.v_proj(memory)

        # Reshape for multi-head
        q = q.view(B, N_q, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N_q, D)
        return self.out_proj(out)


class Mask2FormerDecoderLayer(nn.Module):
    """Single transformer decoder layer with masked cross-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        # Masked cross-attention
        self.cross_attn = MaskedCrossAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, memory, memory_pos=None, attn_mask=None):
        # Cross-attention
        q2 = self.cross_attn(queries, memory, memory_pos, attn_mask)
        queries = self.norm1(queries + q2)

        # Self-attention
        q2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm2(queries + q2)

        # FFN
        queries = self.norm3(queries + self.ffn(queries))
        return queries


# ──────────────────────────────────────────────────────────────────────────────
# Single Mask2Former Head
# ──────────────────────────────────────────────────────────────────────────────

class Mask2FormerHead(nn.Module):
    """
    One Mask2Former decoder head with learnable queries,
    producing per-pixel class predictions through mask classification.
    """

    def __init__(self, num_classes: int, d_model: int = DECODER_DIM,
                 n_heads: int = DECODER_HEADS,
                 n_layers: int = DECODER_LAYERS,
                 n_queries: int = NUM_QUERIES,
                 dropout: float = DECODER_DROPOUT):
        super().__init__()
        self.num_classes = num_classes
        self.n_queries = n_queries
        self.d_model = d_model

        # Learnable queries
        self.query_embed = nn.Embedding(n_queries, d_model)
        self.query_feat  = nn.Embedding(n_queries, d_model)

        # Input projection from FPN
        self.input_proj = nn.Conv2d(FPN_CHANNELS, d_model, 1)

        # Positional encoding for spatial features
        self.pos_enc = PositionEmbeddingSine(d_model // 2)

        # Decoder layers
        self.layers = nn.ModuleList([
            Mask2FormerDecoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Prediction heads
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, fpn_features: list[torch.Tensor]):
        """
        Args:
            fpn_features: list of 4 tensors from the FPN adapter.
                          We use the highest-resolution one as primary.

        Returns:
            pred_logits: (B, N_q, num_classes+1) — classification logits.
            pred_masks:  (B, N_q, H, W) — mask logits at input resolution.
        """
        # Use finest FPN level for highest detail
        feat = fpn_features[0]                       # (B, C, h, w)
        B = feat.shape[0]

        # Project and flatten
        feat_proj = self.input_proj(feat)             # (B, D, h, w)
        pos = self.pos_enc(feat_proj)                 # (B, D, h, w)

        h, w = feat_proj.shape[2:]
        memory = feat_proj.flatten(2).permute(0, 2, 1)    # (B, h*w, D)
        memory_pos = pos.flatten(2).permute(0, 2, 1)      # (B, h*w, D)

        # Initialize queries
        queries = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)

        # Iterative decoding with intermediate mask predictions
        attn_mask = None
        for layer in self.layers:
            queries = layer(queries, memory, memory_pos, attn_mask)

            # Compute intermediate mask prediction for next layer's attention
            mask_embed = self.mask_head(queries)       # (B, N_q, D)
            # Dot product with spatial features to get mask logits
            intermediate_mask = torch.einsum(
                "bqd,bld->bql", mask_embed, memory
            )                                          # (B, N_q, h*w)

            # Create attention mask: where predicted mask is < 0.5, mask it out
            attn_mask = (intermediate_mask.sigmoid() < 0.5)
            attn_mask = attn_mask.unsqueeze(1).expand(
                -1, DECODER_HEADS, -1, -1
            ).float() * -1e9                              # (B, heads, N_q, h*w)

        # Final predictions
        pred_logits = self.class_head(queries)          # (B, N_q, C+1)

        mask_embed = self.mask_head(queries)
        pred_masks = torch.einsum(
            "bqd,bdhw->bqhw", mask_embed, feat_proj
        )                                              # (B, N_q, h, w)

        return pred_logits, pred_masks


# ──────────────────────────────────────────────────────────────────────────────
# Combined Dual-Head Model
# ──────────────────────────────────────────────────────────────────────────────

class DualMask2Former(nn.Module):
    """
    Combines two Mask2Former heads fed by a shared backbone.

    For training: forward returns per-pixel semantic segmentation logits
    that can be directly supervised with CrossEntropy + Dice.

    For inference: returns both instance-level (masks + classes) and
    per-pixel semantic maps.
    """

    def __init__(self):
        super().__init__()
        self.damage_head  = Mask2FormerHead(NUM_DAMAGE_CLASSES)
        self.anatomy_head = Mask2FormerHead(NUM_ANATOMY_CLASSES)

    def _to_semantic(self, pred_logits, pred_masks, num_classes, target_h, target_w):
        """
        Convert query-based predictions to per-pixel semantic map.

        pred_logits: (B, N_q, C+1)
        pred_masks:  (B, N_q, h, w)

        Returns: (B, num_classes, H, W) semantic logits.
        """
        B = pred_logits.shape[0]

        # Get per-query class probabilities (exclude no-object class)
        cls_probs = pred_logits[:, :, :-1].softmax(dim=-1)   # (B, N_q, C)

        # Upsample masks to target resolution
        masks = F.interpolate(
            pred_masks, size=(target_h, target_w),
            mode="bilinear", align_corners=False
        )                                                     # (B, N_q, H, W)

        # Weighted combination: each pixel gets class logits from all queries
        # semantic = sum_q cls_prob_q * mask_q
        semantic = torch.einsum("bqc,bqhw->bchw", cls_probs, masks.sigmoid())

        return semantic

    def forward(self, fpn_features, target_size=IMG_SIZE):
        """
        Args:
            fpn_features: list of 4 FPN tensors from backbone.
            target_size:  output resolution (H == W assumed).

        Returns:
            damage_logits:  (B, NUM_DAMAGE_CLASSES, H, W)
            anatomy_logits: (B, NUM_ANATOMY_CLASSES, H, W)
            raw_outputs: dict with query-level predictions for inference.
        """
        # Head A: Damage
        dmg_logits, dmg_masks = self.damage_head(fpn_features)

        # Head B: Anatomy
        anat_logits, anat_masks = self.anatomy_head(fpn_features)

        # Convert to semantic maps for training loss
        H = W = target_size
        damage_semantic  = self._to_semantic(dmg_logits, dmg_masks,
                                             NUM_DAMAGE_CLASSES, H, W)
        anatomy_semantic = self._to_semantic(anat_logits, anat_masks,
                                             NUM_ANATOMY_CLASSES, H, W)

        raw = {
            "damage_query_logits": dmg_logits,
            "damage_query_masks":  dmg_masks,
            "anatomy_query_logits": anat_logits,
            "anatomy_query_masks":  anat_masks,
        }

        return damage_semantic, anatomy_semantic, raw
