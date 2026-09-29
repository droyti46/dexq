"""Frozen public hip quality graph, imported only by training/conversion."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SharedQualityGraph(nn.Module):
    """One ImageNet ResNet18; expose both task representations in one pass.

    original is the strictly verified graph from router.bootstrap. There are
    no learned DXA neural parameters or task heads in this graph.
    """

    def __init__(self, original):
        super().__init__()
        self.backbone = original.backbone
        self.register_buffer("mean", original.mean.detach().clone())
        self.register_buffer("std", original.std.detach().clone())

    def forward(self, gray_u8):
        gray = gray_u8.to(torch.float32).unsqueeze(1) / 255.
        value = (gray.repeat(1, 3, 1, 1) - self.mean) / self.std
        for index, layer in enumerate(self.backbone):
            value = layer(value)
            if index == 6:
                stage3 = F.adaptive_avg_pool2d(value, (5, 5)).flatten(1)
        global_features = F.adaptive_avg_pool2d(value, (1, 1)).flatten(1)
        # Match the pinned public extractor's channel-major 2x2 grid order.
        # Slice pooling also preserves its treatment of odd feature-map bins.
        height, width = value.shape[-2:]
        bins = []
        for y in range(2):
            y0, y1 = y * height // 2, ((y + 1) * height + 1) // 2
            for x in range(2):
                x0, x1 = x * width // 2, ((x + 1) * width + 1) // 2
                bins.append(value[:, :, y0:y1, x0:x1].mean(dim=(-2, -1)))
        spatial_features = torch.stack(bins, dim=-1).flatten(1)
        return global_features, spatial_features, stage3
