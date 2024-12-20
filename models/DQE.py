# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from typing import Tuple

from models.regression import UpsamplingLayer
from models.transformer import (
    SelfCrossAttentionBlock,
    PrototypeAttentionBlock,
    ImgToPrototypeAttentionBlock,
)


class DQE(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        num_prototype_attn_steps: int,
        num_image_attn_steps: int,
        zero_shot: bool = False
    ) -> None:
        """

        Arguments:
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.prototype_attention = nn.ModuleList()
        self.image_attention = nn.ModuleList()
        self.zero_shot = zero_shot

        if self.zero_shot:
            self.image_to_prototype_attn = ImgToPrototypeAttentionBlock(
                embedding_dim=transformer_dim,
                num_heads=8,
            )

        for _ in range(num_prototype_attn_steps):
            self.prototype_attention.append(
                PrototypeAttentionBlock(
                    embedding_dim=transformer_dim,
                    num_heads=8,
                )
            )

        for _ in range(num_image_attn_steps):
            self.image_attention.append(
                SelfCrossAttentionBlock(
                    embedding_dim=transformer_dim,
                    num_heads=8,
                )
            )

        self.upscale = nn.Sequential(
            UpsamplingLayer(transformer_dim, transformer_dim),
            UpsamplingLayer(transformer_dim, transformer_dim),
        )
        self.upscale_hq = UpsamplingLayer(transformer_dim + 32, transformer_dim)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        prototype_embeddings: torch.Tensor,
        hq_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
        b, c, h, w = image_embeddings.shape
        image_pe = torch.repeat_interleave(image_pe, image_embeddings.shape[0], dim=0)
        if image_pe.shape[1:] != image_embeddings.shape[1:]:
            upsample_pos_emb = nn.UpsamplingBilinear2d(scale_factor=1.5)
            image_pe = upsample_pos_emb(image_pe)
        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        src = image_embeddings

        if self.zero_shot:
            prototype_embeddings = self.image_to_prototype_attn(
                image_f=src, prototypes=prototype_embeddings
            )

        for layer in self.prototype_attention:
            src = layer(image_f=src, prototypes=prototype_embeddings)

        for layer in self.image_attention:
            src = layer(image_f=src, adapted_image_f=image_embeddings, pos_enc=image_pe)
        src = src.transpose(1, 2).view(b, c, h, w)
        src = self.upscale(src)
        src = torch.cat([src, hq_features], dim=1)
        src = self.upscale_hq(src)

        return src
