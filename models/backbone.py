import os.path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d

from models.common import LayerNorm2d
from models.sam_ViT import ImageEncoderViT
from functools import partial


class Backbone(nn.Module):

    def __init__(
        self,
        requires_grad: bool,
        image_size: int,
        model_path: str = None,
    ):

        super(Backbone, self).__init__()

        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        self.num_channels = image_emb_size = 256
        vit_dim = 1280
        transformer_dim = 256
        vit_encoder = ImageEncoderViT(
            depth=32,
            embed_dim=1280,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=16,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[7, 15, 23, 31],
            window_size=14,
            out_chans=256,
        )
        self.backbone = vit_encoder

        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 8, kernel_size=2, stride=2
            ),
        )

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
        )
        if model_path is not None:

            checkpoint = torch.load(
                os.path.join(model_path, "sam_hq_vit_h.pth"), map_location="cpu"
            )
            state_dict = {
                k.replace("image_encoder.", ""): v
                for k, v in checkpoint.items()
                if "image_encoder" in k
            }
            self.backbone.load_state_dict(state_dict)

            state_dict = {
                k.replace("mask_decoder.compress_vit_feat.", ""): v
                for k, v in checkpoint.items()
                if "compress_vit_feat" in k
            }
            self.compress_vit_feat.load_state_dict(state_dict)

            state_dict = {
                k.replace("mask_decoder.embedding_encoder.", ""): v
                for k, v in checkpoint.items()
                if "embedding_encoder" in k
            }
            self.embedding_encoder.load_state_dict(state_dict)

            for n, param in self.named_parameters():
                param.requires_grad_(requires_grad)

    def forward(self, x):
        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed is not None:
            if self.backbone.pos_embed.shape[1:] != x.shape[1:]:
                upsample_pos_emb = nn.UpsamplingBilinear2d(scale_factor=1.5)
                pos_emb = upsample_pos_emb(
                    self.backbone.pos_embed.permute(0, 3, 1, 2)
                ).permute(0, 2, 3, 1)
            else:
                pos_emb = self.backbone.pos_embed
            x = x + pos_emb

        interm_embeddings = []
        for blk in self.backbone.blocks:
            x = blk(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)
        image_embeddings = self.backbone.neck(x.permute(0, 3, 1, 2))

        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(
            vit_features
        )

        return image_embeddings, hq_features
