from torchvision.ops import roi_align

from utils.box_ops import boxes_with_scores
from .backbone import Backbone
from .common import MLP
from .DQE import DQE
from .prompt_encoder import PromptEncoder_DQE
import torch
from torch import nn
from torchvision.transforms import Resize


class GeCo(nn.Module):

    def __init__(
        self,
        image_size: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        train_backbone: bool,
        reduction: int,
        zero_shot: bool,
        model_path: str,
    ):
        super(GeCo, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.num_heads = num_heads
        self.num_classes = 1
        self.model_path = model_path
        self.backbone = Backbone(
            requires_grad=train_backbone, image_size=image_size, model_path=model_path
        )

        self.class_embed = nn.Sequential(nn.Linear(emb_dim, 1), nn.LeakyReLU())
        self.bbox_embed = MLP(emb_dim, emb_dim, 4, 3)

        self.emb_dim = 256
        self.adapt_features = DQE(
            transformer_dim=self.emb_dim,
            num_prototype_attn_steps=3,
            num_image_attn_steps=2,
        )

        self.prompt_encoder = PromptEncoder_DQE(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )

        self.shape_or_objectness = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1**2 * emb_dim),
        )
        self.resize = Resize((512, 512))

    def forward(self, x, bboxes):
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects

        # src = x
        src, src_hq = self.backbone(x)

        bs, c, h, w = src.size()

        bboxes_roi = torch.cat(
            [
                torch.arange(bs, requires_grad=False)
                .to(bboxes.device)
                .repeat_interleave(self.num_objects)
                .reshape(-1, 1),
                bboxes.flatten(0, 1),
            ],
            dim=1,
        )

        # Roi align
        exemplars = (
            roi_align(
                src,
                boxes=bboxes_roi,
                output_size=self.kernel_dim,
                spatial_scale=1.0 / self.reduction,
                aligned=True,
            )
            .permute(0, 2, 3, 1)
            .reshape(bs, self.num_objects * self.kernel_dim**2, self.emb_dim)
        )

        box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device)
        box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]
        box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]

        # Encode shape
        shape = self.shape_or_objectness(box_hw).reshape(bs, -1, self.emb_dim)
        prototype_embeddings = torch.cat([exemplars, shape], dim=1)

        # adapt image feature with prototypes
        adapted_f = self.adapt_features(
            image_embeddings=src,
            image_pe=self.prompt_encoder.get_dense_pe(),
            prototype_embeddings=prototype_embeddings,
            hq_features=src_hq,
        )

        # Predict class [fg, bg] and l,r,t,b
        bs, c, w, h = adapted_f.shape
        adapted_f = adapted_f.view(bs, self.emb_dim, -1).permute(0, 2, 1)
        centerness = self.class_embed(adapted_f).view(bs, w, h, 1).permute(0, 3, 1, 2)
        outputs_coord = (
            self.bbox_embed(adapted_f).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
        )
        outputs, ref_points = boxes_with_scores(centerness, outputs_coord)

        return outputs, ref_points, centerness, outputs_coord


def build_model(args):
    assert args.reduction in [4, 8, 16]

    return GeCo(
        image_size=args.image_size,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        kernel_dim=args.kernel_dim,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        model_path=args.model_path,
    )
