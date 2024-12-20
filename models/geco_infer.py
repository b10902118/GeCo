from torchvision.ops import roi_align

from utils.box_ops import boxes_with_scores
from .backbone import Backbone
from .common import MLP
from .DQE import DQE
import torch
from torch import nn
from torch.nn import functional as F
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
        return_masks: bool = False,
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
        self.backbone = Backbone(requires_grad=train_backbone, image_size=image_size)
        self.class_embed = nn.Sequential(nn.Linear(emb_dim, 1), nn.LeakyReLU())
        self.bbox_embed = MLP(emb_dim, emb_dim, 4, 3)
        self.return_masks = return_masks

        self.emb_dim = 256
        self.adapt_features = DQE(
            transformer_dim=self.emb_dim,
            num_prototype_attn_steps=3,
            num_image_attn_steps=2,
            zero_shot=zero_shot,
        )
        from .prompt_encoder import PromptEncoder_DQE

        self.prompt_encoder = PromptEncoder_DQE(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )

        from segment_anything.modeling import (
            MaskDecoder,
            TwoWayTransformer,
            PromptEncoder,
        )

        prompt_embed_dim = 256
        image_embedding_size = 64
        image_size = 1024
        self.prompt_encoder_sam = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        image_embedding_size = 96
        image_size = 1536
        self.prompt_encoder_sam_ = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        checkpoint = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            map_location="cpu",
        )
        state_dict = {
            k.replace("mask_decoder.", ""): v
            for k, v in checkpoint.items()
            if "mask_decoder" in k
        }
        self.mask_decoder.load_state_dict(state_dict)
        state_dict = {
            k.replace("prompt_encoder.", ""): v
            for k, v in checkpoint.items()
            if "prompt_encoder" in k
        }
        self.prompt_encoder_sam.load_state_dict(state_dict)
        self.prompt_encoder_sam_.load_state_dict(state_dict)

        if self.zero_shot:
            self.exemplars = nn.Parameter(torch.randn(1, emb_dim))
        else:
            self.shape_or_objectness = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, 1**2 * emb_dim),
            )
        self.resize = Resize((512, 512))

    def refine_bounding_boxes(self, features, outputs, return_masks=False):

        batch_masks = []
        batch_iou = []
        batch_bboxes = []
        for i in range(len(outputs)):
            step = 50
            masks = []
            iou_predictions = []
            corrected_bboxes_ = []
            for box_i in range(step, len(outputs[i]["pred_boxes"][0]) + step, step):
                box = (
                    outputs[i]["pred_boxes"][0][(box_i - step) : box_i]
                    * features.shape[-1]
                    * 16
                )
                if features.shape[-1] * 16 == 1024:
                    sparse_embeddings, dense_embeddings = self.prompt_encoder_sam(
                        points=None,
                        boxes=box,
                        masks=None,
                    )
                else:
                    sparse_embeddings, dense_embeddings = self.prompt_encoder_sam_(
                        points=None,
                        boxes=box,
                        masks=None,
                    )
                # # # Predict masks
                masks_, iou_predictions_ = self.mask_decoder(
                    image_embeddings=features[i].unsqueeze(0),
                    image_pe=self.prompt_encoder_sam.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                masks_ = F.interpolate(
                    masks_,
                    (features.shape[-1] * 16, features.shape[-1] * 16),
                    mode="bilinear",
                    align_corners=False,
                )
                masks_ = masks_ > 0
                if return_masks:
                    masks_ = masks_[..., :1024, :1024]
                    masks.append(masks_)
                iou_predictions.append(iou_predictions_)

                corrected_bboxes = torch.zeros((masks_.shape[0], 4), dtype=torch.float)
                masks_ = masks_[:, 0]
                for index, mask_i in enumerate(masks_):
                    y, x = torch.where(mask_i != 0)
                    if y.shape[0] > 0 and x.shape[0] > 0:
                        corrected_bboxes[index, 0] = torch.min(x)
                        corrected_bboxes[index, 1] = torch.min(y)
                        corrected_bboxes[index, 2] = torch.max(x)
                        corrected_bboxes[index, 3] = torch.max(y)
                corrected_bboxes_.append(corrected_bboxes)
            if len(corrected_bboxes_) > 0:
                if return_masks:
                    batch_masks.append(torch.cat(masks, dim=0)[:, 0])
                else:
                    batch_masks.append([])
                batch_bboxes.append(torch.cat(corrected_bboxes_))
                batch_iou.append(torch.cat(iou_predictions).permute(1, 0))
            else:
                batch_masks.append([])
                batch_bboxes.append(torch.tensor([]).to(features.device))
                batch_iou.append(torch.tensor([]).to(features.device))
        return batch_masks, batch_iou, batch_bboxes

    def forward(self, img, bboxes):
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects

        src, src_hq = self.backbone(img)
        bs, c, h, w = src.size()

        if not self.zero_shot:
            prototype_embeddings = self.create_prototypes(src, bboxes)

        else:  # zero shot
            prototype_embeddings = self.exemplars.expand(bs, -1, -1)
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
        outputs, ref_points = boxes_with_scores(
            centerness, outputs_coord, batch_thresh=0.001
        )
        masks, ious, corrected_bboxes = self.refine_bounding_boxes(
            src, outputs, return_masks=self.return_masks
        )

        for i in range(len(outputs)):
            outputs[i]["scores"] = ious[i]
            outputs[i]["pred_boxes"] = (
                corrected_bboxes[i].to(outputs[i]["pred_boxes"].device).unsqueeze(0)
                / img.shape[-1]
            )

        return outputs, ref_points, centerness, outputs_coord, masks

    def create_prototypes(self, src, bboxes):
        bs = src.size(0)
        self.num_objects = bboxes.size(1)

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
        self.kernel_dim = 1

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

        shape = self.shape_or_objectness(box_hw).reshape(bs, -1, self.emb_dim)
        prototype_embeddings = torch.cat([exemplars, shape], dim=1)
        return prototype_embeddings


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
        return_masks=args.output_masks,
    )
