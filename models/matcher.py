import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from utils.box_ops import generalized_box_iou, box_iou


class GeCoMatcher(nn.Module):

    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    def forward(self, outputs, targets, ref_points=None):
        with torch.no_grad():
            bs, num_queries = outputs["box_v"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_bbox = outputs["pred_boxes"].flatten(
                0, 1
            )  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            iou, unions = box_iou(out_bbox, tgt_bbox)
            cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            ]

            non_mathced_gt_bbox_idx = np.nonzero(
                np.logical_not(
                    np.in1d(
                        np.array([i for i in range(tgt_bbox.shape[0])]), indices[0][1]
                    )
                )
            )[0]
            non_mathced_gt_bbox_idx = np.concatenate(
                (
                    non_mathced_gt_bbox_idx,
                    torch.where(iou.max(dim=0)[0] == 0)[0].cpu().numpy(),
                )
            )
            non_mathced_gt_bbox_idx = [
                torch.tensor(non_mathced_gt_bbox_idx, dtype=torch.int64).unique()
            ]
            remove_mask = np.logical_not(
                np.in1d(indices[0][1], non_mathced_gt_bbox_idx[0].cpu())
            )
            ind0 = indices[0][0][remove_mask]
            ind1 = indices[0][1][remove_mask]
            non_mathced_pred_bbox_idx = np.nonzero(
                np.logical_not(
                    np.in1d(
                        np.array([i for i in range(out_bbox.shape[0])]), indices[0][0]
                    )
                )
            )[0]

            match_indexes = [
                (
                    torch.as_tensor(ind0, dtype=torch.int64),
                    torch.as_tensor(ind1, dtype=torch.int64),
                )
            ]
            return match_indexes, non_mathced_gt_bbox_idx, non_mathced_pred_bbox_idx


def build_matcher(args):
    return GeCoMatcher(args.cost_class, args.cost_bbox, args.cost_giou)
