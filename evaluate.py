import json
from torchvision import ops
from torch.nn import DataParallel

from models.geco_infer import build_model
from utils.data import FSC147Dataset, resize_and_pad
from utils.arg_parser import get_argparser

import argparse
import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T

DATASETS = {
    "fsc147": FSC147Dataset,
}


@torch.no_grad()
def evaluate(args):

    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    model = DataParallel(
        build_model(args).to(device), device_ids=[gpu], output_device=gpu
    )
    state_dict = torch.load(os.path.join(args.model_path, f"{args.model_name}.pth"))[
        "model"
    ]
    state_dict = {
        k if "module." in k else "module." + k: v for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)

    for split in ["val", "test"]:
        test = DATASETS[args.dataset](
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
            return_ids=True,
            evaluation=True,
        )
        test_loader = DataLoader(
            test,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers,
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()

        predictions = dict()
        predictions["categories"] = [{"name": "fg", "id": 1}]
        predictions["images"] = list()
        predictions["annotations"] = list()
        anno_id = 1

        for img, bboxes, density_map, ids, gt_bboxes, scaling_factor, padwh in tqdm(
            test_loader
        ):
            img = img.to(device)
            bboxes = bboxes.to(device)
            gt_bboxes = gt_bboxes.to(device)

            outputs, ref_points, centerness, outputs_coord, masks = model(img, bboxes)

            (
                num_objects_gt,
                num_objects_pred,
                nms_bboxes,
                nms_scores,
                nms_masks,
                nms_ref_points,
            ) = postprocess(
                img,
                bboxes,
                gt_bboxes,
                outputs,
                ref_points,
                centerness,
                padwh,
                test,
                ids,
                device,
                plot=False,
            )

            for idx in range(img.shape[0]):
                img_info = {
                    "id": test.map_img_name_to_ori_id()[
                        test.image_names[ids[idx].item()]
                    ],
                    "file_name": "None",
                }
                bboxes = ops.box_convert(nms_bboxes[idx], "xyxy", "xywh")
                bboxes = bboxes * img.shape[-1] / scaling_factor[idx]
                for idxi in range(len(nms_bboxes[idx])):
                    box = bboxes[idxi].detach().cpu()
                    anno = {
                        "id": anno_id,
                        "image_id": test.map_img_name_to_ori_id()[
                            test.image_names[ids[idx].item()]
                        ],
                        "area": int((box[2] * box[3]).item()),
                        "bbox": [
                            int(box[0].item()),
                            int(box[1].item()),
                            int(box[2].item()),
                            int(box[3].item()),
                        ],
                        "category_id": 1,
                        "score": float(nms_scores[idx][idxi].item()),
                    }
                    anno_id += 1
                    predictions["annotations"].append(anno)
                predictions["images"].append(img_info)
            num_objects_gt = density_map.flatten(1).sum(dim=1)
            num_objects_pred = torch.tensor(num_objects_pred)
            ae += torch.abs(num_objects_gt - num_objects_pred).sum()
            se += torch.pow(num_objects_gt - num_objects_pred, 2).sum()
        print(
            f"{split.capitalize()} set",
            f"MAE: {ae.item() / len(test):.2f}",
            f"RMSE: {torch.sqrt(se / len(test)).item():.2f}",
        )
        with open("geco" + "_" + split + ".json", "w") as handle:
            json.dump(predictions, handle)


@torch.no_grad()
def evaluate_zero_shot(args):
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    model = DataParallel(
        build_model(args).to(device), device_ids=[gpu], output_device=gpu
    )
    state_dict = torch.load(os.path.join(args.model_path, f"{args.model_name}.pth"))[
        "model"
    ]
    state_dict = {
        k if "module." in k else "module." + k: v for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)

    for split in ["val", "test"]:
        test = DATASETS[args.dataset](
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
            return_ids=True,
            evaluation=True,
            zero_shot=args.zero_shot,
        )
        test_loader = DataLoader(
            test,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers,
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()

        predictions = dict()
        predictions["categories"] = [{"name": "fg", "id": 1}]
        predictions["images"] = list()
        predictions["annotations"] = list()
        anno_id = 1
        s_f = 12
        for img, bboxes, density_map, ids, gt_bboxes, scaling_factor, padwh in tqdm(
            test_loader
        ):
            img = img.to(device)
            bboxes = bboxes.to(device)
            gt_bboxes = gt_bboxes.to(device)
            scaling_factor = scaling_factor.to(device)

            outputs, ref_points, centerness, outputs_coord, masks = model(img, bboxes)

            pred_boxes = outputs[0]["pred_boxes"]
            scores = outputs[0]["box_v"]
            keep = ops.nms(
                pred_boxes[scores > scores.max() / s_f],
                scores[scores > scores.max() / s_f],
                0.5,
            )
            boxes = pred_boxes[scores > scores.max() / s_f][keep]

            scaled_bboxes = boxes * img.shape[-1]

            a_dim = (
                (scaled_bboxes[:, 2] - scaled_bboxes[:, 0]).mean()
                + (scaled_bboxes[:, 3] - scaled_bboxes[:, 1]).mean()
            ) / 2
            scaling_factor_ = min(1.0, 60 / a_dim.item())
            if scaling_factor_ < 1:
                img_norm = img - img.min()
                img_norm = img_norm / img_norm.max()
                scaling_factor = scaling_factor * scaling_factor_

                img, bboxes, gt_bboxes, scaling_factor_, _ = resize_and_pad(
                    img_norm.squeeze(),
                    bboxes.squeeze().cpu(),
                    gt_bboxes=gt_bboxes.squeeze(),
                    size=img.shape[-1],
                    full_stretch=True,
                    downscale_factor=scaling_factor_,
                )
                img = T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )(img).unsqueeze(0)
                bboxes = bboxes.unsqueeze(0)
                gt_bboxes = gt_bboxes.unsqueeze(0)
                scaling_factor = scaling_factor.unsqueeze(0)
                outputs, ref_points, centerness, outputs_coord, masks = model(
                    img, bboxes
                )

            (
                num_objects_gt,
                num_objects_pred,
                nms_bboxes,
                nms_scores,
                nms_masks,
                nms_ref_points,
            ) = postprocess(
                img,
                bboxes,
                gt_bboxes,
                outputs,
                ref_points,
                centerness,
                padwh,
                test,
                ids,
                device,
                plot=False,
                s_f=s_f,
            )

            for idx in range(img.shape[0]):
                img_info = {
                    "id": test.map_img_name_to_ori_id()[
                        test.image_names[ids[idx].item()]
                    ],
                    "file_name": "None",
                }
                bboxes = ops.box_convert(nms_bboxes[idx], "xyxy", "xywh")
                bboxes = bboxes * img.shape[-1] / scaling_factor[idx]
                for idxi in range(len(nms_bboxes[idx])):
                    box = bboxes[idxi].detach().cpu()
                    anno = {
                        "id": anno_id,
                        "image_id": test.map_img_name_to_ori_id()[
                            test.image_names[ids[idx].item()]
                        ],
                        "area": int((box[2] * box[3]).item()),
                        "bbox": [
                            int(box[0].item()),
                            int(box[1].item()),
                            int(box[2].item()),
                            int(box[3].item()),
                        ],
                        "category_id": 1,
                        "score": float(nms_scores[idx][idxi].item()),
                    }
                    anno_id += 1
                    predictions["annotations"].append(anno)
                predictions["images"].append(img_info)
            num_objects_gt = density_map.flatten(1).sum(dim=1)
            num_objects_pred = torch.tensor(num_objects_pred)
            ae += torch.abs(num_objects_gt - num_objects_pred).sum()
            se += torch.pow(num_objects_gt - num_objects_pred, 2).sum()
        print(
            f"{split.capitalize()} set",
            f"MAE: {ae.item() / len(test):.2f}",
            f"RMSE: {torch.sqrt(se / len(test)).item():.2f}",
        )
        with open("geco_0shot_" + split + ".json", "w") as handle:
            json.dump(predictions, handle)


def postprocess(
    img,
    bboxes,
    gt_bboxes,
    outputs,
    ref_points,
    centerness,
    padwh,
    test,
    ids,
    device,
    plot=False,
    s_f=8,
):
    num_objects_gt = []
    num_objects_pred = []
    nms_bboxes = []
    nms_scores = []
    nms_masks = []
    nms_ref_points = []
    bs, c, h, w = img.shape
    for idx in range(img.shape[0]):

        target_bboxes = (
            gt_bboxes[idx][torch.logical_not((gt_bboxes[idx] == 0).all(dim=1))]
            / img.shape[-1]
        )
        if len(outputs[idx]["pred_boxes"][-1]) == 0:
            nms_bboxes.append(torch.zeros((0, 4)))
            nms_scores.append(torch.zeros((0)))
            num_objects_pred.append(0)
            num_objects_gt.append(len(target_bboxes))
        else:
            keep = ops.nms(
                outputs[idx]["pred_boxes"][
                    outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / s_f
                ],
                outputs[idx]["box_v"][
                    outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / s_f
                ],
                0.5,
            )

            num_objects_gt.append(len(target_bboxes))

            boxes = (
                outputs[idx]["pred_boxes"][
                    outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / s_f
                ]
            )[keep]
            boxes = torch.clamp(boxes, 0, 1)
            scores = (
                outputs[idx]["scores"][
                    outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / s_f
                ]
            )[keep]
            points = (
                ref_points[idx]
                .permute(1, 0)[
                    (outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / s_f)[0]
                ][keep]
                .permute(1, 0)
            )

            # check if bbox is in the padded part padwh = (paded_w, padded_h) and remove
            maxw = (img.shape[-1] - padwh[0]).to(device)
            maxh = (img.shape[-2] - padwh[1]).to(device)
            scores = scores[
                (boxes[:, 0] * h < maxw)
                & (boxes[:, 1] * w < maxh)
                & (boxes[:, 2] * h < maxw)
                & (boxes[:, 3] * w < maxh)
            ]
            boxes = boxes[
                (boxes[:, 0] * h < maxw)
                & (boxes[:, 1] * w < maxh)
                & (boxes[:, 2] * h < maxw)
                & (boxes[:, 3] * w < maxh)
            ]

            nms_bboxes.append(boxes)
            nms_scores.append(scores)
            nms_ref_points.append(points)
            num_objects_pred.append(len(boxes))

            # PLOT
            if plot:
                fig1 = plt.figure(figsize=(8, 8))
                ((ax1_11, ax1_12), (ax1_21, ax1_22)) = fig1.subplots(2, 2)
                fig1.tight_layout(pad=2.5)
                img_ = np.array((img).cpu()[idx].permute(1, 2, 0))  # test.resize512
                img_ = img_ - np.min(img_)
                img_ = img_ / np.max(img_)
                ax1_11.imshow(img_)
                ax1_11.set_title("Input", fontsize=8)
                bboxes_ = np.array(bboxes.cpu())[idx]
                for i in range(3):
                    ax1_11.plot(
                        [
                            bboxes_[i][0],
                            bboxes_[i][0],
                            bboxes_[i][2],
                            bboxes_[i][2],
                            bboxes_[i][0],
                        ],
                        [
                            bboxes_[i][1],
                            bboxes_[i][3],
                            bboxes_[i][3],
                            bboxes_[i][1],
                            bboxes_[i][1],
                        ],
                        c="r",
                    )
                ax1_12.imshow(img_)
                ax1_12.set_title("gt bboxes", fontsize=8)
                target_bboxes = gt_bboxes[idx][
                    torch.logical_not((gt_bboxes[idx] == 0).all(dim=1))
                ]
                bboxes_ = ((target_bboxes)).detach().cpu()
                for i in range(len(bboxes_)):
                    ax1_12.plot(
                        [
                            bboxes_[i][0],
                            bboxes_[i][0],
                            bboxes_[i][2],
                            bboxes_[i][2],
                            bboxes_[i][0],
                        ],
                        [
                            bboxes_[i][1],
                            bboxes_[i][3],
                            bboxes_[i][3],
                            bboxes_[i][1],
                            bboxes_[i][1],
                        ],
                        c="g",
                    )
                ax1_21.imshow(img_)

                bboxes_pred = nms_bboxes[idx]
                bboxes_ = ((bboxes_pred * img_.shape[0])).detach().cpu()
                for i in range(len(bboxes_)):
                    ax1_21.plot(
                        [
                            bboxes_[i][0],
                            bboxes_[i][0],
                            bboxes_[i][2],
                            bboxes_[i][2],
                            bboxes_[i][0],
                        ],
                        [
                            bboxes_[i][1],
                            bboxes_[i][3],
                            bboxes_[i][3],
                            bboxes_[i][1],
                            bboxes_[i][1],
                        ],
                        c="orange",
                        linewidth=0.5,
                    )
                ax1_21.set_title(
                    "#GT-#PRED=" + str(len(target_bboxes) - len(bboxes_pred))
                )
                res = T.Resize((1024, 1024))
                ax1_21.imshow(res(centerness).detach().cpu()[idx][0], alpha=0.6)
                ax1_22.set_title(
                    "Pred, min:"
                    + str(round(torch.min(centerness[idx][0]).item(), 4))
                    + ", max:"
                    + str(round(torch.max(centerness[idx][0]).item(), 4))
                    + " err:"
                    + str(
                        round(
                            torch.abs(
                                centerness[idx].flatten(1).sum(dim=1)
                                - centerness[idx].flatten(1).sum(dim=1)
                            )
                            .sum()
                            .item(),
                            2,
                        )
                    ),
                    fontsize=8,
                )

                img_name = test.image_names[ids[idx].item()]
                # plot ref points
                plt.savefig(img_name, dpi=300)

    return (
        num_objects_gt,
        num_objects_pred,
        nms_bboxes,
        nms_scores,
        nms_masks,
        nms_ref_points,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GeCo", parents=[get_argparser()])
    args = parser.parse_args()
    print(args)
    if args.zero_shot:
        evaluate_zero_shot(args)
    else:
        evaluate(args)
