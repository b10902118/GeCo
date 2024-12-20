import os
import json
import argparse
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TVF


from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
    (img, bboxes, image_names, gt_bboxes, dmap) = zip(*batch)
    if None in gt_bboxes:
        return None, None, None, torch.stack(image_names), None, None
    gt_bboxes_pad = pad_sequence(gt_bboxes, batch_first=True, padding_value=0)
    img = torch.stack(img)
    bboxes = torch.stack(bboxes)
    image_names = torch.stack(image_names)
    dmaps = torch.stack(dmap)
    gt_bboxes = gt_bboxes_pad
    return img, bboxes, image_names, gt_bboxes, dmaps


def xywh_to_x1y1x2y2(xywh):
    x, y, w, h = xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]


def resize_and_pad(
    img,
    bboxes,
    density_map=None,
    size=1024.0,
    gt_bboxes=None,
    full_stretch=True,
    downscale_factor=1,
):
    channels, original_height, original_width = img.shape
    longer_dimension = max(original_height, original_width)
    scaling_factor = size / longer_dimension

    if not full_stretch:
        scaled_bboxes = bboxes * scaling_factor

        a_dim = (
            (scaled_bboxes[:, 2] - scaled_bboxes[:, 0]).mean()
            + (scaled_bboxes[:, 3] - scaled_bboxes[:, 1]).mean()
        ) / 2
        scaling_factor = min(1.0, 80 / a_dim.item()) * scaling_factor

    if downscale_factor != 1:
        scaling_factor = scaling_factor * downscale_factor

    resized_img = torch.nn.functional.interpolate(
        img.unsqueeze(0),
        scale_factor=scaling_factor,
        mode="bilinear",
        align_corners=False,
    )

    if max(resized_img.shape) <= 1024:
        size = 1024
    size = int(size)
    pad_height = max(0, size - resized_img.shape[2])
    pad_width = max(0, size - resized_img.shape[3])

    padded_img = torch.nn.functional.pad(
        resized_img, (0, pad_width, 0, pad_height), mode="constant", value=0
    )[0]

    if density_map is not None:
        original_sum = density_map.sum()
        _, w0, h0 = density_map.shape
        _, W, H = img.shape
        resized_density_map = torch.nn.functional.interpolate(
            density_map.unsqueeze(0), size=(W, H), mode="bilinear", align_corners=False
        )
        resized_density_map = torch.nn.functional.interpolate(
            resized_density_map,
            scale_factor=scaling_factor,
            mode="bilinear",
            align_corners=False,
        )
        padded_density_map = torch.nn.functional.pad(
            resized_density_map, (0, pad_width, 0, pad_height), mode="constant", value=0
        )[0]
        padded_density_map = T.Resize((512, 512), antialias=True)(padded_density_map)
        padded_density_map = (
            padded_density_map / padded_density_map.sum() * original_sum
        )

    bboxes = bboxes * torch.tensor(
        [scaling_factor, scaling_factor, scaling_factor, scaling_factor]
    )

    if gt_bboxes is not None and density_map is not None:
        gt_bboxes = gt_bboxes * torch.tensor(
            [scaling_factor, scaling_factor, scaling_factor, scaling_factor]
        )
        return (
            padded_img,
            bboxes,
            padded_density_map,
            gt_bboxes,
            scaling_factor,
            (pad_width, pad_height),
        )
    if gt_bboxes is not None:
        return padded_img, bboxes, gt_bboxes, scaling_factor, (pad_width, pad_height)
    if density_map is None and gt_bboxes is None:
        return padded_img, bboxes, scaling_factor

    return padded_img, bboxes, padded_density_map


def tiling_augmentation(
    img, bboxes, resize, jitter, tile_size, hflip_p, gt_bboxes=None, density_map=None
):
    def apply_hflip(tensor, apply):
        return TVF.hflip(tensor) if apply else tensor

    def make_tile(x, num_tiles, hflip, hflip_p, jitter=None):
        result = list()
        for j in range(num_tiles):
            row = list()
            for k in range(num_tiles):
                t = jitter(x) if jitter is not None else x
                # if hflip[j, k] < hflip_p:
                #     t = TVF.hflip(t)
                row.append(t)
            result.append(torch.cat(row, dim=-1))
        return torch.cat(result, dim=-2)

    x_tile, y_tile = tile_size
    y_target, x_target = resize.size
    num_tiles = max(int(x_tile.ceil()), int(y_tile.ceil()))
    # whether to horizontally flip each tile
    hflip = torch.rand(num_tiles, num_tiles)

    img = make_tile(img, num_tiles, hflip, hflip_p, jitter)
    c, h, w = img.shape
    img = resize(img[..., : int(y_tile * y_target), : int(x_tile * x_target)])
    if density_map is not None:
        density_map = make_tile(density_map, num_tiles, hflip, hflip_p)
        density_map = density_map[
            ..., : int(y_tile * y_target), : int(x_tile * x_target)
        ]
        original_sum = density_map.sum()
        density_map = T.Resize((512, 512), antialias=True)(density_map)
        density_map = density_map / density_map.sum() * original_sum

    bboxes = bboxes / torch.tensor([w, h, w, h]) * resize.size[0]
    if gt_bboxes is not None:
        gt_bboxes_ = gt_bboxes / torch.tensor([w, h, w, h]) * resize.size[0]
        gt_bboxes_tiled = torch.cat(
            [
                gt_bboxes_,
                gt_bboxes_ + torch.tensor([0, 512, 0, 512]),
                gt_bboxes_ + torch.tensor([512, 0, 512, 0]),
                gt_bboxes_ + torch.tensor([512, 512, 512, 512]),
            ]
        )
        if density_map is None:
            return img, bboxes, gt_bboxes_tiled
        else:
            return img, bboxes, density_map, gt_bboxes_tiled

    return img, bboxes, density_map


class FSC147Dataset(Dataset):

    def __init__(
        self,
        data_path,
        img_size,
        split="train",
        num_objects=3,
        tiling_p=0.5,
        zero_shot=False,
        return_ids=False,
        evaluation=False,
    ):
        from pycocotools.coco import COCO

        self.split = split
        self.data_path = data_path
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size), antialias=True)
        self.resize512 = T.Resize((512, 512), antialias=True)
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        self.return_ids = return_ids
        self.evaluation = evaluation

        with open(
            os.path.join(self.data_path, "annotations", "Train_Test_Val_FSC_147.json"),
            "rb",
        ) as file:
            splits = json.load(file)
            self.image_names = splits[split]
        with open(
            os.path.join(self.data_path, "annotations", "annotation_FSC147_384.json"),
            "rb",
        ) as file:
            self.annotations = json.load(file)
        self.labels = COCO(
            os.path.join(self.data_path, "annotations", "instances_" + split + ".json")
        )
        self.img_name_to_ori_id = self.map_img_name_to_ori_id()

    def get_gt_bboxes(self, idx):
        coco_im_id = self.img_name_to_ori_id[self.image_names[idx]]
        anno_ids = self.labels.getAnnIds([coco_im_id])
        annotations = self.labels.loadAnns(anno_ids)
        bboxes = []
        for a in annotations:
            bboxes.append(xywh_to_x1y1x2y2(a["bbox"]))
        return bboxes

    def __getitem__(self, idx: int):
        img = Image.open(
            os.path.join(self.data_path, "images_384_VarV2", self.image_names[idx])
        ).convert("RGB")

        gt_bboxes = torch.tensor(self.get_gt_bboxes(idx))

        img = T.Compose(
            [
                T.ToTensor(),
            ]
        )(img)

        bboxes = torch.tensor(
            self.annotations[self.image_names[idx]]["box_examples_coordinates"],
            dtype=torch.float32,
        )[:3, [0, 2], :].reshape(-1, 4)[: self.num_objects, ...]

        density_map = torch.from_numpy(
            np.load(
                os.path.join(
                    self.data_path,
                    "gt_density_map_adaptive_1024_1024_SAME",
                    os.path.splitext(self.image_names[idx])[0] + ".npy",
                )
            )
        ).unsqueeze(0)

        tiled = False

        # data augmentation
        if self.split == "train" and torch.rand(1) < self.tiling_p:
            tiled = True
            tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
            img, bboxes, density_map, gt_bboxes = tiling_augmentation(
                img,
                bboxes,
                self.resize,
                self.jitter,
                tile_size,
                self.horizontal_flip_p,
                gt_bboxes=gt_bboxes,
                density_map=density_map,
            )

        elif self.split == "train":
            img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(
                img, bboxes, density_map, full_stretch=True, gt_bboxes=gt_bboxes
            )
        elif not self.evaluation:
            img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(
                img,
                bboxes,
                density_map,
                gt_bboxes=gt_bboxes,
                full_stretch=False,
                size=1024.0,
            )
        else:
            img_, bboxes_, density_map_, gt_bboxes_, scaling_factor_, padwh_ = (
                resize_and_pad(
                    img,
                    bboxes,
                    density_map,
                    gt_bboxes=gt_bboxes,
                    full_stretch=False if not self.zero_shot else True,
                    size=1024.0,
                )
            )
            if (
                (bboxes_[:, 2] - bboxes_[:, 0]).min() < 25
                and (bboxes_[:, 3] - bboxes_[:, 1]).min() < 25
                and not self.zero_shot
            ):
                img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = (
                    resize_and_pad(
                        img,
                        bboxes,
                        density_map,
                        gt_bboxes=gt_bboxes,
                        full_stretch=False,
                        size=1536.0,
                    )
                )
            else:
                img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = (
                    img_,
                    bboxes_,
                    density_map_,
                    gt_bboxes_,
                    scaling_factor_,
                    padwh_,
                )

        if self.split == "train":
            if not tiled:
                img = self.jitter(img)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if (
            self.split == "train"
            and not tiled
            and torch.rand(1) < self.horizontal_flip_p
        ):
            img = TVF.hflip(img)
            density_map = TVF.hflip(density_map)
            bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]
            gt_bboxes[:, [0, 2]] = self.img_size - gt_bboxes[:, [2, 0]]

        gt_bboxes = torch.clamp(gt_bboxes, min=0, max=1024)

        if self.evaluation:
            return (
                img,
                bboxes,
                density_map,
                torch.tensor(idx),
                gt_bboxes,
                scaling_factor,
                padwh,
            )

        else:
            return img, bboxes, torch.tensor(idx), gt_bboxes, density_map

    def __len__(self):
        return len(self.image_names)

    def map_img_name_to_ori_id(
        self,
    ):
        all_coco_imgs = self.labels.imgs
        map_name_2_id = dict()
        for k, v in all_coco_imgs.items():
            img_id = v["id"]
            img_name = v["file_name"]
            map_name_2_id[img_name] = img_id
        return map_name_2_id


def generate_density_maps(data_path, target_size=(1024, 1024)):
    from tqdm import tqdm
    from scipy.ndimage import gaussian_filter

    with open(
        os.path.join(data_path, "annotations/annotation_FSC147_384.json"), "rb"
    ) as file:
        annotations = json.load(file)

    if not os.path.exists(
        os.path.join(data_path, "gt_density_map_adaptive_1024_1024_SAME")
    ):
        os.makedirs(os.path.join(data_path, "gt_density_map_adaptive_1024_1024_SAME"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, (image_name, ann) in enumerate(tqdm(annotations.items())):
        _, h, w = T.ToTensor()(
            Image.open(os.path.join(data_path, "images_384_VarV2", image_name))
        ).size()
        h_ratio, w_ratio = target_size[0] / h, target_size[1] / w

        points = (
            torch.tensor(ann["points"], device=device)
            * torch.tensor([w_ratio, h_ratio], device=device)
        ).long()
        points[:, 0] = points[:, 0].clip(0, target_size[1] - 1)
        points[:, 1] = points[:, 1].clip(0, target_size[0] - 1)

        sigmas = np.array([2, 2])

        dmap = torch.zeros(*target_size)
        for p in range(points.size(0)):
            dmap[points[p, 1], points[p, 0]] += 1
        dmap = gaussian_filter(dmap.cpu().numpy(), sigmas)

        np.save(
            os.path.join(
                data_path,
                "gt_density_map_adaptive_1024_1024_SAME",
                os.path.splitext(image_name)[0] + ".npy",
            ),
            dmap,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Density map generator", add_help=False)
    parser.add_argument("--data_path", default="/storage/datasets/fsc147/", type=str)
    parser.add_argument("--image_size", default=1024, type=int)
    args = parser.parse_args()
    generate_density_maps(args.data_path, (args.image_size, args.image_size))
