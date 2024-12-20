from models.geco import build_model
from utils.box_ops import compute_location, BoxList
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser
from utils.losses import ObjectNormalizedL2Loss, Detection_criterion
from time import perf_counter
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from utils.data import pad_collate
import numpy as np
import random


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

DATASETS = {"fsc147": FSC147Dataset}


def train(args):
    if "SLURM_PROCID" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        gpu = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    model = DistributedDataParallel(
        build_model(args).to(device), device_ids=[gpu], output_device=gpu
    )

    backbone_params = dict()
    non_backbone_params = dict()
    for n, p in model.named_parameters():
        if "backbone" in n:
            backbone_params[n] = p
        else:
            non_backbone_params[n] = p

    optimizer = torch.optim.AdamW(
        [
            {"params": non_backbone_params.values()},
            {"params": backbone_params.values(), "lr": args.backbone_lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)
    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f"{args.model_name}.pth"))
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        best = checkpoint["best_val_ae"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        start_epoch = 0
        best = 10000000000000

    criterion = ObjectNormalizedL2Loss()
    det_criterion = Detection_criterion(
        [[-1, 512]],  # sizes,
        "giou",  # iou_loss_type,
        True,  # center_sample,
        [1],  # fpn_strides,
        30,  # pos_radius,
    )

    train = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split="train",
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot,
    )
    val = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split="val",
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
    )
    train_loader = DataLoader(
        train,
        sampler=DistributedSampler(train),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val,
        sampler=DistributedSampler(val),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
    )

    print(rank)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if rank == 0:
            start = perf_counter()
        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)
        train_ae = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)
        val_rmse = torch.tensor(0.0).to(device)

        train_loader.sampler.set_epoch(epoch)
        model.train()

        for img, bboxes, img_name, gt_bboxes, density_map in train_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)

            optimizer.zero_grad()
            _, _, centerness, lrtb = model(img, bboxes)

            lrtb = lrtb * 512
            location = compute_location(lrtb)
            targets = (
                BoxList(gt_bboxes, (args.image_size, args.image_size), mode="xyxy")
                .to(device)
                .resize((512, 512))
            )

            # obtain the number of objects in batch
            with torch.no_grad():
                num_objects = density_map.sum()
                dist.all_reduce(num_objects)
            det_loss = det_criterion(location, lrtb, targets) / num_objects
            main_loss = criterion(centerness, density_map, num_objects)

            loss = main_loss + det_loss
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss += main_loss * img.size(0)
            train_ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - centerness.flatten(1).sum(dim=1)
            ).sum()

        model.eval()
        with torch.no_grad():
            index_val = 0
            for img, bboxes, img_name, gt_bboxes, density_map in val_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)

                optimizer.zero_grad()

                _, _, centerness, lrtb = model(img, bboxes)

                lrtb = lrtb * 512
                location = compute_location(lrtb)
                targets = (
                    BoxList(gt_bboxes, (args.image_size, args.image_size), mode="xyxy")
                    .to(device)
                    .resize((512, 512))
                )

                # obtain the number of objects in batch
                with torch.no_grad():
                    num_objects = density_map.sum()
                    dist.all_reduce(num_objects)
                det_loss = det_criterion(location, lrtb, targets)
                main_loss = criterion(centerness, density_map, num_objects)

                loss = main_loss + det_loss
                val_loss += loss
                val_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - centerness.flatten(1).sum(dim=1)
                ).sum()
                val_rmse += torch.pow(
                    density_map.flatten(1).sum(dim=1)
                    - centerness.flatten(1).sum(dim=1),
                    2,
                ).sum()

                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        dist.all_reduce(train_loss)
        dist.all_reduce(val_loss)
        dist.all_reduce(val_rmse)
        dist.all_reduce(train_ae)
        dist.all_reduce(val_ae)

        scheduler.step()

        if rank == 0:
            end = perf_counter()
            best_epoch = False
            if val_rmse.item() / len(val) < best:
                best = val_rmse.item() / len(val)
                checkpoint = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_ae": val_ae.item() / len(val),
                }

                torch.save(
                    checkpoint, os.path.join(args.model_path, f"{args.model_name}.pth")
                )
                best_epoch = True
            torch.save(
                checkpoint, os.path.join(args.model_path, f"{args.model_name}_last.pth")
            )

            print(
                f"Epoch: {epoch}",
                f"Train loss: {train_loss.item():.3f}",
                f"Val loss: {val_loss.item():.3f}",
                f"Train MAE: {train_ae.item() / len(train):.3f}",
                f"Val MAE: {val_ae.item() / len(val):.3f}",
                f"Val RMSE: {torch.sqrt(val_rmse / len(val)).item():.2f}",
                f"Epoch time: {end - start:.3f} seconds",
                "best" if best_epoch else "",
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GeCo", parents=[get_argparser()])
    args = parser.parse_args()
    print(args)
    train(args)
