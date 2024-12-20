import argparse


def get_argparser():

    parser = argparse.ArgumentParser("LOCA parser", add_help=False)

    parser.add_argument("--model_name", default="GeCo_updated", type=str)
    parser.add_argument("--model_name_resumed", default="", type=str)
    parser.add_argument("--data_path", default="/storage/datasets/fsc147", type=str)
    parser.add_argument("--model_path", default="./", type=str)
    parser.add_argument("--image_path", default="./", type=str)
    parser.add_argument("--dataset", default="fsc147", type=str)
    parser.add_argument("--reduction", default=16, type=int)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--emb_dim", default=256, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--kernel_dim", default=1, type=int)
    parser.add_argument("--num_objects", default=3, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--backbone_lr", default=0, type=float)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--max_grad_norm", default=0.1, type=float)
    parser.add_argument("--tiling_p", default=0.5, type=float)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--cost_bbox",
        default=1,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    parser.add_argument("--focal_alpha", default=0.25, type=float)
    parser.add_argument("--output_masks", action="store_true")

    return parser
