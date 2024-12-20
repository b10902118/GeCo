from torch.nn import DataParallel
from models.geco_infer import build_model
from utils.arg_parser import get_argparser
import argparse
import torch
from torchvision import transforms as T
import matplotlib.patches as patches
from PIL import Image
from torchvision import ops
from utils.data import resize_and_pad
import matplotlib.pyplot as plt

bounding_boxes = []
global clicked

# Global variables to track drawing state
rect = None
start_x, start_y = None, None


# Event handler for mouse press (start drawing)
def on_press(event):
    global start_x, start_y, rect
    if event.inaxes:
        start_x, start_y = event.xdata, event.ydata  # Store starting point
        # Create a rectangle (but do not draw yet)
        rect = patches.Rectangle(
            (start_x, start_y), 0, 0, linewidth=2, edgecolor="r", facecolor="none"
        )
        event.inaxes.add_patch(rect)
        plt.draw()  # Update plot to show rectangle (even if not yet drawn)


# Event handler for mouse motion (while drawing)
def on_motion(event):
    global start_x, start_y, rect
    if rect is not None and event.inaxes:
        # Update the width and height of the rectangle based on mouse position
        width = event.xdata - start_x
        height = event.ydata - start_y
        rect.set_width(width)
        rect.set_height(height)
        plt.draw()  # Redraw to update the rectangle while dragging


# Event handler for mouse release (end drawing)
def on_release(event):
    global rect
    # Once mouse is released, we finalize the bounding box
    if rect is not None:
        bounding_boxes.append(
            [
                rect.get_x(),
                rect.get_y(),
                rect.get_x() + rect.get_width(),
                rect.get_y() + rect.get_height(),
            ]
        )
        rect = None  # Reset rect after release


@torch.no_grad()
def demo(args):
    img_path = args.image_path
    global fig, ax

    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    model = DataParallel(
        build_model(args).to(device), device_ids=[gpu], output_device=gpu
    )
    model.load_state_dict(
        torch.load("GeCo.pth", weights_only=True)["model"],
        strict=False,
    )

    model.eval()

    image = T.ToTensor()(Image.open(img_path).convert("RGB"))

    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0))
    plt.axis("off")
    # Connect the click event
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    plt.title("Click and drag to draw bboxes, then close window")
    # Show the image
    plt.show()

    bboxes = torch.tensor(bounding_boxes, dtype=torch.float32)

    img, bboxes, scale = resize_and_pad(image, bboxes, full_stretch=False)
    img = img.unsqueeze(0).to(device)
    bboxes = bboxes.unsqueeze(0).to(device)

    outputs, _, _, _, masks = model(img, bboxes)
    del _
    idx = 0
    thr = 4
    keep = ops.nms(
        outputs[idx]["pred_boxes"][
            outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / thr
        ],
        outputs[idx]["box_v"][
            outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / thr
        ],
        0.5,
    )

    boxes = (
        outputs[idx]["pred_boxes"][
            outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / thr
        ]
    )[keep]

    bboxes = torch.clamp(boxes, 0, 1)

    plt.clf()
    plt.imshow(image.permute(1, 2, 0))
    if args.output_masks:
        masks_ = masks[idx][
            (outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / thr)[0]
        ]
        N_masks = masks_.shape[0]
        indices = torch.randint(
            1, N_masks + 1, (1, N_masks), device=masks_.device
        ).view(-1, 1, 1)
        masks = (masks_ * indices).sum(dim=0)
        mask_display = (
            T.Resize(
                (int(img.shape[2] / scale), int(img.shape[3] / scale)),
                interpolation=T.InterpolationMode.NEAREST,
            )(masks.cpu().unsqueeze(0))[0]
        )[: image.shape[1], : image.shape[2]]
        cmap = plt.cm.tab20  # Use a colormap with distinct colors
        norm = plt.Normalize(vmin=0, vmax=N_masks)
        del masks
        del masks_
        del outputs
        rgba_image = cmap(norm(mask_display))
        rgba_image[mask_display == 0, -1] = 0
        plt.imshow(rgba_image, alpha=0.6)

    pred_boxes = (
        bboxes.cpu() / torch.tensor([scale, scale, scale, scale]) * img.shape[-1]
    )
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]

        plt.plot(
            [box[0], box[0], box[2], box[2], box[0]],
            [box[1], box[3], box[3], box[1], box[1]],
            linewidth=0.7,
            color="orange",
        )

    pred_boxes = bounding_boxes
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]
        plt.plot(
            [box[0], box[0], box[2], box[2], box[0]],
            [box[1], box[3], box[3], box[1], box[1]],
            linewidth=2,
            color="red",
        )
    plt.title("Number of selected objects:" + str(len(bboxes)))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GeCo", parents=[get_argparser()])
    args = parser.parse_args()
    demo(args)
