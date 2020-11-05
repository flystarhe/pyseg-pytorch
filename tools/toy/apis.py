from pyseg.datasets.toy import ToyDataset as ToyDataset
from pyseg.models.fcn import get_model
from pyseg.utils.misc import collate_fn
from pyseg.losses.utils import _make_target
from pyseg.losses.utils import _balance_target


import cv2 as cv
import numpy as np
import os
import torch


def draw_bbox(img, bboxes, labels):
    img = np.ascontiguousarray(img)
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = [int(val) for val in bbox]
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        cv.putText(img, "{}".format(label), (x1, y1), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0))
    return img


def simple_show(image, output, target, out_dir, mapping):
    # image (Tensor[C, H, W]): type
    # output (Tensor[H, W]): type
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=dtype, device=device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=dtype, device=device)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)

    image = image.mul(std).add(mean)
    image = image.mul(255).byte()

    output = torch.stack([output for _ in range(3)], 0)
    output = output.mul(255).byte()

    image = np.transpose(image.cpu().numpy(), (1, 2, 0))
    output = np.transpose(output.cpu().numpy(), (1, 2, 0))

    h, w, _ = image.shape
    output = cv.resize(output, (w, h), interpolation=cv.INTER_NEAREST)

    bboxes = target["bboxes"]
    labels = target["labels"]
    filename = target["filename"]

    labels = [mapping.get(label, "_FG") for label in labels]

    image = draw_bbox(image, bboxes, labels)
    output = draw_bbox(output, bboxes, labels)

    out_file = os.path.join(out_dir, filename)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    img = np.concatenate((image, output), axis=1)
    cv.imwrite(out_file, cv.cvtColor(img, cv.COLOR_RGB2BGR))
    return out_file
