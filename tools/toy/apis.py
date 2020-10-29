import albumentations as A
import cv2 as cv
import json
import numpy as np
import os
from collections import defaultdict
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import resnet
from torchvision.models.segmentation import IntermediateLayerGetter
from torchvision.transforms import functional as F


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


class FCNHead(nn.Sequential):

    def __init__(self, in_channels, channels):
        inter_channels = 256
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class FCN(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self, classes, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.classes = classes
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.size()
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            result["aux"] = x

        result["input_shape"] = input_shape
        return result


class ToyDataset(Dataset):
    """COCO format."""

    def __init__(self, data_path, split):
        splits = ["train", "test", "val"]
        assert split in splits, "Split '{}' not supported for Toy".format(split)
        self._data_path, self._split = data_path, split

        ann_file = os.path.join(data_path, "annotations/{}.json".format(split))
        assert os.path.exists(ann_file), "Annotation file not found: {}".format(ann_file)
        self._ann_file = ann_file

        self.create_index()

        self.transform = A.Compose([
            A.RandomCrop(height=800, width=800, p=1.0),
            A.RandomBrightnessContrast(p=0.2),
            A.HorizontalFlip(p=0.5),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def create_index(self):
        coco = load_json(self._ann_file)

        cats = coco["categories"]
        classes = [cat["name"] for cat in cats]
        cat2label = {cat["id"]: i for i, cat in enumerate(cats)}

        imgs = {img["id"]: img["file_name"] for img in coco["images"]}

        anns = defaultdict(list)
        for ann in coco["annotations"]:
            x1, y1, w, h = ann["bbox"]
            bbox = [x1, y1, x1 + w, y1 + h]
            label = cat2label[ann["category_id"]]
            anns[ann["image_id"]].append(dict(bbox=bbox, label=label))

        self.imgs = imgs
        self.anns = anns
        self.classes = classes
        self.ids = list(sorted(self.imgs.keys()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        filename = self.imgs[img_id]

        bboxes, labels = [], []
        for ann in self.anns[img_id]:
            bboxes.append(ann["bbox"])
            labels.append(ann["label"])

        image = cv.imread(os.path.join(self._data_path, filename))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self._split == "train":
            for _ in range(30):
                transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
                if len(bboxes) == 0 or len(transformed["bboxes"]) > 0:
                    break
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]

        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std, inplace=True)

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        target = dict(id=img_id, filename=filename, bboxes=bboxes, labels=labels)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images, 0)
    return images, targets


def get_model(backbone_name="resnet50", num_classes=1, aux_loss=False, pretrained=True, classes=None):
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained, replace_stride_with_dilation=[False, True, True])

    return_layers = {"layer4": "out"}
    if aux_loss:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux_loss:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    inplanes = 2048
    classifier = FCNHead(inplanes, num_classes)

    model = FCN(classes, backbone, classifier, aux_classifier)
    return model
