import albumentations as A
import cv2 as cv
import numpy as np
import os
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


from pyseg.utils.io import load_json


class ToyDataset(Dataset):
    """COCO format.

    DATA_ROOT
    ├── annotations
    │   ├── test.json
    │   ├── train.json
    │   └── val.json
    ├── coco.json
    └── images
        ├── CODE1/
        ├── CODE2/
        ├── ...
        └── CODEn/
    """

    def __init__(self, data_path, split):
        splits = ["train", "test", "val"]
        assert split in splits, "Split '{}' not supported for Toy".format(split)
        self._data_path, self._split = data_path, split

        ann_file = os.path.join(data_path, "annotations/{}.json".format(split))
        assert os.path.exists(ann_file), "Annotation file not found: {}".format(ann_file)
        self._ann_file = ann_file

        self.create_index()

        if split == "train":
            self.transform = A.Compose([
                #A.Resize(height=1024, width=1024, interpolation=cv.INTER_LINEAR, p=1.0),
                #A.LongestMaxSize(max_size=1024, interpolation=cv.INTER_LINEAR, p=1.0),
                A.SmallestMaxSize(max_size=512, interpolation=cv.INTER_LINEAR, p=1.0),
                A.RandomResizedCrop(height=480, width=480, scale=(0.08, 1.0), p=1.0),
                #A.RandomCrop(height=256, width=256, p=1.0),
                A.RandomBrightnessContrast(p=0.2),
                A.HorizontalFlip(p=0.5),
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))
        else:
            self.transform = A.Compose([
                A.SmallestMaxSize(max_size=512, interpolation=cv.INTER_LINEAR, p=1.0),
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def create_index(self):
        coco = load_json(self._ann_file)

        cats = coco["categories"]
        classes = ["_BG"] + [cat["name"] for cat in cats]
        cat2label = {cat["id"]: i for i, cat in enumerate(cats, 1)}

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

        image = cv.imread(os.path.join(self._data_path, filename), 1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self._split == "train":
            for _ in range(30):
                transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
                if len(bboxes) == 0 or len(transformed["bboxes"]) > 0:
                    break
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]
        else:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]

        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std, inplace=True)
        target = dict(id=img_id, filename=filename, bboxes=bboxes, labels=labels)
        return image, target

    def __len__(self):
        return len(self.ids)
