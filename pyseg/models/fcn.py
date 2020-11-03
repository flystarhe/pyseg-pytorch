from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import resnet

from ._utils import IntermediateLayerGetter


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


def get_model(backbone_name="resnet50", aux_loss=False, pretrained=True, classes=None):
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained, replace_stride_with_dilation=[False, True, True])

    return_layers = {"layer4": "out"}
    if aux_loss:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if classes is None:
        classes = ["_BG", "_FG"]
    num_classes = len(classes)

    aux_classifier = None
    if aux_loss:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    inplanes = 2048
    classifier = FCNHead(inplanes, num_classes)

    model = FCN(classes, backbone, classifier, aux_classifier)
    return model
