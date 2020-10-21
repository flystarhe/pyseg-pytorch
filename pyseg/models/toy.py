import torch
import torch.nn.functional as F
from torch import nn

from pyseg.losses.simple_pool import create_mask
from pyseg.models.backbone.resnet import resnet_fpn_backbone


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        layers = [
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class Toy(nn.Module):

    def __init__(self, num_classes=1, backbone_name="resnet50", pretrained=False, trainable_layers=3, returned_layers=None):
        """
        Arguments:
            backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
            pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
            trainable_layers (int): With 5 meaning all backbone layers are trainable
            returned_layers (list[int] or None): Default: `[2, 3]`
        """
        if returned_layers is None:
            returned_layers = [2, 3]
        stride = 2 ** (returned_layers[0] + 1)

        self.stride = stride
        self.num_classes = num_classes
        self.backbone = resnet_fpn_backbone(backbone_name, pretrained,
                                            trainable_layers=trainable_layers, returned_layers=returned_layers)
        in_channels = self.backbone.out_channels
        self.classifier = FCNHead(in_channels, num_classes)

    def eager_outputs(self, loss, outputs):
        if self.training:
            return loss

        return outputs

    def forward(self, inputs, targets=None):
        # inputs (Tensor): images to be processed
        # targets (List[Dict[BBoxList, LabelList]] or None): ground-truth
        features = self.backbone(inputs)
        outputs = self.classifier(features["feat0"])

        loss = 0.0
        if self.training:
            num_images = inputs.size(0)
            assert num_images == len(targets)

            factor = 1.0 / num_images
            for img_id in range(num_images):
                _loss = self.forward_single(outputs[img_id], targets[img_id])
                loss += _loss * factor

        return self.eager_outputs(loss, outputs)

    def forward_single(self, feat, target):
        # feat (Tensor): a 2D feature map
        # target (Dict[BBoxList, LabelList] or None): ground-truth
        bg_mask, fg_mask = create_mask(7, feat.detach(), target["bboxes"], self.stride)

        loss_bg = 0.0
        if bg_mask.any():
            x = torch.masked_select(feat, bg_mask)
            y = torch.zeros_like(x, requires_grad=False)
            loss_bg = F.binary_cross_entropy_with_logits(x, y)

        loss_fg = 0.0
        if fg_mask.any():
            x = torch.masked_select(feat, fg_mask)
            y = torch.ones_like(x, requires_grad=False)
            loss_fg = F.binary_cross_entropy_with_logits(x, y)

        return (loss_bg + loss_fg) * 0.5
