"""
This module provides a function to load and configure the backbone model for GPS-RetinaNet.

:module: backbone
"""

import torchvision.models as models
import torch.nn as nn

def get_backbone(backbone_name='resnet50', pretrained=True):
    """
    Loads and returns the specified backbone model, removing the fully connected layer.

    :param backbone_name: str, name of the backbone model ('resnet50' or 'resnet101').
    :param pretrained: bool, if True, loads pretrained weights.
    :return: nn.Module, the backbone model without the fully connected layer.
    """
    if backbone_name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    elif backbone_name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
    else:
        raise ValueError("Unsupported backbone model")

    # Remove the fully connected layer
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    return backbone
