import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, resnet101
from torchvision.ops import FeaturePyramidNetwork

class GPSRetinaNet(nn.Module):
    """
    GPS-RetinaNet model for object detection and geo-localization.

    :param num_classes: int, number of object classes.
    :param backbone: str, name of the backbone model ('resnet50' or 'resnet101').
    :param pretrained: bool, if True, loads pretrained weights for the backbone.
    """
    def __init__(self, num_classes, backbone='resnet50', pretrained=True):
        super(GPSRetinaNet, self).__init__()

        # Load the backbone model
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            in_channels_list = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            self.backbone = resnet101(pretrained=pretrained)
            in_channels_list = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract layers from the backbone
        self.layer1 = nn.Sequential(*list(self.backbone.children())[:5])
        self.layer2 = self.backbone.layer2  
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4 

        # Create Feature Pyramid Network using backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,  # Channels of the layers we are going to use for FPN
            out_channels=256
        )
        
        # Classification, bounding box, and GPS regression subnets
        self.cls_subnet = self._make_subnet(256, num_classes * 9)  # 9 anchors
        self.box_subnet = self._make_subnet(256, 4 * 9)  # 9 anchors
        self.gps_subnet = self._make_subnet(256, 2 * 9)  # 2 coordinates (latitude, longitude) for each anchor
    
    def _make_subnet(self, in_channels, out_channels):
        """
        Creates a subnet for classification, bounding box regression, or GPS regression.

        :param in_channels: int, number of input channels.
        :param out_channels: int, number of output channels.
        :return: nn.Sequential, the subnet.
        """
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the GPS-RetinaNet model.

        :param x: torch.Tensor, input images.
        :return: tuple of lists of torch.Tensor, predicted class logits, bounding box regressions, and GPS regressions.
        """
        # Extract features using the backbone
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # Build feature pyramid
        features = self.fpn({
            '0': c1,
            '1': c2,
            '2': c3,
            '3': c4,
        })
        
        # Predict class, bounding box, and GPS coordinates
        cls_logits = [self.cls_subnet(f) for f in features.values()]
        box_regression = [self.box_subnet(f) for f in features.values()]
        gps_regression = [self.gps_subnet(f) for f in features.values()]
        
        return cls_logits, box_regression, gps_regression
