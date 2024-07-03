"""
This module defines the GPS Loss for geo-localization tasks.

:module: gps_loss
"""

import torch.nn.functional as F
import torch.nn as nn

class GPSLoss(nn.Module):
    """
    GPS Loss for regressing geo-localization coordinates.

    """
    def __init__(self):
        super(GPSLoss, self).__init__()
    
    def forward(self, gps_pred, gps_target):
        """
        Forward pass of the GPS Loss.

        :param gps_pred: torch.Tensor, predicted GPS coordinates.
        :param gps_target: torch.Tensor, ground truth GPS coordinates.
        :return: torch.Tensor, computed GPS Loss.
        """
        return F.smooth_l1_loss(gps_pred, gps_target)
