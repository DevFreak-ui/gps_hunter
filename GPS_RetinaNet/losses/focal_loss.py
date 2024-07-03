"""
This module defines the Focal Loss for handling class imbalance in object detection tasks.

:module: focal_loss
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    :param gamma: float, focusing parameter for modulating factor (1-pt)^gamma.
    :param alpha: float, balancing factor for class imbalance.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        """
        Forward pass of the Focal Loss.

        :param inputs: torch.Tensor, predicted class logits.
        :param targets: torch.Tensor, ground truth class labels.
        :return: torch.Tensor, computed Focal Loss.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
