import torch
from losses.focal_loss import FocalLoss
from losses.gps_loss import GPSLoss

def loss_fn(cls_logits, box_regression, gps_regression, cls_targets, box_targets, gps_targets):
    """
    Computes the total loss combining classification, bounding box, and GPS regression losses.

    :param cls_logits: list of torch.Tensor, predicted class logits from different FPN levels.
    :param box_regression: list of torch.Tensor, predicted bounding box regressions from different FPN levels.
    :param gps_regression: list of torch.Tensor, predicted GPS regressions from different FPN levels.
    :param cls_targets: torch.Tensor, ground truth class labels.
    :param box_targets: torch.Tensor, ground truth bounding boxes.
    :param gps_targets: torch.Tensor, ground truth GPS coordinates.
    :return: torch.Tensor, total loss.
    """
    # Concatenate outputs from all FPN levels
    cls_logits = torch.cat([logit.permute(0, 2, 3, 1).reshape(logit.size(0), -1) for logit in cls_logits], dim=1)
    box_regression = torch.cat([reg.permute(0, 2, 3, 1).reshape(reg.size(0), -1) for reg in box_regression], dim=1)
    gps_regression = torch.cat([reg.permute(0, 2, 3, 1).reshape(reg.size(0), -1) for reg in gps_regression], dim=1)

    # Flatten the targets
    cls_targets = cls_targets.view(cls_targets.size(0), -1)
    box_targets = box_targets.view(box_targets.size(0), -1)
    gps_targets = gps_targets.view(gps_targets.size(0), -1)

    # Ensure the target sizes match the input sizes
    cls_targets = cls_targets[:, :cls_logits.size(1)]
    box_targets = box_targets[:, :box_regression.size(1)]
    gps_targets = gps_targets[:, :gps_regression.size(1)]

    cls_loss = FocalLoss()(cls_logits, cls_targets)
    box_loss = torch.nn.functional.smooth_l1_loss(box_regression, box_targets)
    gps_loss = GPSLoss()(gps_regression, gps_targets)
    return cls_loss + box_loss + gps_loss

def train_model(model, dataloader, optimizer, device):
    """
    Trains the GPS-RetinaNet model for one epoch.

    :param model: nn.Module, the GPS-RetinaNet model.
    :param dataloader: DataLoader, data loader for the dataset.
    :param optimizer: torch.optim.Optimizer, optimizer for training.
    :param device: torch.device, device to train the model on.
    """
    model.train()
    for images, targets in dataloader:
        images = images.to(device)
        cls_targets = targets['labels'].to(device)
        box_targets = targets['boxes'].to(device)
        gps_targets = targets['gps'].to(device)
        
        optimizer.zero_grad()
        cls_logits, box_regression, gps_regression = model(images)
        loss = loss_fn(cls_logits, box_regression, gps_regression, cls_targets, box_targets, gps_targets)
        loss.backward()
        optimizer.step()
