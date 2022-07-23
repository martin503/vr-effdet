import torch
import torch.nn as nn
import torch.nn.functional as F
from src.helpers import discretize


class Loss:
    """
    Loss computer.
    """
    def __init__(self, mode='bce', **kwargs):
        self.mode = mode
        if mode == 'bce':
            self.f = F.binary_cross_entropy
        elif mode == 'focal':
            self.f = FocalLoss(**kwargs)
        else:
            raise NotImplementedError
    
    def loss(self, mask_logits, mask_targets, edge_logits, edge_targets, edge_ratio, iou_threshold, round_ratio):
        mask_loss = self.f(mask_logits, mask_targets)
        edge_loss = self.f(edge_logits, edge_targets)
        mask_round_loss = self.f(discretize(mask_logits, iou_threshold), mask_targets)
        edge_round_loss = self.f(discretize(edge_logits, iou_threshold), edge_targets)
        return mask_loss + edge_ratio * edge_loss + round_ratio * (mask_round_loss + edge_ratio * edge_round_loss), mask_loss, edge_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        loss = (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
