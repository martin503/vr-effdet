import torch
from torchmetrics.functional import jaccard_index
from src.helpers import discretize


class Metric:
    """
    Metrics computer.
    """
    @staticmethod
    def iou(preds, targets, threshold=0.5):
        assert preds.shape == targets.shape and len(preds.shape) == 4
        running = 0.0
        for pred, target in zip(preds, targets):
            pred = discretize(pred, threshold)
            pred = pred.type(torch.bool)
            target = target.type(torch.bool)
            numerator = torch.sum(pred & target)
            denominator = torch.sum(pred | target)
            running += numerator / denominator
        return running / preds.shape[0]

    @staticmethod
    def jaccard_index(target, pred, threshold=0.5):
        assert len(torch.unique(pred)[0]) == 2
        return jaccard_index(pred, target.type(torch.int), 2, threshold=threshold)
