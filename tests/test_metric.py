import torch
import random
from src.utils.metric import *


class TestIou:
    def test_sizes(self):
        for _ in range(1):
            batch = random.randint(1, 100)
            channel = random.randint(1, 100)
            h = random.randint(1, 100)
            w = random.randint(1, 100)
            pred = torch.randint(0, 1, (batch, channel, h, w))
            tar = torch.randint(0, 1, (batch, channel, h, w))
            Metric.iou(pred, tar)

    def test_iou_values(self):
        pred = torch.tensor([[[[0, 0, 1, 0, 0],
                             [0, 1, 0, 1, 0],
                             [1, 0, 1, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0]]]])
        tar = torch.tensor([[[[0, 0, 1, 0, 1],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]]]])
        iou = Metric.iou(pred, tar)
        assert iou == 0.5

        pred = torch.tensor([[[[1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0]]]])
        tar = torch.tensor([[[[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]]]])
        iou = Metric.iou(pred, tar, threshold=0.7)
        assert iou == 0.5
        iou = Metric.iou(pred, tar, threshold=0.5)
        assert iou == 1

        pred = torch.tensor([[[[1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0]]],
                            [[[1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0],
                             [1, 0.6, 0, 0, 0]]]])
        tar = torch.tensor([[[[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]]],
                           [[[1, 1, 0, 0, 0],
                              [1, 1, 0, 0, 0],
                              [1, 1, 0, 0, 0],
                              [1, 1, 0, 0, 0],
                              [1, 1, 0, 0, 0]]]])
        iou = Metric.iou(pred, tar, threshold=0.5)
        assert iou == 1
