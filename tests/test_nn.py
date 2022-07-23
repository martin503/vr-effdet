import torch
import random
import pytest
from src.nn import *


class TestDSConv2D:
    def test_channels(self):
        with pytest.raises(ValueError):
            x = torch.rand(1, 3, 3)
            conv = DSConv2D(1, 1, 1)
            conv(x)

        x = torch.rand(1, 1, 3, 3)
        conv = DSConv2D(1, 3, 1)
        assert conv(x).shape[1] == 3

        x = torch.rand(2, 5, 3, 3)
        conv = DSConv2D(5, 10, 1)
        assert conv(x).shape[1] == 10

    def test_kwargs(self):
        x = torch.rand(2, 1, 5, 5)

        conv = DSConv2D(1, 3, 3, padding=0, stride=1, dilation=2, bias=False)
        r = conv(x)
        assert r.shape[0] == 2
        assert r.shape[1] == 3
        assert r.shape[2] == 1
        assert r.shape[3] == 1

        conv = DSConv2D(1, 2, 5, padding=1, stride=2, dilation=1, bias=True)
        r = conv(x)
        assert r.shape[0] == 2
        assert r.shape[1] == 2
        assert r.shape[2] == 2
        assert r.shape[3] == 2

    def test_dimensions(self, device):
        for _ in range(100):
            batch_size = random.randint(1, 100)
            in_ch = random.randint(1, 100)
            out_ch = random.randint(1, 100)
            w = random.randint(1, 100)
            h = random.randint(1, 100)
            x = torch.rand(batch_size, in_ch, h, w).to(device)

            conv = DSConv2D(in_ch, out_ch, 1).to(device)
            r = conv(x)
            assert r.shape[0] == batch_size
            assert r.shape[1] == out_ch
            assert r.shape[2] == h
            assert r.shape[3] == w

            conv = DSConv2D(in_ch, out_ch, 3, padding=1, stride=1).to(device)
            r = conv(x)
            assert r.shape[0] == batch_size
            assert r.shape[1] == out_ch
            assert r.shape[2] == h
            assert r.shape[3] == w


class TestBiFPN:
    def test_channels(self, device):
        for _ in range(100):
            wh = [64, 32, 16, 8, 4]
            in_ch = [random.randint(1, 100) for _ in range(5)]
            feat_maps = [torch.rand(2, in_ch[i], wh[i], wh[i]).to(device) for i in (range(5))]
            conv = BiFPN(in_ch, 10, 13).to(device)
            r = conv(feat_maps)
            assert all([r[i].shape[1] == 13 for i in range(5)])

    def test_dimensions(self, device):
        wh = [256, 128, 64, 32, 16, 8]
        for j in range(2, 7):
            for _ in range(10):
                in_ch = [random.randint(1, 100) for _ in range(j)]
                feat_maps = [torch.rand(2, in_ch[i], wh[i], wh[i]).to(device) for i in (range(j))]
                conv = BiFPN(in_ch, 16, 32).to(device)
                r = conv(feat_maps)
                assert all([r[i].shape[0] == 2 for i in range(j)])
                assert all([r[i].shape[1] == 32 for i in range(j)])
                assert all([r[i].shape[2] == wh[i] for i in range(j)])
                assert all([r[i].shape[3] == wh[i] for i in range(j)])


class TestEffNet:
    def test_dimensions(self, device):
        for _ in range(5):
            batch_size = random.randint(1, 2)
            in_ch = 3
            x = torch.rand(batch_size, in_ch, 128, 128).to(device)

            blocks = [3, 4, 5, 6, 7]
            out_ch = [24, 40, 80, 112, 192]
            wh = [32, 16, 8, 8, 4]
            eff = EffNet(blocks).to(device)
            r = eff(x)
            assert len(r) == len(blocks)
            for i in range(len(r)):
                assert r[i].shape[0] == batch_size
                assert r[i].shape[1] == out_ch[i]
                assert r[i].shape[2] == wh[i]
                assert r[i].shape[3] == wh[i]

        for j in range(3):
            blocks = [2, 3, 4, 6, 8]
            out_ch = [16, 24, 40, 112, 320]
            wh = [64, 32, 16, 8, 4]
            eff = EffNet(blocks[j:]).to(device)
            r = eff(x)
            assert len(r) == len(blocks[j:])
            for i in range(len(r)):
                assert r[i].shape[0] == batch_size
                assert r[i].shape[1] == (out_ch[j:])[i]
                assert r[i].shape[2] == (wh[j:])[i]
                assert r[i].shape[3] == (wh[j:])[i]


class TestHead:
    def test_dimensions(self):
        sigmas = [nn.Identity, nn.SiLU, nn.Sigmoid]
        ch = [10, 20, 30]
        kernels = [1, 1]
        kwargs = [{}, {}]
        x = torch.rand(4, ch[0], 64, 64)

        conv = Head(sigmas, ch, kernels, kwargs)
        r = conv(x)
        assert r.shape[0] == 4
        assert r.shape[1] == 1
        assert r.shape[2] == 64
        assert r.shape[3] == 64

    def test_kwargs(self):
        sigmas = [nn.Identity, nn.SiLU, nn.ReLU6, nn.Sigmoid]
        ch = [10, 20, 30, 11]
        kernels = [1, 3, 2]
        kwargs = [{}, {'padding': 1}, {'stride': 2, 'padding': 0}]
        x = torch.rand(4, ch[0], 64, 64)

        conv = Head(sigmas, ch, kernels, kwargs)
        r = conv(x)
        assert r.shape[0] == 4
        assert r.shape[1] == 1
        assert r.shape[2] == 32
        assert r.shape[3] == 32


class TestEfficientDet:
    def test_dimensions(self, device):
        blocks = [2, 3, 4, 6, 8]

        in_ch = [16, 24, 40, 112, 320]
        b1_args = [in_ch, 5, 10]

        in_ch = [b1_args[-1]] * 5
        b2_args = [in_ch, 20, 3]

        sigmas = [nn.Identity, nn.SiLU, nn.Sigmoid]
        ch = [b2_args[-1], 20, 30]
        kernels = [1, 1]
        kwargs = [{}, {}]
        mask_args = [sigmas, ch, kernels, kwargs]

        sigmas = [nn.Identity, nn.SiLU, nn.ReLU6, nn.Sigmoid]
        ch = [b2_args[-1], 20, 30, 11]
        kernels = [1, 3, 2]
        kwargs = [{}, {'padding': 1}, {'stride': 2, 'padding': 0}]
        edge_args = [sigmas, ch, kernels, kwargs]

        x = torch.rand(2, 3, 128, 128).to(device)

        conv = EfficientDet(blocks, [b1_args, b2_args], mask_args, edge_args).to(device)
        r = conv(x)
        assert r[0].shape[0] == 2
        assert r[0].shape[1] == 1
        assert r[0].shape[2] == 64
        assert r[0].shape[3] == 64
        assert r[1].shape[0] == 2
        assert r[1].shape[1] == 1
        assert r[1].shape[2] == 32
        assert r[1].shape[3] == 32

        blocks = [2, 3, 4, 5, 7]

        in_ch = [16, 24, 40, 80, 192]
        b1_args = [in_ch, 5, 10]

        in_ch = [b1_args[-1]] * 5
        b2_args = [in_ch, 2, 6]

        sigmas = [nn.Identity, nn.SiLU, nn.Sigmoid]
        ch = [b2_args[-1], 13, 9]
        kernels = [1, 1]
        kwargs = [{}, {}]
        mask_args = [sigmas, ch, kernels, kwargs]

        sigmas = [nn.Sigmoid]
        ch = [b2_args[-1]]
        kernels = []
        kwargs = []
        edge_args = [sigmas, ch, kernels, kwargs]

        x = torch.rand(4, 3, 128, 128).to(device)

        conv = EfficientDet(blocks, [b1_args, b2_args], mask_args, edge_args).to(device)
        r = conv(x)
        assert r[0].shape[0] == 4
        assert r[0].shape[1] == 1
        assert r[0].shape[2] == 64
        assert r[0].shape[3] == 64
        assert r[1].shape[0] == 4
        assert r[1].shape[1] == 1
        assert r[1].shape[2] == 64
        assert r[1].shape[3] == 64
