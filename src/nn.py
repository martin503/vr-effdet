import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, sigma, in_ch, out_ch, kernel_size, **kwargs):
        """Simple convolution with BatchNorm and activation

        Args:
            sigma (torch.nn.Module): activation
            in_ch (int): in channels
            out_ch (int): out channels
            kernel_size (int): kernel_size
        """
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, **kwargs)
        self.norm = nn.BatchNorm2d(out_ch)
        self.sigma = sigma()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.sigma(x)
        return x


class DSConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, **kwargs):
        """Depthwise Separable Convolution

        Args:
            in_ch (int): in channels
            out_ch (int): out channels
            kernel_size (Tuple(int)): kernel size of depthwise convolution
        """
        super(DSConv2D, self).__init__()
        assert 'groups' not in kwargs.keys()
        self.depthwise_conv = nn.Conv2d(in_ch, in_ch, kernel_size, groups=in_ch, **kwargs)
        self.depth_feature_extractor = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        self.norm = nn.BatchNorm2d(out_ch)
        self.sigma = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depth_feature_extractor(x)
        x = self.norm(x)
        x = self.sigma(x)
        return x


class BiFPN(nn.Module):
    def __init__(self, in_ch, p_ch, out_ch):
        """Bi-directional Feature Pyramid Network
        Implemented with fast normalized fusion per-feature.

        Args:
            in_ch (List[int]): in channels with respect to input feature maps in forward
            p_ch (int): fixed number of channels that all feature maps share after first convolution
            out_ch (int): fixed number of channels that all feature maps share after forward pass
        """
        assert len(in_ch) > 1
        super(BiFPN, self).__init__()
        self.__in_ch = in_ch
        self.eps = 1e-4
        td = torch.empty(len(in_ch), 2)
        nn.init.xavier_normal_(td, gain=nn.init.calculate_gain('relu'))
        self.td = nn.Parameter(td, requires_grad=True)
        out = torch.empty(len(in_ch), 3)
        nn.init.xavier_normal_(out, gain=nn.init.calculate_gain('relu'))
        self.out = nn.Parameter(out, requires_grad=True)

        for i, ch in enumerate(in_ch):
            setattr(self, 'P' + str(i), Conv2D(nn.ReLU, ch, p_ch, 1, stride=1, padding=0))
            if i != 0 and i != len(in_ch) - 1:
                setattr(self, 'Ptd' + str(i), DSConv2D(p_ch, p_ch, 3, stride=1, padding=1))
            if i != 0:
                setattr(self, 'conv' + str(i), DSConv2D(out_ch, p_ch, 2, stride=2, padding=0))
            setattr(self, 'Pout' + str(i), DSConv2D(p_ch, out_ch, 3, stride=1, padding=1))

    def forward(self, feat_maps):
        assert all([feat_maps[i].shape[2] == 2 * feat_maps[i + 1].shape[2] and
                    feat_maps[i].shape[3] == 2 * feat_maps[i + 1].shape[3] and
                    feat_maps[i].shape[0] == feat_maps[i + 1].shape[0] for i in range(len(feat_maps) - 1)])
        assert all(self.__in_ch[i] == feat_maps[i].shape[1] for i in range(len(feat_maps)))

        # P
        for i in range(len(feat_maps)):
            feat_maps[i] = getattr(self, 'P' + str(i))(feat_maps[i])

        # TD
        feat_maps_td = [None for _ in range(len(feat_maps))]
        for i in range(len(feat_maps) - 1, -1, -1):
            if i == 0 or i == len(feat_maps) - 1:
                feat_maps_td[i] = feat_maps[i]
            else:
                params = F.relu(self.td[i])
                feat_maps_td[i] = (params[0] * self.__upsample(feat_maps_td[i + 1]) +
                                   params[1] * feat_maps[i]) / (sum(params) + self.eps)
                feat_maps_td[i] = getattr(self, 'Ptd' + str(i))(feat_maps_td[i])

        # OUT
        feat_maps_out = [None for _ in range(len(feat_maps))]
        for i in range(len(feat_maps)):
            params = F.relu(self.out[i])
            if i == 0:
                feat_maps_out[i] = (params[0] * self.__upsample(feat_maps_td[i + 1]) +
                                    params[1] * feat_maps[i]) / (sum(params[:2]) + self.eps)
            else:
                conv = getattr(self, 'conv' + str(i))
                if i == len(feat_maps) - 1:
                    feat_maps_out[i] = (params[0] * conv(feat_maps_out[i - 1]) +
                                        params[1] * feat_maps[i]) / (sum(params[:2]) + self.eps)
                else:
                    feat_maps_out[i] = (params[0] * conv(feat_maps_out[i - 1]) +
                                        params[1] * feat_maps[i] +
                                        params[2] * feat_maps_td[i]) / (sum(params) + self.eps)
            feat_maps_out[i] = getattr(self, 'Pout' + str(i))(feat_maps_out[i])

        return feat_maps_out

    def __upsample(self, feat_map):
        return F.interpolate(feat_map, scale_factor=2, mode='nearest')


class EffNet(nn.Module):
    def __init__(self, blocks):
        """EfficientNet-b0 with pretrained weights

        Args:
            blocks (List[int]): ids of effnet layers which feature maps will be computed in forward
        """
        super(EffNet, self).__init__()
        eff_b0 = torchvision.models.efficientnet_b0(pretrained=True)
        self.__num_blocks = len(blocks)
        blocks = [0] + blocks
        for i in range(self.__num_blocks):
            setattr(self, 'B' + str(i), nn.Sequential(*list(eff_b0.features.children())[blocks[i]:blocks[i + 1]]))

    def forward(self, imgs):
        feat_maps = [None for _ in range(self.__num_blocks + 1)]
        feat_maps[0] = imgs
        for i in range(self.__num_blocks):
            feat_maps[i + 1] = getattr(self, 'B' + str(i))(feat_maps[i])
        return feat_maps[1:]


class Head(nn.Module):
    def __init__(self, sigmas, channels, kernels, kwargs):
        """Head for final predictions

        Args:
            sigmas (torch.nn.Module): activation for each additional extractor
            channels (List[int]): channels of additional extractors
            kernels (List[int]): kernels of additional extractors
            kwargs (List[Dict]): kwargs of additional extractors
        """
        super(Head, self).__init__()
        assert (len(channels) > 0 and len(kernels) == len(kwargs) and
                len(channels) == len(sigmas) and len(channels) == len(kernels) + 1)
        self.__num_convs = len(kernels)
        for i in range(len(kernels)):
            setattr(self, 'E' + str(i), Conv2D(sigmas[i], channels[i], channels[i + 1], kernels[i], **kwargs[i]))
        self.final_extractor = nn.Conv2d(channels[-1], 1, 1, stride=1, padding=0)
        self.final_activation = sigmas[-1]()

    def forward(self, feat_map):
        """forward pass

        Args:
            feat_map (torch.Tensor): final, biggest BiFPN feature map

        Returns:
            torch.Tensor: predictions of segmentation
        """
        for i in range(self.__num_convs):
            feat_map = getattr(self, 'E' + str(i))(feat_map)
        pred = self.final_extractor(feat_map)
        pred = self.final_activation(pred)
        return pred


class EfficientDet(nn.Module):
    def __init__(self, blocks, bifpns_args, mask_args, edge_args):
        """EfficientDet for edge aware semantic segmentation

        Args:
            blocks (List[int]): ids of effnet layers which feature maps will be computed in EffNet
            bifpns_args (List): arguments for BiFPNs
            mask_args (List): arguments for mask Head
            edge_args (List): arguments for edge Head
        """
        assert len(bifpns_args) > 0
        super(EfficientDet, self).__init__()
        self.backbone = EffNet(blocks)
        self.__num_bifpns = len(bifpns_args)
        for i in range(len(bifpns_args)):
            setattr(self, 'BiFPN' + str(i), BiFPN(*bifpns_args[i]))
        self.mask_head = Head(*mask_args)
        self.edge_head = Head(*edge_args)

    def forward(self, imgs):
        feat_maps = self.backbone(imgs)
        for i in range(self.__num_bifpns):
            feat_maps = getattr(self, 'BiFPN' + str(i))(feat_maps)
        mask_pred = self.mask_head(feat_maps[0])
        edge_pred = self.edge_head(feat_maps[0])
        return mask_pred, edge_pred

    def get_non_backbone_params(self):
        def make_gen():
            for i in range(self.__num_bifpns):
                for p in getattr(self, 'BiFPN' + str(i)).parameters():
                    yield p
            for p in self.mask_head.parameters():
                yield p
            for p in self.edge_head.parameters():
                yield p
        return make_gen()

    def get_backbone_params(self):
        return self.backbone.parameters()
