import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init, normal_init, xavier_init
from mmcv.runner import load_checkpoint

from mmdet.utils import get_root_logger
from ..builder import BACKBONES

from collections import OrderedDict
import math
from torch.nn import init as init

class conv_bn_relu(nn.Module):
    """docstring for conv_bn_relu"""

    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        out = self.norm(self.conv(x))
        if self.activation:
            out = F.relu(out, inplace=True)
        return out
    
    
class conv_relu(nn.Module):
    """docstring for conv_relu"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              bias=False, **kwargs)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        return out
    
    
    
class _DenseLayer(nn.Module):
    """docstring for _DenseLayer"""

    def __init__(self, num_input_features, growth_rate, bottleneck_width, drop_rate):
        super(_DenseLayer, self).__init__()
        growth_rate = growth_rate // 2
        inter_channel = int(growth_rate * bottleneck_width / 4) * 4

        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            print('adjust inter_channel to ', inter_channel)

        self.branch1a = conv_bn_relu(
            num_input_features, inter_channel, kernel_size=1)
        self.branch1b = conv_bn_relu(
            inter_channel, growth_rate, kernel_size=3, padding=1)

        self.branch2a = conv_bn_relu(
            num_input_features, inter_channel, kernel_size=1)
        self.branch2b = conv_bn_relu(
            inter_channel, growth_rate, kernel_size=3, padding=1)
        self.branch2c = conv_bn_relu(
            growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = self.branch1a(x)
        out1 = self.branch1b(out1)

        out2 = self.branch2a(x)
        out2 = self.branch2b(out2)
        out3 = self.branch2c(out2)

        out = torch.cat([x, out1, out2], dim=1)
        return out
    
    
class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _StemBlock(nn.Module):

    def __init__(self, num_input_channels, num_init_features):
        super(_StemBlock, self).__init__()

        num_stem_features = int(num_init_features / 2)

        self.stem1 = conv_bn_relu(
            num_input_channels, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem2a = conv_bn_relu(
            num_init_features, num_stem_features, kernel_size=1, stride=1, padding=0)
        self.stem2b = conv_bn_relu(
            num_stem_features, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem3 = conv_bn_relu(
            2 * num_init_features, num_init_features, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], dim=1)
        out = self.stem3(out)

        return out


class ResBlock(nn.Module):
    """docstring for ResBlock"""

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.res1a = conv_relu(in_channels, 128, kernel_size=1)
        self.res1b = conv_relu(128, 128, kernel_size=3, padding=1)
        self.res1c = conv_relu(128, 256, kernel_size=1)

        self.res2a = conv_relu(in_channels, 256, kernel_size=1)

    def forward(self, x):
        out1 = self.res1a(x)
        out1 = self.res1b(out1)
        out1 = self.res1c(out1)

        out2 = self.res2a(x)
        out = out1 + out2
        return out
    
    
def add_extras(i, batch_norm=False):
    layers = []
    in_channels = i
    channels = [128, 256, 128, 256, 128, 256]
    stride = [1, 2, 1, 1, 1, 1]
    padding = [0, 1, 0, 0, 0, 0]

    for k, v in enumerate(channels):
        if k % 2 == 0:
            if batch_norm:
                layers += [conv_bn_relu(in_channels, v,
                                        kernel_size=1, padding=padding[k])]
            else:
                layers += [conv_relu(in_channels, v,
                                     kernel_size=1, padding=padding[k])]
        else:
            if batch_norm:
                layers += [conv_bn_relu(in_channels, v,
                                        kernel_size=3, stride=stride[k], padding=padding[k])]
            else:
                layers += [conv_relu(in_channels, v,
                                     kernel_size=3, stride=stride[k], padding=padding[k])]
        in_channels = v

    return layers


def add_resblock(nchannels):
    layers = []
    for k, v in enumerate(nchannels):
        layers += [ResBlock(v)]
    return layers


# python tools/train.py configs/pascal_voc/Pelee_voc0712.py


@BACKBONES.register_module()
class SSDPeleenet(nn.Module):
    """VGG Backbone network for single-shot-detection

    Args:
        input_size (int): width and height of input, from {304,512}.
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        out_indices (Sequence[int]): Output from which stages.

    Example:
        >>> self = SSDVGG(input_size=304, depth=11)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 304, 304)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 512, 19, 19)
        (1, 704, 10, 10)
        (1, 256, 5, 5)
        (1, 256, 3, 3)
        (1, 256, 1, 1)
    """

    def __init__(self, input_size,
                        growth_rate,
                        block_config,
                        num_init_features,
                        bottleneck_width,
                        drop_rate):
        # TODO: in_channels for mmcv.VGG
        super(SSDPeleenet, self).__init__()
        assert input_size in (300, 304, 512)
        
        self.size = input_size
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = num_init_features
        self.bottleneck_width = bottleneck_width
        self.drop_rate = drop_rate

        self.features = nn.Sequential(OrderedDict([
            ('stemblock', _StemBlock(3, self.num_init_features)),
        ]))

        if type(self.growth_rate) is list:
            growth_rates = self.growth_rate
            assert len(
                growth_rates) == 4, 'The growth rate must be the list and the size must be 4'
        else:
            growth_rates = [self.growth_rate] * 4

        if type(self.bottleneck_width) is list:
            bottleneck_widths = self.bottleneck_width
            assert len(
                bottleneck_widths) == 4, 'The bottleneck width must be the list and the size must be 4'
        else:
            bottleneck_widths = [self.bottleneck_width] * 4

        # Each denseblock
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bottleneck_widths[i], growth_rate=growth_rates[i], drop_rate=self.drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rates[i]

            self.features.add_module('transition%d' % (i + 1), conv_bn_relu(
                num_features, num_features, kernel_size=1, stride=1, padding=0))

            if i != len(self.block_config) - 1:
                self.features.add_module('transition%d_pool' % (
                    i + 1), nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))
                num_features = num_features
                
        extras = add_extras(704, batch_norm=True)
        self.extras = nn.ModuleList(extras)

        nchannels = [512, 704, 256, 256, 256]

        resblock = add_resblock(nchannels)
        self.resblock = nn.ModuleList(resblock)

            
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
                    
            for m in self.extras.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
                
            for m in self.resblock.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

        
 

    def forward(self, x):
        outs = []
        for k, feat in enumerate(self.features):
            x = feat(x)
            if k == 8 or k == len(self.features) - 1:
                outs.append(x)

        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 2 == 1:
                outs.append(x)

        for k, x in enumerate(outs):
            #print(k, ': ' , x.size())
            outs[k] = self.resblock[k](x)
            #print('resblock: ' , outs[k].size())
        
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)
