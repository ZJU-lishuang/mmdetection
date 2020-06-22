# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..registry import NECKS
from ..builder import NECKS

# from mmdet.models.utils import ConvLayer

from mmcv.cnn import xavier_init
from mmcv.runner import load_checkpoint


class ConvLayer(nn.Module):
    """Basic 'conv' layer, including:
     A Conv2D layer with desired channels and kernel size,
     A batch-norm layer,
     and A leakyReLu layer with neg_slope of 0.1.
     (Didn't find too much resource what neg_slope really is.
     By looking at the darknet source code, it is confirmed the neg_slope=0.1.
     Ref: https://github.com/pjreddie/darknet/blob/master/src/activations.h)
     Please note here we distinguish between Conv2D layer and Conv layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, lrelu_neg_slope=0.1):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_neg_slope)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)

        return out


class DetectionNeck(nn.Module):
    """The DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
    1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n,
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)
    out_channels = n
    """

    def __init__(self, in_channels, out_channels):
        super(DetectionNeck, self).__init__()
        # assert double_out_channels % 2 == 0  #assert out_channels is an even number
        # out_channels = double_out_channels // 2
        double_out_channels = out_channels * 2
        self.conv1 = ConvLayer(in_channels, out_channels, 1)
        self.conv2 = ConvLayer(out_channels, double_out_channels, 3)
        self.conv3 = ConvLayer(double_out_channels, out_channels, 1)
        self.conv4 = ConvLayer(out_channels, double_out_channels, 3)
        self.conv5 = ConvLayer(double_out_channels, out_channels, 1)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        out = self.conv5(tmp)
        return out

class DetectionSPPNeck(nn.Module):
    """The DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
    1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n,
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)
    out_channels = n
    """

    def __init__(self, in_channels, out_channels):
        super(DetectionSPPNeck, self).__init__()
        # assert double_out_channels % 2 == 0  #assert out_channels is an even number
        # out_channels = double_out_channels // 2
        double_out_channels = out_channels * 2
        self.conv1 = ConvLayer(in_channels, out_channels, 1)
        self.conv2 = ConvLayer(out_channels, double_out_channels, 3)
        self.conv3 = ConvLayer(double_out_channels, out_channels, 1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.convspp=ConvLayer(out_channels*4,out_channels,1)

        self.conv4 = ConvLayer(out_channels, double_out_channels, 3)
        self.conv5 = ConvLayer(double_out_channels, out_channels, 1)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp1=self.maxpool1(tmp)
        tmp2=self.maxpool2(tmp)
        tmp3=self.maxpool3(tmp)
        tmp=torch.cat([tmp,tmp1,tmp2,tmp3],1)
        tmp=self.convspp(tmp)

        tmp = self.conv4(tmp)
        out = self.conv5(tmp)
        return out


@NECKS.register_module
class YoloNeck(nn.Module):  #the name YoloNeck is wrong ,need to modify ,like cspneck

    """The tail side of the YoloNet.
    It will take the result from DarkNet53BackBone and do some upsampling and concatenation.
    It will finally output the detection result.
    Assembling YoloNetTail and DarkNet53BackBone will give you final result"""

    def __init__(self):
        super(YoloNeck, self).__init__()

        self.conv1 = ConvLayer(256, 128, 1)
        self.conv2 = ConvLayer(512, 256, 1)
        self.conv3 = ConvLayer(1024, 512, 1)
        self.convdown3=ConvLayer(512, 256, 1)
        self.upsample3=nn.Upsample(scale_factor=2)
        self.detect2=DetectionNeck(512, 256)
        self.convdown2=ConvLayer(256, 128, 1)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.detectout1=DetectionNeck(256, 128)
        self.convup1=ConvLayer(128, 256, 3,2)
        self.detectout2 = DetectionNeck(512, 256)
        self.convup2=ConvLayer(256, 512, 3,2)
        self.detectout3 = DetectionNeck(1024, 512)
        self.detectSPP=DetectionSPPNeck(1024, 512)

    def forward(self, x):
        assert len(x) == 3
        x1, x2, x3 = x
        out1 = self.conv1(x1)
        out2 = self.conv2(x2)
        # out3 = self.conv3(x3)
        out3=self.detectSPP(x3)

        tmp3=self.convdown3(out3)
        tmp3=self.upsample3(tmp3)
        out2=torch.cat([out2,tmp3], 1)
        out2=self.detect2(out2)

        tmp2=self.convdown2(out2)
        tmp2=self.upsample2(tmp2)
        out1 = torch.cat([out1, tmp2], 1)
        out1 = self.detectout1(out1)

        tmp1 = self.convup1(out1)
        out2= torch.cat([out2, tmp1], 1)
        out2 = self.detectout2(out2)

        tmp2 = self.convup2(out2)
        out3 = torch.cat([out3, tmp2], 1)
        out3 = self.detectout3(out3)

        return out3, out2, out1

        # return out1,out2,out3

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        else:
            raise TypeError('pretrained must be a str or None')
