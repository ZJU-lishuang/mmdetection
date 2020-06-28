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

class DetectionFPNNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DetectionFPNNeck, self).__init__()
        self.convdown = ConvLayer(in_channels, out_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.detect = DetectionNeck(in_channels, out_channels)

    def forward(self,x):
        input, output=x
        tmp = self.convdown(input)
        tmp = self.upsample(tmp)
        output = torch.cat([output, tmp], 1)
        output = self.detect(output)

        return output

class DetectionPAFPNNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DetectionPAFPNNeck, self).__init__()
        self.convup = ConvLayer(in_channels, out_channels, 3, 2)
        self.detectout = DetectionNeck(int(out_channels*2), out_channels)

    def forward(self,x):
        input, output=x
        tmp = self.convup(input)
        output = torch.cat([output, tmp], 1)
        output = self.detectout(output)

        return output

@NECKS.register_module
class PANETSPPNeck(nn.Module):  #the name YoloNeck is wrong ,need to modify ,like cspneck

    """The tail side of the YoloNet.
    It will take the result from DarkNet53BackBone and do some upsampling and concatenation.
    It will finally output the detection result.
    Assembling YoloNetTail and DarkNet53BackBone will give you final result"""

    def __init__(self,in_channels=[256,512,1024],out_channels=[128,256,512],num_outs=3,start_level=0,
                 end_level=-1,SPP=True):
        super(PANETSPPNeck, self).__init__()
        self.num_ins = len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.fpn_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level-1):
            l_conv=ConvLayer(in_channels[i], out_channels[i], 1)
            fpn_conv=DetectionFPNNeck(in_channels[i], out_channels[i])
            pafpn_conv = DetectionPAFPNNeck(out_channels[i], in_channels[i])
            self.fpn_convs.append(fpn_conv)
            self.pafpn_convs.append(pafpn_conv)
            self.lateral_convs.append(l_conv)
        #SPP
        self.SPP = SPP
        if self.SPP==True:
            self.lateral_convs.append(DetectionSPPNeck(in_channels[self.backbone_end_level-1], out_channels[self.backbone_end_level-1]))
        else:
            self.lateral_convs.append(DetectionNeck(in_channels[self.backbone_end_level-1], out_channels[self.backbone_end_level-1]))


    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

         # part 2: add bottom-up path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i-1]=self.fpn_convs[i-1]((laterals[i],laterals[i-1]))

        outs = []
        outs.append(laterals[0])
        outs.extend([
            pafpn_conv((laterals[i], laterals[i + 1]))
            for i, pafpn_conv in enumerate(self.pafpn_convs)
        ])

        return outs[::-1]



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
