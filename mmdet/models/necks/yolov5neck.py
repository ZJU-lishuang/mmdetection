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
from ..utils.experimental import *

@NECKS.register_module
class Yolov5Neck(nn.Module):  #the name YoloNeck is wrong ,need to modify ,like cspneck

    """The tail side of the YoloNet.
    It will take the result from DarkNet53BackBone and do some upsampling and concatenation.
    It will finally output the detection result.
    Assembling YoloNetTail and DarkNet53BackBone will give you final result"""

    def __init__(self,gd=1,gw=1):
        super(Yolov5Neck, self).__init__()
        ch = lambda x: math.ceil(x * gw / 8) * 8
        dn = lambda x: max(round(x * gd), 1) if x > 1 else 1
        self.layer_9 = BottleneckCSP(ch(1024), ch(1024), dn(3))
        self.layer_10 =Conv(ch(1024), ch(512), 1, 1)
        self.layer_11 =nn.Upsample(None, 2, 'nearest')
        self.layer_12 = None
        self.layer_13 =BottleneckCSP(ch(1024), ch(512), dn(3))

        self.layer_14 =Conv(ch(512), ch(256), 1, 1)
        self.layer_15 =nn.Upsample(None, 2, 'nearest')
        self.layer_16 =None
        self.layer_17 =BottleneckCSP(ch(512), ch(256), dn(3))
        self.layer_18=None

        self.layer_19 = Conv(ch(256), ch(256),3, 2)
        self.layer_20 = None
        self.layer_21 = BottleneckCSP(ch(512), ch(512), dn(3))
        self.layer_22 = None

        self.layer_23 = Conv(ch(512), ch(512), 3, 2)
        self.layer_24 = None
        self.layer_25 = BottleneckCSP(ch(1024), ch(1024), dn(3))
        self.layer_26 = None


    def forward(self, x):
        assert len(x) == 3
        x1, x2, x3 = x

        out=self.layer_9(x3)
        out3=self.layer_10(out)
        out = self.layer_11(out3)
        out=torch.cat([out,x2],1)
        out=self.layer_13(out)

        out2=self.layer_14(out)
        out = self.layer_15(out2)
        out = torch.cat([out, x1],1)
        out1 = self.layer_17(out)  #small

        out = self.layer_19(out1)
        out=torch.cat([out, out2],1)
        out2 = self.layer_21(out)

        out = self.layer_23(out2)
        out = torch.cat([out, out3],1)
        out3 = self.layer_25(out)

        return out3, out2, out1



    def init_weights(self, pretrained=None):
        # pretrained = '/home/lishuang/Disk/gitlab/traincode/yolov5l_state_dict.pt'
        #pretrained must be from ultralytics/yolov5
        if isinstance(pretrained, str):
            pretrained_checkpoint = torch.load(pretrained)
            model_dict = self.state_dict()
            pretrained_dict = {k.replace('model.', 'layer_'): v for k, v in pretrained_checkpoint.items() if
                               k.replace('model.', 'layer_') in model_dict}
            model_dict = self.state_dict()
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print("init_neck_weights")

        # if isinstance(pretrained, str):
        #     logger = logging.getLogger()
        #     load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        else:
            raise TypeError('pretrained must be a str or None')
