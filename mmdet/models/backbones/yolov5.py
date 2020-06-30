import math
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from ..builder import BACKBONES
# from .experimental import *
from ..utils.experimental import *
from mmdet.utils import get_root_logger

from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm


def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-4
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

@BACKBONES.register_module()
class YoloV5(nn.Module):
    def __init__(self,gd,gw ,dim_in=3):
        super(YoloV5, self).__init__()
        ch = lambda x: math.ceil(x*gw/8)*8
        dn = lambda x: max(round(x*gd), 1) if x > 1 else 1
        # self.backbone = nn.ModuleList([
        #     Focus(dim_in, ch(64), k=3),  # 1-P1/2
        #     Conv(ch(64), ch(128), 3, 2), # 2-p2/4
        #     Bottleneck(ch(128), ch(128)),
        #     Conv(ch(128), ch(256), 3, 2), # 3-p3/8
        #     BottleneckCSP(ch(256), ch(256), dn(9)),
        #     Conv(ch(256), ch(512), 3, 2), # 4-p4/16
        #     BottleneckCSP(ch(512), ch(512), dn(9)),
        #     Conv(ch(512), ch(1024), 3, 2), # 8-p5/32
        #     SPP(ch(1024), ch(1024), (5, 9, 13)),
        #     BottleneckCSP(ch(1024), ch(1024), dn(6)),
        #     ])
        self.backbone=nn.ModuleList(
            [
                Focus(dim_in, ch(64), k=3),  # 1-P1/2
                Conv(ch(64), ch(128), 3, 2), # 2-p2/4
                BottleneckCSP(ch(128), ch(128), dn(3)),
                Conv(ch(128), ch(256), 3, 2), # 3-p3/8
                BottleneckCSP(ch(256), ch(256), dn(9)),
                Conv(ch(256), ch(512), 3, 2), # 4-p4/16
                BottleneckCSP(ch(512), ch(512), dn(9)),
                Conv(ch(512), ch(1024), 3, 2), # 8-p5/32
                SPP(ch(1024), ch(1024), (5, 9, 13)),
            ]
        )

    
    


    def init_weights(self, pretrained=None):
        # pretrained = '/home/lishuang/Disk/gitlab/traincode/yolov5l_state_dict.pt'
        # pretrained must be from ultralytics/yolov5
        if isinstance(pretrained, str):
                pretrained_checkpoint=torch.load(pretrained)
                model_dict = self.state_dict()
                pretrained_dict = {k.replace('model.', 'backbone.'): v for k, v in pretrained_checkpoint.items() if k.replace('model.', 'backbone.') in model_dict}
                model_dict = self.state_dict()
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict)
                print("init_backbone_weights")
        # if isinstance(pretrained, str):
        #     logger = get_root_logger()
        #     load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
    
    def forward(self, x):
        out = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [6, 4]:
                out.append(x)
        out.append(x)

        return out


#  @registry.BACKBONES.register("YOLOV5")
#  def build_yolov5_backbone(cfg, dim_in=3):
#      body = YoloV5(cfg, dim_in)
#      model = nn.Sequential(OrderedDict([("body", body)]))
#      model.out_channels = cfg.MODEL.YOLOV5.BACKBONE_OUT_CHANNELS
#      return model

