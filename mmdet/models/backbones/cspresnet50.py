#cspresnet50-panet-spp
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.ops import build_plugin_layer
from mmdet.utils import get_root_logger
from ..builder import BACKBONES
# from ..utils import CSPResLayer
import torch

class CSPResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))


        super(CSPResLayer, self).__init__(*layers)

class CSPBottleneck(nn.Module):
    expansion = 2  #从残差网络的4改为2，维度减半，分为两个部分
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(CSPBottleneck, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)
        # inplanes kernel_size x kernel_size x planes  64 1x1x64
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)


            if self.downsample is not None:
                identity = self.downsample(x)  #shortcut ?

            out += identity  #shortcut

            return out

        out = _inner_forward(x)

        out = self.relu(out)

        return out

class CSPBigBottleneck(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(CSPBigBottleneck, self).__init__()
        block = CSPBottleneck
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1_stride = 1
        self.conv2_stride = stride
        self.conv3_stride = stride
        self.conv4_stride = stride

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, inplanes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes * 2, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes * 2, postfix=3)
        self.norm4_name, norm4 = build_norm_layer(norm_cfg, planes * 2, postfix=4)

        self.conv1 = build_conv_layer(
            conv_cfg,
            planes,
            inplanes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            conv_cfg,
            planes * 2,
            planes * 2,
            kernel_size=1,
            stride=self.conv2_stride,
            # padding=dilation,
            # dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes*2,
            kernel_size=1,
            stride=self.conv3_stride,
            # padding=dilation,
            # dilation=dilation,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.conv4 = build_conv_layer(
            conv_cfg,
            planes * 4,
            planes * 2,
            kernel_size=1,
            stride=self.conv4_stride,
            # padding=dilation,
            # dilation=dilation,
            bias=False)
        self.add_module(self.norm4_name, norm4)

        self.res_layer = self.make_cspres_layer(
            block=block,
            inplanes=inplanes,
            planes=planes,
            num_blocks=num_blocks,
            stride=stride,
            dilation=dilation,
            norm_cfg=norm_cfg)

        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.downsample = downsample

    def make_cspres_layer(self, **kwargs):
        return CSPResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    @property
    def norm4(self):
        return getattr(self, self.norm4_name)

    def forward(self, x):
        if self.downsample is not None:
            out = self.downsample(x)
            out = self.relu(out)
        else:
            out=x

        identity = out

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)

        out=self.res_layer(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        identity = self.conv3(identity)
        identity = self.norm3(identity)
        identity = self.relu(identity)


        out=torch.cat([out,identity], 1)

        out = self.conv4(out)
        out = self.norm4(out)
        out = self.relu(out)

        return out

@BACKBONES.register_module()
class CSPResNet(nn.Module):
    arch_settings = {
        50: (CSPBottleneck, (3, 3, 5, 2))
    }

    def __init__(self,
                 depth=50,
                 strides=1,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 conv_cfg=None,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 zero_init_residual=True):
        super(CSPResNet, self).__init__()
        #norm_cfg = dict(type='BN', eps=1e-04,momentum=0.03,requires_grad=True)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.strides = strides
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        assert max(out_indices) < num_stages

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride=self.strides
            planes = base_channels * 2 ** i
            downsample = None
            if self.inplanes != planes:
                downsample = []
                conv_stride = 2
                downsample.extend([
                    build_conv_layer(
                        conv_cfg,
                        planes,
                        planes,
                        kernel_size=3,
                        stride=conv_stride,
                        padding=1,
                        bias=False),
                    build_norm_layer(norm_cfg, planes)[1]
                ])
                downsample = nn.Sequential(*downsample)
            res_layer = CSPBigBottleneck(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                num_blocks=num_blocks,
                downsample=downsample,
                norm_cfg=norm_cfg)
            layer_name = f'layer{i + 1}'
            self.inplanes = planes * 4
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self,x):

        x = self.conv1(x)

        x = self.norm1(x)
        x = self.relu(x)

        x = self.maxpool(x)


        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)



            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, CSPBottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, CSPBottleneck):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    # def train(self, mode=True):
    #     super(CSPResNet, self).train(mode)
    #     self._freeze_stages()
    #     if mode and self.norm_eval:
    #         for m in self.modules():
    #             # trick: eval have effect on BatchNorm only
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()


if __name__ == '__main__':
    torch.manual_seed(1)
    model=CSPResNet()
    print(model)
    # model.load_state_dict(torch.load('../../../checkpoints/csresnet50.pt'))  #need to check the module one by one
    model.eval()
    # print(model(torch.ones(1, 3, 256, 256)))

    print(model(torch.ones(1, 3, 256, 256)))
