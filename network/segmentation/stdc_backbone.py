"""
Implementation of paper: "Rethinking BiSeNet For Real-time Semantic Segmentation", https://arxiv.org/abs/2104.13188
Based on original implementation: https://github.com/MichaelFan01/STDC-Seg, cloned 23/08/2021, commit 59ff37f
"""
"""Backbone for PP-LiteSeg"""

import torch
import torch.nn as nn
from typing import Union, List, Tuple
from abc import ABC, abstractmethod

from torch import nn as nn


PRETRAINED = {'stdc1': './weights/stdc1#imagenet.pth',
              'stdc2': './weights/stdc2#imagenet.pth'}


class ConvBNReLU(nn.Module):
    """
    Class for Convolution2d-Batchnorm2d-Relu layer. Default behaviour is Conv-BN-Relu. To exclude Batchnorm module use
        `use_normalization=False`, to exclude Relu activation use `use_activation=False`.
    For convolution arguments documentation see `nn.Conv2d`.
    For batchnorm arguments documentation see `nn.BatchNorm2d`.
    For relu arguments documentation see `nn.Relu`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 use_normalization: bool = True,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None,
                 use_activation: bool = True,
                 inplace: bool = False):

        super(ConvBNReLU, self).__init__()
        self.seq = nn.Sequential()
        self.seq.add_module("conv", nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=groups,
                                              bias=bias,
                                              padding_mode=padding_mode))

        if use_normalization:
            self.seq.add_module("bn", nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                                                     track_running_stats=track_running_stats, device=device,
                                                     dtype=dtype))
        if use_activation:
            self.seq.add_module("relu", nn.ReLU(inplace=inplace))

    def forward(self, x):
        return self.seq(x)



class STDCBlock(nn.Module):
    """
    STDC building block, known as Short Term Dense Concatenate module.
    In STDC module, the kernel size of first block is 1, and the rest of them are simply set as 3.
    Args:
        steps (int): The total number of convs in this module, 1 conv 1x1 and (steps - 1) conv3x3.
    """
    def __init__(self, in_channels: int, out_channels: int, steps: int, stride: int = 1):
        super(STDCBlock, self).__init__()
        assert steps in [2, 3, 4], f"only 2, 3, 4 steps number are supported, found: {steps}"

        self.stride = stride

        self.conv_list = nn.ModuleList()
        # build first step conv 1x1.
        self.conv_list.append(ConvBNReLU(in_channels, out_channels // 2, kernel_size=1, bias=False))
        # avg pool in skip if stride = 2.
        self.skip_step1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1) if stride == 2 else nn.Identity()

        in_channels = out_channels // 2
        mid_channels = in_channels
        # build rest conv3x3 layers.
        for idx in range(1, steps):
            if idx < steps - 1:
                mid_channels //= 2
            conv = ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_list.append(conv)
            in_channels = mid_channels

        # add dw conv before second step for down sample if stride = 2.
        if stride == 2:
            self.conv_list[1] = nn.Sequential(
                ConvBNReLU(out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1,
                           groups=out_channels // 2, use_activation=False, bias=False),
                self.conv_list[1])

    def forward(self, x):
        out_list = []
        # run first conv
        x = self.conv_list[0](x)
        out_list.append(self.skip_step1(x))

        for conv in self.conv_list[1:]:
            x = conv(x)
            out_list.append(x)

        out = torch.cat(out_list, dim=1)
        return out


class AbstractSTDCBackbone(nn.Module, ABC):
    """
    All backbones for STDC segmentation models must implement this class.
    """
    def validate_backbone(self):
        assert len(self.get_backbone_output_number_of_channels()) == 3,\
            f"Backbone for STDC segmentation must output 3 feature maps," \
            f" found: {len(self.get_backbone_output_number_of_channels())}."

    @abstractmethod
    def get_backbone_output_number_of_channels(self) -> List[int]:
        """
        :return: list on stages num channels.
        """
        raise NotImplementedError()


class STDCBackbone(AbstractSTDCBackbone):
    def __init__(self,
                 block_types: list,
                 ch_widths: list,
                 num_blocks: list,
                 stdc_steps: int = 4,
                 in_channels: int = 3,
                 out_down_ratios: Union[tuple, list] = (32,)):
        """
        :param block_types: list of block type for each stage, supported `conv` for ConvBNRelu with 3x3 kernel.
        :param ch_widths: list of output num of channels for each stage.
        :param num_blocks: list of the number of repeating blocks in each stage.
        :param stdc_steps: num of convs steps in each block.
        :param in_channels: num channels of the input image.
        :param out_down_ratios: down ratio of output feature maps required from the backbone,
            default (32,) for classification.
        """
        super(STDCBackbone, self).__init__()
        assert len(block_types) == len(ch_widths) == len(num_blocks),\
            f"STDC architecture configuration, block_types, ch_widths, num_blocks, must be defined for the same number" \
            f" of stages, found: {len(block_types)} for block_type, {len(ch_widths)} for ch_widths, " \
            f"{len(num_blocks)} for num_blocks"

        self.out_widths = []
        self.stages = nn.ModuleDict()
        self.out_stage_keys = []
        down_ratio = 2
        for block_type, width, blocks in zip(block_types, ch_widths, num_blocks):
            block_name = f"block_s{down_ratio}"
            self.stages[block_name] = self._make_stage(in_channels=in_channels, out_channels=width,
                                                       block_type=block_type, num_blocks=blocks, stdc_steps=stdc_steps)
            if down_ratio in out_down_ratios:
                self.out_stage_keys.append(block_name)
                self.out_widths.append(width)
            in_channels = width
            down_ratio *= 2

    def _make_stage(self,
                    in_channels: int,
                    out_channels: int,
                    block_type: str,
                    num_blocks: int,
                    stdc_steps: int = 4):
        """
        :param in_channels: input channels of stage.
        :param out_channels: output channels of stage.
        :param block_type: stage building block, supported `conv` for 3x3 ConvBNRelu, or `stdc` for STDCBlock.
        :param num_blocks: num of blocks in each stage.
        :param stdc_steps: number of conv3x3 steps in each STDC block, referred as `num blocks` in paper.
        :return: nn.Module
        """
        if block_type == "conv":
            block = ConvBNReLU
            kwargs = {"kernel_size": 3, "padding": 1, "bias": False}
        elif block_type == "stdc":
            block = STDCBlock
            kwargs = {"steps": stdc_steps}
        else:
            raise ValueError(f"Block type not supported: {block_type}, excepted: `conv` or `stdc`")

        # first block to apply stride 2.
        blocks = nn.ModuleList([
            block(in_channels, out_channels, stride=2, **kwargs)
        ])
        # build rest of blocks
        for i in range(num_blocks - 1):
            blocks.append(block(out_channels, out_channels, stride=1, **kwargs))

        return nn.Sequential(*blocks)

    def forward(self, x):
        outputs = []
        for stage_name, stage in self.stages.items():
            x = stage(x)
            if stage_name in self.out_stage_keys:
                outputs.append(x)
        return tuple(outputs)

    def get_backbone_output_number_of_channels(self) -> List[int]:
        return self.out_widths


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int, dropout: float):
        super(SegmentationHead, self).__init__()
        self.seg_head = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Dropout(dropout),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.seg_head(x)

    def replace_num_classes(self, num_classes: int):
        """
        This method replace the last Conv Classification layer to output a different number of classes.
        Note that the weights of the new layers are random initiated.
        """
        old_cls_conv = self.seg_head[-1]
        self.seg_head[-1] = nn.Conv2d(old_cls_conv.in_channels, num_classes, kernel_size=1, bias=False)


class STDC1Backbone(STDCBackbone):
    def __init__(self, in_channels: int = 3, out_down_ratios: Union[tuple, list] = (32,), pretrained: bool = False):
        super().__init__(block_types=["conv", "conv", "stdc", "stdc", "stdc"],
                         ch_widths=[32, 64, 256, 512, 1024], num_blocks=[1, 1, 2, 2, 2], stdc_steps=4,
                         in_channels=in_channels, out_down_ratios=out_down_ratios)
        if pretrained:
            try:
                d = torch.load(PRETRAINED['stdc1'], map_location='cpu')['state_dict']
                self.load_state_dict(d)
            except:
                print('could not load pretrained weights')


class STDC2Backbone(STDCBackbone):
    def __init__(self, in_channels: int = 3, out_down_ratios: Union[tuple, list] = (32,), pretrained: bool = False):
        super().__init__(block_types=["conv", "conv", "stdc", "stdc", "stdc"],
                         ch_widths=[32, 64, 256, 512, 1024], num_blocks=[1, 1, 4, 5, 3], stdc_steps=4,
                         in_channels=in_channels, out_down_ratios=out_down_ratios)
        if pretrained:
            try:
                d = torch.load(PRETRAINED['stdc2'], map_location='cpu')['state_dict']
                self.load_state_dict(d)
            except:
                print('could not load pretrained weights')
