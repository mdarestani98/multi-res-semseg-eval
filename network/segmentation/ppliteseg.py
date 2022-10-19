"""Implementation of PP-LiteSeg, https://github.com/Deci-AI/super-gradients"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Any

from network.segmentation.stdc_backbone import AbstractSTDCBackbone, SegmentationHead, STDC2Backbone, STDC1Backbone, \
    ConvBNReLU
from utils.config import DotDict
from utils.tools import NetworkHandler


class SegmentationModule(ABC, nn.Module):
    def __init__(self, use_aux_heads: bool):
        super().__init__()
        self._use_aux_heads = use_aux_heads

    @property
    def use_aux_heads(self):
        return self._use_aux_heads

    @use_aux_heads.setter
    def use_aux_heads(self, use_aux: bool):
        """
        public setter for self._use_aux_heads, called every time an assignment to self.use_aux_heads is applied.
        if use_aux is False, `_remove_auxiliary_heads` is called to delete auxiliary and detail heads.
        if use_aux is True, and self._use_aux_heads was already set to False a ValueError is raised, recreating
            aux and detail heads outside init method is not allowed, and the module should be recreated.
        """
        if use_aux is True and self._use_aux_heads is False:
            raise ValueError("Cant turn use_aux_heads from False to True. Try initiating the module again with"
                             " `use_aux_heads=True` or initiating the auxiliary heads modules manually.")
        if not use_aux:
            self._remove_auxiliary_heads()
        self._use_aux_heads = use_aux

    @abstractmethod
    def _remove_auxiliary_heads(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def backbone(self) -> nn.Module:
        """
        For SgTrainer load_backbone compatibility.
        """
        raise NotImplementedError()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SPPM(nn.Module):
    """
    Simple Pyramid Pooling context Module.
    """
    def __init__(self,
                 in_channels: int,
                 inter_channels: int,
                 out_channels: int,
                 pool_sizes: List[Union[int, Tuple[int, int]]],
                 align_corners: bool = False):
        """
        :param inter_channels: num channels in each pooling branch.
        :param out_channels: The number of output channels after pyramid pooling module.
        :param pool_sizes: spatial output sizes of the pooled feature maps.
        """
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                ConvBNReLU(in_channels, inter_channels, kernel_size=1, bias=False),
            ) for pool_size in pool_sizes
        ])
        self.conv_out = ConvBNReLU(inter_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.pool_sizes = pool_sizes

    def forward(self, x):
        out = None
        input_shape = x.shape[2:]
        for branch in self.branches:
            y = branch(x)
            y = F.interpolate(y, size=input_shape, mode='bilinear', align_corners=self.align_corners)
            out = y if out is None else out + y
        out = self.conv_out(out)
        return out

    def prep_model_for_conversion(self, input_size: Union[tuple, list], stride_ratio: int = 32, **kwargs):
        """
        Replace Global average pooling with fixed kernels Average pooling, since dynamic kernel sizes are not supported
        when compiling to ONNX: `Unsupported: ONNX export of operator adaptive_avg_pool2d, input size not accessible.`
        """
        input_size = [x / stride_ratio for x in input_size[-2:]]
        for branch in self.branches:
            global_pool: nn.AdaptiveAvgPool2d = branch[0]
            out_size = global_pool.output_size
            out_size = out_size if isinstance(out_size, (tuple, list)) else (out_size, out_size)
            kernel_size = [int(i / o) for i, o in zip(input_size, out_size)]
            branch[0] = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)


class UAFM(nn.Module):
    """
    Unified Attention Fusion Module, which uses mean and max values across the spatial dimensions.
    """
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 up_factor: int,
                 align_corners: bool = False):
        """
        :params in_channels: num_channels of input feature map.
        :param skip_channels: num_channels of skip connection feature map.
        :param out_channels: num out channels after features fusion.
        :param up_factor: upsample scale factor of the input feature map.
        """
        super().__init__()
        self.conv_atten = nn.Sequential(
            ConvBNReLU(4, 2, kernel_size=3, padding=1, bias=False),
            ConvBNReLU(2, 1, kernel_size=3, padding=1, bias=False, use_activation=False)
        )

        self.proj_skip = nn.Identity() if skip_channels == in_channels else \
            ConvBNReLU(skip_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.up_x = nn.Identity() if up_factor == 1 else \
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=align_corners)
        self.conv_out = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, skip):
        """
        :param x: input feature map to upsample before fusion.
        :param skip: skip connection feature map.
        """
        x = self.up_x(x)
        skip = self.proj_skip(skip)

        atten = torch.cat([
            *self._avg_max_spatial_reduce(x, use_concat=False),
            *self._avg_max_spatial_reduce(skip, use_concat=False)
        ], dim=1)
        atten = self.conv_atten(atten)
        atten = torch.sigmoid(atten)

        out = x * atten + skip * (1 - atten)
        out = self.conv_out(out)
        return out

    @staticmethod
    def _avg_max_spatial_reduce(x, use_concat: bool = False):
        reduced = [
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]
        ]
        if use_concat:
            reduced = torch.cat(reduced, dim=1)
        return reduced


class PPLiteSegEncoder(nn.Module):
    """
    Encoder for PPLiteSeg, include backbone followed by a context module.
    """
    def __init__(self,
                 backbone: AbstractSTDCBackbone,
                 projection_channels_list: List[int],
                 context_module: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.context_module = context_module
        feats_channels = backbone.get_backbone_output_number_of_channels()
        self.proj_convs = nn.ModuleList([
            ConvBNReLU(feat_ch, proj_ch, kernel_size=3, padding=1, bias=False)
            for feat_ch, proj_ch in zip(feats_channels, projection_channels_list)
        ])
        self.projection_channels_list = projection_channels_list

    def get_output_number_of_channels(self) -> List[int]:
        channels_list = self.projection_channels_list
        if hasattr(self.context_module, "out_channels"):
            channels_list.append(self.context_module.out_channels)
        return channels_list

    def forward(self, x):
        feats = self.backbone(x)
        y = self.context_module(feats[-1])
        feats = [conv(f) for conv, f in zip(self.proj_convs, feats)]
        return feats + [y]


class PPLiteSegDecoder(nn.Module):
    """
    PPLiteSegDecoder using UAFM blocks to fuse feature maps.
    """
    def __init__(self,
                 encoder_channels: List[int],
                 up_factors: List[int],
                 out_channels: List[int],
                 align_corners: bool):
        super().__init__()
        # Make a copy of channels list, to prevent out of scope changes.
        encoder_channels = encoder_channels.copy()
        encoder_channels.reverse()
        in_channels = encoder_channels.pop(0)

        self.up_stages = nn.ModuleList()
        for skip_ch, up_factor, out_ch in zip(encoder_channels, up_factors, out_channels):
            self.up_stages.append(UAFM(
                in_channels=in_channels,
                skip_channels=skip_ch,
                out_channels=out_ch,
                up_factor=up_factor,
                align_corners=align_corners
            ))
            in_channels = out_ch

    def forward(self, feats: List[torch.Tensor]):
        feats.reverse()
        x = feats.pop(0)
        for up_stage, skip in zip(self.up_stages, feats):
            x = up_stage(x, skip)
        return x


class PPLiteSegBase(SegmentationModule):
    """
    The PP_LiteSeg implementation based on PaddlePaddle.
    The original article refers to "Juncai Peng, Yi Liu, Shiyu Tang, Yuying Hao, Lutao Chu,
    Guowei Chen, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang,Baohua Lai,
    Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma. PP-LiteSeg: A Superior Real-Time Semantic
    Segmentation Model. https://arxiv.org/abs/2204.02681".
    """
    def __init__(self,
                 num_classes,
                 backbone: AbstractSTDCBackbone,
                 projection_channels_list: List[int],
                 sppm_inter_channels: int,
                 sppm_out_channels: int,
                 sppm_pool_sizes: List[int],
                 align_corners: bool,
                 decoder_up_factors: List[int],
                 decoder_channels: List[int],
                 head_scale_factor: int,
                 head_upsample_mode: str,
                 head_mid_channels: int,
                 dropout: float,
                 use_aux_heads: bool,
                 aux_hidden_channels: List[int],
                 aux_scale_factors: List[int]
                 ):
        """
        :param backbone: Backbone nn.Module should implement the abstract class `AbstractSTDCBackbone`.
        :param projection_channels_list: channels list to project encoder features before fusing with the decoder
            stream.
        :param sppm_inter_channels: num channels in each sppm pooling branch.
        :param sppm_out_channels: The number of output channels after sppm module.
        :param sppm_pool_sizes: spatial output sizes of the pooled feature maps.
        :param decoder_up_factors: list upsample factor per decoder stage.
        :param decoder_channels: list of num_channels per decoder stage.
        :param head_scale_factor: scale factor for final the segmentation head logits.
        :param head_upsample_mode: upsample mode to final prediction sizes, see UpsampleMode for valid options.
        :param head_mid_channels: num of hidden channels in segmentation head.
        :param use_aux_heads: set True when training, output extra Auxiliary feature maps from the encoder module.
        :param aux_hidden_channels: List of hidden channels in auxiliary segmentation heads.
        :param aux_scale_factors: list of uppsample factors for final auxiliary heads logits.
        """
        super().__init__(use_aux_heads=use_aux_heads)

        # Init Encoder
        backbone_out_channels = backbone.get_backbone_output_number_of_channels()
        assert len(backbone_out_channels) == len(projection_channels_list), \
            f"The length of backbone outputs ({backbone_out_channels}) should match the length of projection channels" \
            f"({len(projection_channels_list)})."
        context = SPPM(in_channels=backbone_out_channels[-1],
                       inter_channels=sppm_inter_channels,
                       out_channels=sppm_out_channels,
                       pool_sizes=sppm_pool_sizes,
                       align_corners=align_corners)
        self.encoder = PPLiteSegEncoder(backbone=backbone,
                                        context_module=context,
                                        projection_channels_list=projection_channels_list)
        encoder_channels = self.encoder.get_output_number_of_channels()

        # Init Decoder
        self.decoder = PPLiteSegDecoder(encoder_channels=encoder_channels,
                                        up_factors=decoder_up_factors,
                                        out_channels=decoder_channels,
                                        align_corners=align_corners)

        # Init Segmentation classification heads
        self.seg_head = nn.Sequential(
            SegmentationHead(in_channels=decoder_channels[-1],
                             mid_channels=head_mid_channels,
                             num_classes=num_classes,
                             dropout=dropout),
            nn.Upsample(scale_factor=head_scale_factor, mode=head_upsample_mode, align_corners=align_corners)
        )
        # Auxiliary heads
        if self.use_aux_heads:
            encoder_out_channels = projection_channels_list
            self.aux_heads = nn.ModuleList([
                nn.Sequential(
                    SegmentationHead(backbone_ch, hidden_ch, num_classes, dropout=dropout),
                    nn.Upsample(scale_factor=scale_factor, mode=head_upsample_mode, align_corners=align_corners)
                ) for backbone_ch, hidden_ch, scale_factor in zip(encoder_out_channels, aux_hidden_channels,
                                                                  aux_scale_factors)
            ])
        self.init_params()

    def _remove_auxiliary_heads(self):
        if hasattr(self, "aux_heads"):
            del self.aux_heads

    @property
    def backbone(self) -> nn.Module:
        """
        Support SG load backbone when training.
        """
        return self.encoder.backbone

    def forward(self, x):
        feats = self.encoder(x)
        if self.use_aux_heads:
            enc_feats = feats[:-1]
        x = self.decoder(feats)
        x = self.seg_head(x)
        if not self.use_aux_heads:
            return x
        aux_feats = [aux_head(feat) for feat, aux_head in zip(enc_feats, self.aux_heads)]
        return tuple([x] + aux_feats)

    def replace_head(self, new_num_classes: int, **kwargs):
        for module in self.modules():
            if isinstance(module, SegmentationHead):
                module.replace_num_classes(new_num_classes)


class Handler(NetworkHandler):
    @staticmethod
    def get_from_config(cfg: DotDict) -> Any:
        if cfg.type == 'b':
            return PPLiteSegB(cfg.out_channel, cfg.in_channel, cfg.dropout, cfg.aux_output)
        elif cfg.type == 't':
            return PPLiteSegT(cfg.out_channel, cfg.in_channel, cfg.dropout, cfg.aux_output)
        else:
            raise ValueError(f'Only type b and t are valid, got {cfg.type}')


class PPLiteSegB(PPLiteSegBase):
    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.0, aux: bool = False):
        backbone = STDC2Backbone(in_channels=in_channels,
                                 out_down_ratios=[8, 16, 32])
        super().__init__(num_classes=num_classes,
                         backbone=backbone,
                         projection_channels_list=[96, 128, 128],
                         sppm_inter_channels=128,
                         sppm_out_channels=128,
                         sppm_pool_sizes=[1, 2, 4],
                         align_corners=False,
                         decoder_up_factors=[1, 2, 2],
                         decoder_channels=[128, 96, 64],
                         head_scale_factor=8,
                         head_upsample_mode="bilinear",
                         head_mid_channels=64,
                         dropout=dropout,
                         use_aux_heads=aux,
                         aux_hidden_channels=[32, 64, 64],
                         aux_scale_factors=[8, 16, 32])


class PPLiteSegT(PPLiteSegBase):
    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.0, aux: bool = False):
        backbone = STDC1Backbone(in_channels=in_channels,
                                 out_down_ratios=[8, 16, 32])
        super().__init__(num_classes=num_classes,
                         backbone=backbone,
                         projection_channels_list=[64, 128, 128],
                         sppm_inter_channels=128,
                         sppm_out_channels=128,
                         sppm_pool_sizes=[1, 2, 4],
                         align_corners=False,
                         decoder_up_factors=[1, 2, 2],
                         decoder_channels=[128, 64, 32],
                         head_scale_factor=8,
                         head_upsample_mode="bilinear",
                         head_mid_channels=32,
                         dropout=dropout,
                         use_aux_heads=aux,
                         aux_hidden_channels=[32, 64, 64],
                         aux_scale_factors=[8, 16, 32])


def check():
    model = PPLiteSegB(19, 3, 0.0, True).cuda()
    model.eval()
    x = torch.randn((32, 3, 320, 320)).cuda()
    y = model(x)
    print([yy.shape for yy in y])


if __name__ == '__main__':
    check()
