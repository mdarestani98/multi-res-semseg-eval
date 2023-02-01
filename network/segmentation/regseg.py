"""Implementation of RegSeg, https://github.com/RolandGao/RegSeg"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import DotDict
from utils.tools import NetworkHandler


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if apply_act:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DilatedConv(nn.Module):
    def __init__(self, w, dilations, group_width, stride, bias):
        super().__init__()
        num_splits = len(dilations)
        assert (w % num_splits == 0)
        temp = w // num_splits
        assert (temp % group_width == 0)
        groups = temp // group_width
        convs = []
        for d in dilations:
            convs.append(nn.Conv2d(temp, temp, 3, padding=d, dilation=d, stride=stride, bias=bias, groups=groups))
        self.convs = nn.ModuleList(convs)
        self.num_splits = num_splits

    def forward(self, x):
        x = torch.tensor_split(x, self.num_splits, dim=1)
        res = []
        for i in range(self.num_splits):
            res.append(self.convs[i](x[i]))
        return torch.cat(res, dim=1)


class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(w_in, w_se, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(w_se, w_in, 1, bias=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.act1(self.conv1(y))
        y = self.act2(self.conv2(y))
        return x * y


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg = nn.AvgPool2d(2, 2, ceil_mode=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.avg = None
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.avg is not None:
            x = self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, group_width, stride, attention="se"):
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        if len(dilations) == 1:
            dilation = dilations[0]
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=groups,
                                   padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = DilatedConv(out_channels, dilations, group_width=group_width, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU(inplace=True)
        if attention == "se":
            self.se = SEModule(out_channels, in_channels // 4)
        elif attention == "se2":
            self.se = SEModule(out_channels, out_channels // 4)
        else:
            self.se = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x


class RegSegBody(nn.Module):
    def __init__(self, ds):
        super().__init__()
        gw = 16
        attention = "se"
        self.stage4 = DBlock(32, 48, [1], gw, 2, attention)
        self.stage8 = nn.Sequential(
            DBlock(48, 128, [1], gw, 2, attention),
            DBlock(128, 128, [1], gw, 1, attention),
            DBlock(128, 128, [1], gw, 1, attention)
        )
        self.stage16 = nn.Sequential(
            DBlock(128, 256, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock(256, 256, d, gw, 1, attention)),
            DBlock(256, 320, ds[-1], gw, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        return {"4": x4, "8": x8, "16": x16}

    def channels(self):
        return {"4": 48, "8": 128, "16": 320}


class RegSegBody2(nn.Module):
    def __init__(self, ds):
        super().__init__()
        gw = 24
        attention = "se"
        self.stage4 = nn.Sequential(
            DBlock(32, 48, [1], gw, 2, attention),
            DBlock(48, 48, [1], gw, 1, attention),
        )
        self.stage8 = nn.Sequential(
            DBlock(48, 120, [1], gw, 2, attention),
            *generate_stage(5, lambda: DBlock(120, 120, [1], gw, 1, attention)),
        )
        self.stage16 = nn.Sequential(
            DBlock(120, 336, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock(336, 336, d, gw, 1, attention)),
            DBlock(336, 384, ds[-1], gw, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        return {"4": x4, "8": x8, "16": x16}

    def channels(self):
        return {"4": 48, "8": 120, "16": 384}


class Exp2_Decoder26(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4, channels8, channels16 = channels["4"], channels["8"], channels["16"]
        self.head16 = ConvBnAct(channels16, 128, 1)
        self.head8 = ConvBnAct(channels8, 128, 1)
        self.head4 = ConvBnAct(channels4, 8, 1)
        self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv4 = ConvBnAct(64 + 8, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x4, x8, x16 = x["4"], x["8"], x["16"]
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


class Exp2_Decoder29(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4, channels8, channels16 = channels["4"], channels["8"], channels["16"]
        self.head16 = ConvBnAct(channels16, 256, 1)
        self.head8 = ConvBnAct(channels8, 256, 1)
        self.head4 = ConvBnAct(channels4, 16, 1)
        self.conv8 = ConvBnAct(256, 128, 3, 1, 1)
        self.conv4 = ConvBnAct(128 + 16, 128, 3, 1, 1)
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        x4, x8, x16 = x["4"], x["8"], x["16"]
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


class RegSeg(nn.Module):
    # exp48_decoder26 is what we call RegSeg in our paper
    # exp53_decoder29 is a larger version of exp48_decoder26
    def __init__(self, name, num_classes, pretrained="", ablate_decoder=False, change_num_classes=False):
        super().__init__()
        self.stem = ConvBnAct(3, 32, 3, 2, 1)
        body_name, decoder_name = name.split("_")
        if "exp48" == body_name:
            self.body = RegSegBody([[1], [1, 2]] + 4 * [[1, 4]] + 7 * [[1, 14]])
        elif "exp53" == body_name:
            self.body = RegSegBody2([[1], [1, 2]] + 4 * [[1, 4]] + 7 * [[1, 14]])
        else:
            raise NotImplementedError()
        if "decoder26" == decoder_name:
            self.decoder = Exp2_Decoder26(num_classes, self.body.channels())
        elif "decoder29" == decoder_name:
            self.decoder = Exp2_Decoder29(num_classes, self.body.channels())
        else:
            raise NotImplementedError()
        if pretrained != "" and not ablate_decoder:
            dic = torch.load(pretrained, map_location='cpu')
            if type(dic) == dict and "model" in dic:
                dic = dic['model']
            if change_num_classes:
                current_model = self.state_dict()
                new_state_dict = {}
                print("change_num_classes: True")
                for k in current_model:
                    if dic[k].size() == current_model[k].size():
                        new_state_dict[k] = dic[k]
                    else:
                        print(k)
                        new_state_dict[k] = current_model[k]
                self.load_state_dict(new_state_dict, strict=True)
            else:
                self.load_state_dict(dic, strict=True)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.stem(x)
        x = self.body(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class Handler(NetworkHandler):
    @staticmethod
    def get_from_config(cfg: DotDict) -> Any:
        return RegSeg(cfg.type, cfg.out_channel)


def generate_stage(num, block_fun):
    blocks = []
    for _ in range(num):
        blocks.append(block_fun())
    return blocks


def generate_stage2(ds, block_fun):
    blocks = []
    for d in ds:
        blocks.append(block_fun(d))
    return blocks


def check():
    model = RegSeg('exp48_decoder26', 19).cuda()
    model.train()
    x = torch.randn((16, 3, 1024, 1024), requires_grad=True).cuda()
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    check()
