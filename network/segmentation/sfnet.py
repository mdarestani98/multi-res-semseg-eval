# Implementation of SFNet ResNet series.
# Author: Xiangtai(lxt@pku.edu.cn)
# Date: 2022/5/20

"""https://github.com/lxtGH/SFSegNets"""

from typing import Any

import torch.nn as nn
import torch
import torch.nn.functional as F

from network.segmentation.resnet_d import resnet50, resnet101, resnet18
from network.segmentation.stdcnet import STDCNet813, STDCNet1446
from utils.config import DotDict
from utils.tools import NetworkHandler


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=1, padding=0, dilation=1,
                      bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class SPSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(SPSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        self.down = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=1, bias=False),
            norm_layer(out_features)
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages]
        sum_feat = self.down(feats)

        for feat in priors:
            sum_feat = sum_feat + feat

        bottle = self.bottleneck(sum_feat)
        return bottle


class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class AlignedModuleV2(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModuleV2, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(outplane * 2, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        flow_gates = self.flow_gate(torch.cat([h_feature, l_feature], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class AlignedModuleV2PoolingAttention(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModuleV2PoolingAttention, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        h_feature_mean = torch.mean(h_feature, dim=1).unsqueeze(1)
        l_feature_mean = torch.mean(low_feature, dim=1).unsqueeze(1)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(low_feature, dim=1)[0].unsqueeze(1)

        flow_gates = self.flow_gate(torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class UpperNetHead(nn.Module):
    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=None, fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False, global_context="ppm"):
        super(UpperNetHead, self).__init__()
        if fpn_inplanes is None:
            fpn_inplanes = [256, 512, 1024, 2048]
        if global_context == "ppm":
            self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            size = conv_x.size()[2:]
            f = F.interpolate(f, size, mode='bilinear', align_corners=True)
            # f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x, out


class UpperNetAlignHead(nn.Module):
    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=None, fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False, global_context="ppm"):
        super(UpperNetAlignHead, self).__init__()
        if fpn_inplanes is None:
            fpn_inplanes = [256, 512, 1024, 2048]
        if global_context == "ppm":
            self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        elif global_context == "sppm":
            self.ppm = SPSPModule(inplane, sizes=(1, 2, 4), norm_layer=norm_layer, out_features=fpn_dim)

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            if conv3x3_type == 'conv':
                self.fpn_out_align.append(
                    AlignedModule(inplane=fpn_dim, outplane=fpn_dim // 2)
                )

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x, out


class UpperNetAlignHeadV2(nn.Module):
    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=None, fpn_dim=256,
                 fpn_dsn=False, fa_type="spatial", global_context="ppm"):
        super(UpperNetAlignHeadV2, self).__init__()

        if fpn_inplanes is None:
            fpn_inplanes = [256, 512, 1024, 2048]
        if global_context == "ppm":
            self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        elif global_context == "sppm":
            self.ppm = SPSPModule(inplane, sizes=(1, 2, 4), norm_layer=norm_layer, out_features=fpn_dim)

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes:  # total 2 planes
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out_align = []
        if fa_type == "spatial":
            self.flow_align_module = AlignedModuleV2(inplane=fpn_dim, outplane=fpn_dim // 2)
        elif fa_type == "spatial_atten":
            self.flow_align_module = AlignedModuleV2PoolingAttention(inplane=fpn_dim, outplane=fpn_dim // 2)

        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                norm_layer(fpn_dim),
                nn.ReLU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(fpn_dim * 2, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):

        # p2, p3, p4, p5(ppm)
        psp_out = self.ppm(conv_out[-1])

        p2 = conv_out[0]
        p4 = conv_out[2]

        p2 = self.fpn_in[0](p2)
        p4 = self.fpn_in[1](p4)

        fusion_out = self.flow_align_module([p2, psp_out])
        output_size = fusion_out.size()[2:]

        p4 = nn.functional.interpolate(
            p4,
            output_size,
            mode='bilinear', align_corners=True)

        x = self.conv_last(torch.cat([fusion_out, p4], dim=1))

        return x, None


class AlignNetResNet(nn.Module):
    def __init__(self, num_classes, trunk='resnet-101', variant='D', head_type="v1",
                 skip='m1', skip_num=48, fpn_dsn=False, fa_type="spatial", global_context="ppm", pretrained=False):
        super(AlignNetResNet, self).__init__()
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.fpn_dsn = fpn_dsn

        if trunk == trunk == 'resnet-50-deep':
            resnet = resnet50(pretrained=pretrained)
        elif trunk == 'resnet-101-deep':
            resnet = resnet101(pretrained=pretrained)
        elif trunk == 'resnet-18-deep':
            resnet = resnet18(pretrained=pretrained)
        else:
            raise ValueError("Not a valid network arch")

        resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        del resnet

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
        else:
            print("Not using Dilation ")

        if trunk == 'resnet-18-deep':
            inplane_head = 512
            if head_type == "v2":
                self.head = UpperNetAlignHeadV2(inplane_head, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                                fpn_inplanes=[64, 256, 512], fpn_dim=128, fpn_dsn=fpn_dsn,
                                                fa_type=fa_type, global_context=global_context
                                                )
            else:
                self.head = UpperNetAlignHead(inplane_head, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                              fpn_inplanes=[64, 128, 256, 512], fpn_dim=128, fpn_dsn=fpn_dsn,
                                              global_context=global_context
                                              )
        else:
            inplane_head = 2048
            self.head = UpperNetAlignHead(inplane_head, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                          fpn_dsn=fpn_dsn)

    def forward(self, x):
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        x = self.head([x1, x2, x3, x4])
        x = [x[0]] + x[1]
        x = [F.interpolate(xt, x_size[2:], mode='bilinear', align_corners=True) for xt in x if xt is not None]
        main_out = x[0]
        if self.training:
            if not self.fpn_dsn:
                return main_out
            return tuple(x)
        return main_out


class AlignNetSTDCnet(nn.Module):

    def __init__(self, num_classes, trunk='stdc1', flow_conv_type="conv",
                 head_type="v1", fpn_dsn=False, global_context="ppm", fa_type="spatial", pretrained=False):
        super(AlignNetSTDCnet, self).__init__()
        self.fpn_dsn = fpn_dsn

        if trunk == 'stdc1':
            self.backbone = STDCNet813(norm_layer=nn.BatchNorm2d, pretrain_model=pretrained)
        elif trunk == 'stdc2':
            self.backbone = STDCNet1446(norm_layer=nn.BatchNorm2d, pretrain_model=pretrained)
        else:
            raise ValueError("Not a valid network arch")
        if head_type == "v2":
            self.head = UpperNetAlignHeadV2(1024, num_class=num_classes, norm_layer=nn.BatchNorm2d, fa_type=fa_type,
                                            fpn_inplanes=[64, 512], fpn_dim=128, fpn_dsn=fpn_dsn,
                                            global_context=global_context)
        else:
            self.head = UpperNetAlignHead(inplane=1024, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                          fpn_inplanes=[64, 256, 512, 1024], fpn_dim=64, conv3x3_type=flow_conv_type,
                                          fpn_dsn=fpn_dsn, global_context=global_context)

    def forward(self, x):
        x_size = x.size()  # 800
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)

        x = self.head([feat4, feat8, feat16, feat32])
        x = [F.interpolate(xt, x_size[2:], mode='bilinear', align_corners=True) for xt in x if xt is not None]
        main_out = x[0]
        if self.training:
            if not self.fpn_dsn:
                return main_out
            else:
                return tuple(x)
        return main_out


class Handler(NetworkHandler):
    @staticmethod
    def get_from_config(cfg: DotDict) -> Any:
        if cfg.head_type is None:
            cfg.head_type = 'v1'
        if cfg.fa_type is None:
            cfg.fa_type = 'spatial'
        if 'resnet' in cfg.backbone:
            return AlignNetResNet(num_classes=cfg.out_channel, trunk=cfg.backbone, fpn_dsn=cfg.fpn_dsn,
                                  head_type=cfg.head_type, fa_type=cfg.fa_type, pretrained=cfg.pretrained)
        elif 'stdc' in cfg.backbone:
            return AlignNetSTDCnet(num_classes=cfg.out_channel, trunk=cfg.backbone, fpn_dsn=cfg.fpn_dsn,
                                   head_type=cfg.head_type, fa_type=cfg.fa_type, pretrained=cfg.pretrained)
