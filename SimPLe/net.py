import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

'''Vanilla Unet for size [96, 128, 128]'''
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.conv_blocks_context = nn.ModuleList([
            DoubleConv(in_channels_1=in_channels, out_channels_1=16, kernel_size_1=[1, 3, 3], stride_1=(1, 1, 1),
                       padding_1=[0, 1, 1],
                       in_channels_2=16, out_channels_2=16, kernel_size_2=[1, 3, 3], stride_2=(1, 1, 1),
                       padding_2=[0, 1, 1]),
            DoubleConv(in_channels_1=16, out_channels_1=32, kernel_size_1=[3, 3, 3], stride_1=[2, 2, 2],
                       padding_1=[1, 1, 1],
                       in_channels_2=32, out_channels_2=32, kernel_size_2=[3, 3, 3], stride_2=(1, 1, 1),
                       padding_2=[1, 1, 1]),
            DoubleConv(in_channels_1=32, out_channels_1=64, kernel_size_1=[3, 3, 3], stride_1=[2, 2, 2],
                       padding_1=[1, 1, 1],
                       in_channels_2=64, out_channels_2=64, kernel_size_2=[3, 3, 3], stride_2=(1, 1, 1),
                       padding_2=[1, 1, 1]),
            DoubleConv(in_channels_1=64, out_channels_1=128, kernel_size_1=[3, 3, 3], stride_1=[2, 2, 2],
                       padding_1=[1, 1, 1],
                       in_channels_2=128, out_channels_2=128, kernel_size_2=[3, 3, 3], stride_2=(1, 1, 1),
                       padding_2=[1, 1, 1]),
            DoubleConv(in_channels_1=128, out_channels_1=256, kernel_size_1=[3, 3, 3], stride_1=[2, 2, 2],
                       padding_1=[1, 1, 1],
                       in_channels_2=256, out_channels_2=256, kernel_size_2=[3, 3, 3], stride_2=(1, 1, 1),
                       padding_2=[1, 1, 1]),
        ])

        self.conv_blocks_localization = nn.ModuleList([
            DoubleConv(in_channels_1=256, out_channels_1=128, kernel_size_1=[3, 3, 3], stride_1=(1, 1, 1),
                       padding_1=[1, 1, 1],
                       in_channels_2=128, out_channels_2=128, kernel_size_2=[3, 3, 3], stride_2=(1, 1, 1),
                       padding_2=[1, 1, 1]),
            DoubleConv(in_channels_1=128, out_channels_1=64, kernel_size_1=[3, 3, 3], stride_1=(1, 1, 1),
                       padding_1=[1, 1, 1],
                       in_channels_2=64, out_channels_2=64, kernel_size_2=[3, 3, 3], stride_2=(1, 1, 1),
                       padding_2=[1, 1, 1]),
            DoubleConv(in_channels_1=64, out_channels_1=32, kernel_size_1=[3, 3, 3], stride_1=(1, 1, 1),
                       padding_1=[1, 1, 1],
                       in_channels_2=32, out_channels_2=32, kernel_size_2=[3, 3, 3], stride_2=(1, 1, 1),
                       padding_2=[1, 1, 1]),
            DoubleConv(in_channels_1=32, out_channels_1=16, kernel_size_1=[3, 3, 3], stride_1=(1, 1, 1),
                       padding_1=[1, 1, 1],
                       in_channels_2=16, out_channels_2=16, kernel_size_2=[3, 3, 3], stride_2=(1, 1, 1),
                       padding_2=[1, 1, 1]),
        ])

        self.tu = nn.ModuleList([
            nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False),
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False),
            nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False),
            nn.ConvTranspose3d(32, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
            ])

        self.seg = nn.Conv3d(16, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

    def forward(self, x):
        encoders_features = []
        for i in range(len(self.conv_blocks_context)):
            x = self.conv_blocks_context[i](x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, encoders_features[u]), dim=1)
            x = self.conv_blocks_localization[u](x)
        output = self.seg(x)
        return output, x

class DoubleConv(nn.Module):
    def __init__(self, in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1,
                 in_channels_2, out_channels_2, kernel_size_2, stride_2, padding_2):
        super(DoubleConv, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv3d(in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1)),
                # ('dropout', nn.Dropout3d(p=0.5, inplace=True)),
                ('lrule', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ('instnorm',
                 nn.InstanceNorm3d(out_channels_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))
            ])),
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv3d(in_channels_2, out_channels_2, kernel_size_2, stride_2, padding_2)),
                # ('dropout', nn.Dropout3d(p=0.5, inplace=True)),
                ('lrule', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ('instnorm',
                 nn.InstanceNorm3d(out_channels_2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))
            ]))
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = Unet(1, 2).cuda()
    out, _ = net(torch.randn(1, 1, 96, 128, 128).cuda())
    print(out.shape)
