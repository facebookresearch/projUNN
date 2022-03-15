"""
File: models.py
Created Date: Wed Mar 09 2022
Author: Randall Balestriero
-----
Last Modified: Wed Mar 09 2022 3:47:40 AM
Modified By: Randall Balestriero
-----
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
from torch import nn
from .layers import FullWidthConv2d


def conv_block(in_channels, out_channels, width, pool=False, pool_no=2, unitary=False):
    layers = [
        FullWidthConv2d(in_channels, out_channels, kernel_size=width, bias=False)
        if unitary
        else nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, image_length, unitary):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64, image_length, unitary=unitary)
        self.conv2 = conv_block(64, 128, image_length, pool=True, unitary=unitary)
        self.res1 = nn.Sequential(
            conv_block(128, 128, image_length // 2, unitary=unitary),
            conv_block(128, 128, image_length // 2, unitary=unitary),
        )

        self.conv3 = conv_block(128, 256, image_length // 2, pool=True, unitary=unitary)
        self.conv4 = conv_block(
            256, 256, image_length // 4, pool=False, unitary=unitary
        )
        self.res2 = nn.Sequential(
            conv_block(256, 256, image_length // 4, unitary=unitary),
            conv_block(256, 256, image_length // 4, unitary=unitary),
        )

        self.MP = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.MP(out)
        out = self.classifier(out.flatten(1))
        return out
