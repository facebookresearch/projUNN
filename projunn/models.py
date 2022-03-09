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
import torch
from . import utils


class FullWidthConv2d(nn.Conv2d):
    def __init__(
        self,
        n_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        dtype="real",
        init_as_id=False,
        **kwargs,
    ):
        self.real = dtype == "real"
        self.stride = stride
        if self.real:
            if isinstance(kernel_size, int):
                self.input_size = [kernel_size, kernel_size]
                kernel_size = [kernel_size, kernel_size // 2 + 1]
            else:
                self.input_size = kernel_size
                kernel_size = [kernel_size[0], kernel_size[1] // 2 + 1]
        else:
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            self.input_size = kernel_size

        self.pre_init = True
        super().__init__(
            out_channels, out_channels, kernel_size, bias=bias, stride=stride, **kwargs
        )
        self.pre_init = False

        if self.out_channels != self.in_channels:
            raise ValueError(
                "Number of output channels (out_channels) must equal number of input channels (in_channels)"
            )
        self.weight.data = (
            self.weight.data.reshape(self.out_channels, self.in_channels, -1)
            .permute(2, 0, 1)
            .to(torch.cfloat)
        )
        self.weight.needs_projection = True
        self.weight.ortho_regularizer_size = kernel_size
        if self.real:
            self.fft_op = torch.fft.rfft2
            self.ifft_op = torch.fft.irfft2
            self.get_hermitize_ids()
        else:
            self.fft_op = torch.fft.fft2
            self.ifft_op = torch.fft.ifft2

        self.reset_parameters(init_as_id=init_as_id)

        self.need_check_input_dim = True  # sets to False once checked that kernel is valid size (i.e. full support)

    def hermitize(self):
        # matches matrices along specific dimensions to maintain orthogonality
        with torch.no_grad():
            self.weight[self.hermitize_ids[1]] = self.weight[self.hermitize_ids[0]]

    def get_hermitize_ids(self):
        id_list = []
        match_list = []
        for i in range(self.weight.shape[0]):
            match_id = get_matching_hermitian_fft_dim(
                i, self.kernel_size, self.input_size
            )
            if i > match_id:
                id_list.append(i)
                match_list.append(match_id)
        self.hermitize_ids = (match_list, id_list)

    def reset_parameters(self, init_as_id=False):
        if not self.pre_init:
            for i in range(self.weight.shape[0]):
                if init_as_id:
                    nn.init.eye_(self.weight.data[i])
                else:
                    utils.orthogonal_(self.weight.data[i], real_only=self.real)
            if self.real:
                self.hermitize()

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def get_required_input_range(self):
        if self.real:
            return [
                self.kernel_size[0] * (self.kernel_size[1] - 1) * 2,
                self.kernel_size[0] * (self.kernel_size[1] * 2 - 1),
            ]
        else:
            return [self.kernel_size[0] * self.kernel_size[1]] * 2

    def forward(self, x):
        diff = self.out_channels // x.size(1)
        if diff > 1:
            x = x.repeat(1, diff, 1, 1)
        elif diff < 1:
            raise RuntimeError("can not go to less channels")
        N_pixels, cout, cin = self.weight.shape
        batches, _, n, m = x.shape

        if self.need_check_input_dim:
            self.need_check_input_dim = False
            valid_pixels = self.get_required_input_range()
            if n * m != valid_pixels[0] and n * m != valid_pixels[1]:
                raise ValueError(
                    f"kernel size ({self.weight.shape}) must match input size ({x.shape})"
                )

        xfft = (
            self.fft_op(x, s=[n, m])
            .permute(2, 3, 1, 0)
            .reshape(self.kernel_size[0] * self.kernel_size[1], cin, batches)
        )

        yfft = (self.weight @ xfft).reshape(
            self.kernel_size[0], self.kernel_size[1], cout, batches
        )
        y = self.ifft_op(yfft.permute(3, 2, 0, 1), s=[n, m])
        if self.bias is not None:
            y += self.bias[:, None, None]
        if self.stride[0] > 1 or self.stride[1] > 1:
            y = y[:, :, :: self.stride[0], :: self.stride[1]]
        return y


def one_to_two_fft_indices(i, kernel_size):
    return i // kernel_size[1], i % kernel_size[1]


def two_to_one_fft_indices(i, j, kernel_size):
    return i * kernel_size[1] + j


def get_matching_hermitian_fft_dim(i, kernel_size, input_size=None):
    if input_size is None:
        input_size = kernel_size.copy()
        input_size[1] *= 2
    a, b = one_to_two_fft_indices(i, kernel_size)
    if b == 0 or (b == (kernel_size[1] - 1) and (input_size[1] % 2) == 0):
        if a == 0:
            return -1
        elif a * 2 == (kernel_size[0]):
            return -1
        else:
            return two_to_one_fft_indices(-1 * a % kernel_size[0], b, kernel_size)
    else:
        return -1


def conv_block(in_channels, out_channels, width, pool=False, pool_no=2):
    layers = [
        FullWidthConv2d(in_channels, out_channels, kernel_size=width, bias=False)
        if in_channels > 3
        else nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64, 32)
        self.conv2 = conv_block(64, 128, 32, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128, 16), conv_block(128, 128, 16))

        self.conv3 = conv_block(128, 256, 16, pool=True)
        self.conv4 = conv_block(256, 256, 8, pool=True)
        self.res2 = nn.Sequential(conv_block(256, 256, 4), conv_block(256, 256, 4))

        self.MP = nn.MaxPool2d(2)
        self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.MP(out)
        out_emb = self.FlatFeats(out)
        out = self.classifier(out_emb)
        return out
