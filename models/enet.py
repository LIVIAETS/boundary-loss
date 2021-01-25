#!/usr/bin/env python3.7

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def conv_block_3_3(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def conv_block_Asym(in_dim, out_dim, kernelSize):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernelSize, 1], padding=tuple([2, 0])),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernelSize], padding=tuple([0, 2])),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        nn.PReLU()
    )


def upSampleConv(nin, nout, kernel_size=3, upscale=2, padding=1, bias=False):
    return nn.Sequential(
        # nn.Upsample(scale_factor=upscale),
        interpolate(mode='nearest', scale_factor=upscale),
        convBatch(nin, nout, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
        convBatch(nout, nout, kernel_size=3, stride=1, padding=1, bias=bias),
    )


class interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, cin):
        return F.interpolate(cin, mode=self.mode, scale_factor=self.scale_factor)


class BottleNeckDownSampling(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim):
        super().__init__()

        # Main branch
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)
        # Secondary branch
        self.conv0 = nn.Conv2d(in_dim, int(in_dim / projectionFactor), kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU0 = nn.PReLU()

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Main branch
        maxpool_output, indices = self.maxpool0(input)

        # Secondary branch
        c0 = self.conv0(input)
        b0 = self.bn0(c0)
        p0 = self.PReLU0(b0)

        c1 = self.conv1(p0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        p2 = self.block2(p1)

        do = self.do(p2)

        # Zero padding the feature maps from the main branch
        depth_to_pad = abs(maxpool_output.shape[1] - do.shape[1])
        padding = torch.zeros(maxpool_output.shape[0], depth_to_pad, maxpool_output.shape[2],
                              maxpool_output.shape[3], device=maxpool_output.device)
        maxpool_output_pad = torch.cat((maxpool_output, padding), 1)
        output = maxpool_output_pad + do

        # _, c, _, _ = maxpool_output.shape
        # output = do
        # output[:, :c, :, :] += maxpool_output

        final_output = self.PReLU3(output)

        return final_output, indices


class BottleNeckNormal(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, dropoutRate):
        super(BottleNeckNormal, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))
        self.block1 = conv_block_3_3(int(in_dim / projectionFactor), int(in_dim / projectionFactor))
        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block_1(in_dim, out_dim)

    def forward(self, input):
        # Main branch
        # Secondary branch
        b0 = self.block0(input)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        if self.in_dim > self.out_dim:
            output = self.conv_out(input) + do
        else:
            output = input + do
        output = self.PReLU_out(output)

        return output


class BottleNeckDownSamplingDilatedConv(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim, dilation):
        super(BottleNeckDownSamplingDilatedConv, self).__init__()
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Secondary branch
        b0 = self.block0(input)

        c1 = self.conv1(b0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        b2 = self.block2(p1)

        do = self.do(b2)

        output = input + do
        output = self.PReLU3(output)

        return output


class BottleNeckNormal_Asym(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, dropoutRate):
        super(BottleNeckNormal_Asym, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))
        self.block1 = conv_block_Asym(int(in_dim / projectionFactor), int(in_dim / projectionFactor), 5)
        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block_1(in_dim, out_dim)

    def forward(self, input):
        # Main branch
        # Secondary branch
        b0 = self.block0(input)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        if self.in_dim > self.out_dim:
            output = self.conv_out(input) + do
        else:
            output = input + do
        output = self.PReLU_out(output)

        return output


class BottleNeckDownSamplingDilatedConvLast(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim, dilation):
        super(BottleNeckDownSamplingDilatedConvLast, self).__init__()
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.conv_out = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):

        # Secondary branch
        b0 = self.block0(input)

        c1 = self.conv1(b0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        b2 = self.block2(p1)

        do = self.do(b2)

        output = self.conv_out(input) + do
        output = self.PReLU3(output)

        return output


class BottleNeckUpSampling(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim):
        super(BottleNeckUpSampling, self).__init__()
        # Main branch
        self.conv0 = nn.Conv2d(in_dim, int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU0 = nn.PReLU()

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Secondary branch
        c0 = self.conv0(input)
        b0 = self.bn0(c0)
        p0 = self.PReLU0(b0)

        c1 = self.conv1(p0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        p2 = self.block2(p1)

        do = self.do(p2)

        return do