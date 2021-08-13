import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    cardinality = 32

    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(),
            nn.Conv2d(out_channel//2, out_channel//2, kernel_size=3, stride=stride, padding=1, groups= Bottleneck.cardinality,
                  bias=False),
            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(),
            nn.Conv2d(out_channel//2, out_channel, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.shortcut = nn.Sequential()
        if in_channel != out_channel:
            self.shortcut.add_module("short_cunv",
                                     nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module("short_bn",nn.BatchNorm2d(out_channel))

    def forward(self, x):
        return F.relu(self.residual(x) + self.shortcut(x))

class ResNext(nn.Module):
    def __init__(self, nblocks):
        super().__init__()
        self.in_channel = 64
        self.out_channel = 256

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.conv2 = self.block(nblocks[0], 1)
        self.conv3 = self.block(nblocks[1], 2)
        self.conv4 = self.block(nblocks[2], 2)
        self.conv5 = self.block(nblocks[3], 2)
        self.avg_pool = nn.AvgPool2d((1,1))
        self.linear = nn.Linear(self.in_channel,10)

    def block(self, nblock, stride):
        res_block = nn.Sequential()
        strides = [stride] + [1]*(nblock-1)

        for ind, s in enumerate(strides):
            res_block.add_module(f"block {ind}",Bottleneck(self.in_channel, self.out_channel, s))
            self.in_channel = self.out_channel
        self.out_channel = self.out_channel*2

        return res_block

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def ResNext50_32x4d():
    return ResNext([3, 4, 6, 3])