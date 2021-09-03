import torch
import torch.nn as nn

in_channels = 256
channels = 512
stride = 1
dilation = 2
out_channels = 512

conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
bn1 = nn.BatchNorm2d(channels)

conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
bn2 = nn.BatchNorm2d(channels)


conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
bn = nn.BatchNorm2d(out_channels)
downsample = nn.Sequential(conv, bn)


seq = nn.Sequential(conv1,bn1,conv2,bn2,conv,bn,downsample)


print(seq)