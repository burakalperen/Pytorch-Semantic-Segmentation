import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        

class _AsppConv(nn.Module):
    def __init__(self,in_channels, out_channels, atrous_rate, bias):
        super(_AsppConv,self).__init__()
        self.ASPPconvBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding = atrous_rate, dilation = atrous_rate, bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        return self.ASPPconvBlock(x)

class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(_AsppPooling,self).__init__()
        self.asspPool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ) 

    def forward(self,x):
        h = x.size()[2]
        w = x.size()[3]
       
        pool = self.asspPool(x)
        out = F.upsample(pool, size = (h,w), mode="bilinear", align_corners=True)
        return out

class _ASPP(nn.Module):
    def __init__(self,num_classes, in_channels,atrous_rates,bias):
        super(_ASPP,self).__init__()
        out_channels = 256
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.conv_3x3_1 = _AsppConv(in_channels, out_channels, rate1, bias = bias)
        self.conv_3x3_2 = _AsppConv(in_channels, out_channels, rate2, bias = bias)
        self.conv_3x3_3 = _AsppConv(in_channels, out_channels, rate3, bias = bias)
        self.pool = _AsppPooling(in_channels, out_channels, bias = bias)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size = 1, bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5) # birinde var birinde yok
        )

        self.out = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self,feature_map):
        feat1 = self.conv_1x1(feature_map)
        feat2 = self.conv_3x3_1(feature_map)
        feat3 = self.conv_3x3_2(feature_map)
        feat4 = self.conv_3x3_3(feature_map)
        feat5 = self.pool(feature_map)
        print(feat1.shape,feat2.shape,feat3.shape,feat4.shape,feat5.shape)
        x = torch.cat((feat1,feat2,feat3,feat4,feat5),dim = 1)
        x = self.project(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = _ASPP(512,(6,12,18),bias = False).to(device)

    x = torch.randn(2,512,100,100).to(device)

    out = model(x)
    print(x.shape)