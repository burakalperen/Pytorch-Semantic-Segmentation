import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from torchvision.models.resnet import Bottleneck
from torchvision.models.vgg import make_layers

# deeplab kanal olarak 512 alıyor


def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

    return layer

# def make_layer(block, in_channels, channels, num_blocks, stride = 1, dilation = 1):
#     strides = [stride] + [1]*(num_blocks - 1)

#     blocks = []
#     for stride in strides:
#         blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
#         in_channels = block.expansion*channels

#     layer = nn.Sequential(*blocks)
#     return layer

class ResNet_Bottleneck_OS16(nn.Module):
    def __init__(self,num_layers):
        super(ResNet_Bottleneck_OS16,self).__init__()

        if num_layers == 50:
            resnet = models.resnet50()
            self.resnet = nn.Sequential(*list(resnet.children())[:-3]) # 7-8 gitti
            print("pretrained resnet50")
        elif num_layers == 101:
            resnet = models.resnet101()
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
            print("pretrained resnet101")

        elif num_layers == 152:
            resnet = models.resnet152()
            self.resnet = nn.Sequential(*list(resnet.childrenn())[:-3])
            print("pretrained resnet152")
        else:
            raise Exception("num_layers must be in {50, 101, 152}")

        self.layer_5 = make_layer(Bottleneck, in_channels=4*256, channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self,x):
        c4 = self.resnet(x)
        output = self.layer_5(c4)
        return output


if __name__ == "__main__":
    resnet = models.resnet50()
    
    model_2 = nn.Sequential(*list(resnet.children())[:-3])

    #summary(model_2.cuda(),(3,224,224))

    # print(resnet)
    # print("************************************************")
    # print("************************************************")
    # print("************************************************")
    # print("************************************************")
    # print(model_2)

    # bottleneck_os16 --> 50,101,152 
    # basicblock_os16 --> 18,34
    # basicblock_os8 --> 18,34



    layer5 = make_layer(Bottleneck, in_channels=4*256, channels=512, num_blocks=3, stride=1, dilation=2)

    print(layer5)