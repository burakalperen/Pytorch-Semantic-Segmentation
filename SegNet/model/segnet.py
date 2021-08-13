import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        batchNorm_momentum = 0.1

        self.EncoderConv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels,momentum=batchNorm_momentum),
            nn.ReLU()
        )

    def forward(self,x):
        return self.EncoderConv(x)

class maxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)

    def forward(self,x):
        out,index = self.pooling(x)
        return out,index



class SegNet(nn.Module):
    def __init__(self,number_class,in_channel,out_channel):
        super().__init__()
        self.number_class = number_class
        #ENCODER
        self.pooling = maxPool()
        # Stage 1
        self.stage1_1 = ConvBlock(in_channels = in_channel, out_channels = out_channel)        
        self.stage1_2 = ConvBlock(in_channels = out_channel, out_channels = out_channel)
        # Stage 2
        self.stage2_1 = ConvBlock(in_channels = out_channel, out_channels = out_channel * 2)
        self.stage2_2 = ConvBlock(in_channels = out_channel * 2, out_channels = out_channel * 2)
        # Stage 3 
        self.stage3_1 = ConvBlock(in_channels = out_channel * 2, out_channels = out_channel * 4)
        self.stage3_2 = ConvBlock(in_channels = out_channel * 4, out_channels = out_channel * 4)
        self.stage3_3 = ConvBlock(in_channels = out_channel * 4, out_channels = out_channel * 4)
        # Stage 4
        self.stage4_1 = ConvBlock(in_channels = out_channel * 4, out_channels = out_channel * 8)
        self.stage4_2 = ConvBlock(in_channels = out_channel * 8, out_channels = out_channel * 8)
        self.stage4_3 = ConvBlock(in_channels = out_channel * 8, out_channels = out_channel * 8)
        # Stage 5
        self.stage5_1 = ConvBlock(in_channels = out_channel * 8, out_channels = out_channel * 8)
        self.stage5_2 = ConvBlock(in_channels = out_channel * 8, out_channels = out_channel * 8)
        self.stage5_3 = ConvBlock(in_channels = out_channel * 8, out_channels = out_channel * 8)

        #DECODER
        # Stage 5(first step after last encoder step)
        self.dstage5_1 = ConvBlock(in_channels = out_channel * 8, out_channels = out_channel * 8)
        self.dstage5_2 = ConvBlock(in_channels = out_channel * 8, out_channels = out_channel * 8)
        self.dstage5_3 = ConvBlock(in_channels = out_channel * 8, out_channels = out_channel * 8)
        # Stage 4 
        self.dstage4_1 = ConvBlock(in_channels = out_channel * 8, out_channels = out_channel * 8)
        self.dstage4_2 = ConvBlock(in_channels = out_channel * 8, out_channels = out_channel * 8)
        self.dstage4_3 = ConvBlock(in_channels = out_channel * 8, out_channels = out_channel * 4)
        # Stage 3
        self.dstage3_1 = ConvBlock(in_channels = out_channel * 4, out_channels = out_channel * 4)
        self.dstage3_2 = ConvBlock(in_channels = out_channel * 4, out_channels = out_channel * 4)
        self.dstage3_3 = ConvBlock(in_channels = out_channel * 4, out_channels = out_channel * 2)
        # Stage 2
        self.dstage2_1 = ConvBlock(in_channels = out_channel * 2, out_channels = out_channel * 2)
        self.dstage2_2 = ConvBlock(in_channels = out_channel * 2, out_channels = out_channel)
        # Stage 1
        self.dstage1_1 = ConvBlock(in_channels = out_channel, out_channels = out_channel)
        self.dstage1_2 = nn.Conv2d(in_channels = out_channel, out_channels = number_class, kernel_size = 3, padding = 1)


    def forward(self,x):
        #print("Input: ",x.shape)
        x11 = self.stage1_1(x)
        x12 = self.stage1_2(x11)
        #print("Stage1 conv out: ",x12.shape)
        dim1 = x12.size()
        x1,idx1 = self.pooling(x12)
        #print(f"Stage1 pooling out: {x1.shape} and indexes: {idx1.shape}")
        x21 = self.stage2_1(x1)
        x22 = self.stage2_2(x21)
        #print("Stage2 conv out: ",x22.shape)
        dim2 = x22.size()
        x2,idx2 = self.pooling(x22)
        #print(f"Stage2 pooling out: {x2.shape} and indexes: {idx2.shape}")
        x31 = self.stage3_1(x2)
        x32 = self.stage3_2(x31)
        x33 = self.stage3_3(x32)
        #print("Stage3 conv out: ",x33.shape)
        dim3 = x33.size()
        x3,idx3 = self.pooling(x33)
        #print(f"Stage3 pooling out: {x3.shape} and indexes: {idx3.shape}")
        x41 = self.stage4_1(x3)
        x42 = self.stage4_2(x41)
        x43 = self.stage4_3(x42)
        #print("Stage4 conv out: ",x43.shape)
        dim4 = x43.size()
        x4,idx4 = self.pooling(x43)
        #print(f"Stage4 pooling out: {x4.shape} and indexes: {idx4.shape}")
        x51 = self.stage5_1(x4)
        x52 = self.stage5_1(x51)
        x53 = self.stage5_1(x52)
        #print("Stage5 conv out: ",x53.shape)
        dim5 = x53.size()
        x5,idx5 = self.pooling(x53)
        #print(f"Stage5 pooling out: {x5.shape} and indexes: {idx5.shape}")

        # Decoder
        #print("\n***** DECODER PART *****")
        x5d = F.max_unpool2d(x5, idx5, kernel_size = 2, stride = 2, output_size = dim5)
        #print("Unpool stage5: ",x5d.shape)
        x51d = self.stage5_1(x5d)
        x52d = self.stage5_2(x51d)
        x53d = self.stage5_3(x52d)
        #print("Decoder stage5 out: ",x53d.shape)
        x4d = F.max_unpool2d(x53d, idx4, kernel_size = 2, stride = 2, output_size = dim4)
        #print("Unpool stage4: ",x4d.shape)
        x41d = self.dstage4_1(x4d)
        x42d = self.dstage4_2(x41d)
        x43d = self.dstage4_3(x42d)
        #print("Decoder stage4 out: ",x43d.shape)
        x3d = F.max_unpool2d(x43d, idx3, kernel_size = 2, stride = 2, output_size = dim3)
        #print("Unpool stage3: ",x3d.shape)
        x31d = self.dstage3_1(x3d)
        x32d = self.dstage3_2(x31d)
        x33d = self.dstage3_3(x32d)
        #print("Decoder stage3 out: ",x33d.shape)
        x2d = F.max_unpool2d(x33d, idx2, kernel_size = 2, stride = 2, output_size = dim2)
        #print("Unpool stage2: ",x2d.shape)
        x21d = self.dstage2_1(x2d)
        x22d = self.dstage2_2(x21d)
        #print("Decoder stage2 out: ",x22d.shape)
        x1d = F.max_unpool2d(x22d, idx1, kernel_size = 2, stride = 2, output_size = dim1)
        #print("Unpool stage1: ",x1d.shape)
        x11d = self.dstage1_1(x1d)
        #print(x11d.shape)
        x12d = self.dstage1_2(x11d)
        #print("Decoder stage1 out: ",x12d.shape)
        if self.number_class == 1:
            return x12d
        elif self.number_class != 1:
            #out = F.log_softmax(x12d,dim=1) # for NLLLoss 
            #return out # for NLLLoss
            return x12d  # for CrossEntropyLoss



# model = SegNet(number_class = 4, in_channel = 3,out_channel = 64)

# inp = torch.rand(1,3,200,200)

# out,without_softmax = model(inp)

# print(without_softmax)
# print(out)

# # print(without_softmax.shape)
# # print(out.shape)

