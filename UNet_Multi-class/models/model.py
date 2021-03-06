import torch
import torch.nn as nn
import torch.nn.functional as F

    
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(BaseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding, stride)
        
        
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x
    
    
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(DownConv, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size, padding, stride)
    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
        return x


class UpConv(nn.Module):
    def __init__ (self, in_channels, in_channels_skip, out_channels, kernel_size, padding, stride):
        super(UpConv, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, padding=0, stride=2)
        self.conv_block = BaseConv(in_channels=in_channels + in_channels_skip, out_channels= out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        
    def forward(self, x, x_skip):
        # print("X shape: ",x.shape)
        # print("X_skip shape: ",x_skip.shape)
        x = self.conv_trans1(x)
        # print("X shape after conv_trans1: ",x.shape)
        x = torch.cat((x, x_skip), dim=1)
        # print("X shape after cat: {} X_skip shape after cat: {}".format(x.shape,x_skip.shape))
        x= self.conv_block(x)
        # print("Final x shape: ",x.shape)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_class, kernel_size, padding, stride):
        super(UNet, self).__init__()

        self.init_conv = BaseConv(in_channels, out_channels, kernel_size, padding, stride)

        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size, padding, stride)

        self.down2 = DownConv(2 * out_channels, 4 * out_channels, kernel_size, padding, stride)

        self.down3 = DownConv(4 * out_channels, 8 * out_channels, kernel_size, padding, stride)

        self.up3 = UpConv(8 * out_channels, 4 * out_channels, 4 * out_channels, kernel_size, padding, stride)

        self.up2 = UpConv(4 * out_channels, 2 * out_channels, 2 * out_channels, kernel_size, padding, stride)

        self.up1 = UpConv(2 * out_channels, out_channels, out_channels, kernel_size, padding, stride)

        self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # Decoder
        x_up = self.up3(x3, x2)
        x_up = self.up2(x_up, x1)
        x_up = self.up1(x_up, x)
        x_out = F.log_softmax(self.out(x_up), 1)
        return x_out
    

# # Create 10-class segmentation dummy image and target
# nb_classes = 2
# x = torch.randn(1, 3, 96, 96)
# y = torch.randint(0, nb_classes, (1, 96, 96))

# model = UNet(in_channels=3,
#              out_channels=64,
#              n_class=2,
#              kernel_size=3,
#              padding=1,
#              stride=1)

# if torch.cuda.is_available():
#     model = model.to('cuda')
#     x = x.to('cuda')
#     y = y.to('cuda')

# out = model(x)
# print(out.shape)

# # criterion = nn.NLLLoss()
# # optimizer = optim.Adam(model.parameters(), lr=1e-3)

# # # Training loop
# # for epoch in range(2):
# #     optimizer.zero_grad()

# #     output = model(x)
# #     loss = criterion(output, y)
# #     loss.backward()
#     optimizer.step()

#     print('Epoch {}, Loss {}'.format(epoch, loss.item()))