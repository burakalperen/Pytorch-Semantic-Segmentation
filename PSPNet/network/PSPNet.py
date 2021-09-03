import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.models  
import torchvision
import torch.nn.functional as F
import torch
import torch.optim as optim
from torchvision.models import resnet

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module,nn.Conv2d) or isinstance(module,nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module,nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _PyramidPoolingModule(nn.Module):
    def __init__(self,in_dim,reduction_dim,setting):
        super(_PyramidPoolingModule,self).__init__()
        self.features = []
         # 4 ayrı pooling için sequential moduller oluşturuyoruz
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        #sequential modülleri bir listede tutuyoruz
        self.features = nn.ModuleList(self.features)
        
    def forward(self,x):
        x_size = x.size()
        out = [x] # En son 4 pooling çıktısını ve feature map'i concatlıcaz. O yüzden feature map'i out içerisinde tutuyoruz.
        for f in self.features:
            pyramid_element  = F.interpolate(f(x),x_size[2:],mode="bilinear",align_corners=True)
            out.append(pyramid_element) # pooling yapılan tensorler upsample edilip concat için out'a atılıyor
        out = torch.cat(out,1) # en son hepsi channel seviyesinde concat ediliyor (2,4096,12,12)
        return out

   
        
class PSPNet(nn.Module):
    def __init__(self,layers,num_classes,training,pretrained=False,use_aux=False,dropout = 0.1):
        super(PSPNet,self).__init__()
        
        self.training = training
        self.use_aux = use_aux

        if layers == 50:
            resnet = torchvision.models.resnet50()
        elif layers == 101:
            resnet = torchvision.models.resnet101() # resnet50,resnet101 veresnet 152 de çalışıyor
        elif layers == 152:
            resnet = torchvision.models.resnet152()

        if pretrained:
            resnet.load_state_dict(torch.load("path"))
        
       
        
        self.layer0 = nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu,resnet.maxpool)
        self.layer1, self.layer2, self.layer3,self.layer4 = resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4
        

        # normalde (1,3,96,96) input için layer3'te (1,1024,6,6), layer4'te (1,2048,3,3) çıktısını alıyoruz.
        # bu çıktılar resmi çok küçülttüğü için layer 3 ve layer4 ü güncelliyoruz.
        # yeni update ile (1,3,96,96) inputta layer3 çıktısı (1,1024,12,12), layer4 çıktısı (1,2048,12,12) oluyor. 
        for n,m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2,2), (2,2), (1,1)
            elif 'downsample.0' in n:
                m.stride = (1,1)
        for n,m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2,2), (2,2), (1,1)
            elif 'downsample.0' in n:
                m.stride = (1,1)

        self.ppm = _PyramidPoolingModule(2048,512,(1,2,3,6)) # (input_dim of pyramid,out_dim of pyramid,(pyramid pooling sizes))
        self.final = nn.Sequential(
            nn.Conv2d(4096,512,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(512,momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512,num_classes,kernel_size=1)
        )


        if self.training:
            if self.use_aux:
                self.aux = nn.Sequential(
                    nn.Conv2d(1024,256,kernel_size=3,padding=1,bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout),
                    nn.Conv2d(256,num_classes,kernel_size=1)
            )


        # if self.training:
        #     if use_aux:
        #         self.aux_logits = nn.Conv2d(1024,num_classes,kernel_size=1)
        #         initialize_weights(self.aux_logits)
        
        #initialize_weights(self.ppm,self.final)
    

    def forward(self,x):
        x_size = x.size()
        #the following values is calculated for 1,3,96,96 input size
        x = self.layer0(x)#(1,64,24,24)
        x = self.layer1(x)#(1,256,24,24)
        x = self.layer2(x)#(1,512,12,12)
        x_temp = self.layer3(x)#(1,1024,12,12)
        x = self.layer4(x_temp) # (1,2048,12,12)
        #print("After resnet101: ",x.shape) 

        x = self.ppm(x)
        # print("Concat after ppm: ",x.shape)
        
        x = self.final(x)
        # print("After final layer: ",x.shape)

        x = F.interpolate(x, size=(x_size[2],x_size[3]), mode='bilinear', align_corners=True)

        if self.training and self.use_aux:
            aux = self.aux(x_temp)
            aux = F.interpolate(aux, size = (x_size[2],x_size[3]),mode = "bilinear", align_corners=True)
            return x,aux

        return x





# DUMMY
if __name__ == "__main__":
    n_classes = 2
    model = PSPNet(layers = 101,num_classes=n_classes,training = True,use_aux = True)
    
    x = torch.randn(2, 3, 120, 120)
    
    print("Input shape: ",x.shape)
    y,aux = model(x)
    print("Output shape: ",y.shape) #(batch_Size,num_classes,img_height,img_width)
    print("Aux shape: ",aux.shape) #(batch_Size,num_classes,img_height,img_width)
  