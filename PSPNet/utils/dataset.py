import os
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from PIL import Image
from natsort import natsorted
import numpy as np
from torchvision.transforms.functional import to_tensor
import torch.nn as nn

class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths,n_classes):

        self.imgDir = image_paths
        self.maskDir = target_paths
        self.n_classes = n_classes
        self.Images = natsorted(os.listdir(self.imgDir))
        self.Masks = natsorted(os.listdir(self.maskDir))
        self.to_gray = transforms.Grayscale()
        
        self.to_resize = False
        self.resize_ratio = 2
        
        #color-label mapping
        self.mapping = {
            0:0,
            220:0,
            147:1,
            72:2
            #254:1 
        }
        

    # for multi-class    
    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask==k] = self.mapping[k]
        return mask
    

    # for different size of images in same batch
    def _padding(self,image,mask):
        imgC,imgH,imgW = image.shape
        #maskH,maskW = mask.shape

        size_difference = abs(imgW - imgH)
        # print("Image shape before padding: ",image.shape) # channel,h,w
        # print("Mask shape before padding: ",mask.shape) # channel,h,w        
        if size_difference != 0:
            add_pad = nn.ZeroPad2d((0,0,size_difference,0))
            image = add_pad(image)
            mask = add_pad(mask)
            #print("Image shape after padding: ",image.shape) # channel,h,w
            #print("Mask shape after padding: ",mask.shape) # channel,h,w 
            return image,mask
        else:
            return image,mask
    

    # resize with keeping aspect-ratio
    def _resize(self,image,mask):
        
        #print(f"Image size before resize: {image.size}")
        #print(f"Mask size before resize: {mask.size}")
        
        image = image.resize((image.size[0]//self.resize_ratio,image.size[1]//self.resize_ratio))
        mask = mask.resize((mask.size[0]//self.resize_ratio,mask.size[1]//self.resize_ratio))
        
        # print(f"Image size after resize: {image.size}")
        # print(f"Mask size after resize: {mask.size}\n")
        return image,mask
    
    def __len__(self): 
        return len(self.Images)
    
    def __getitem__(self, index):

        image = Image.open(self.imgDir + self.Images[index])
        mask = Image.open(self.maskDir + self.Masks[index])
        
        if self.n_classes == 1:
            mask= self.to_gray(mask)
            image = to_tensor(image)
            mask = to_tensor(mask)

        else:
            mask=self.to_gray(mask)
            mask = np.array(mask)
            mask = self.mask_to_class(mask)
            image = to_tensor(image)
            mask = torch.from_numpy(np.array(mask))


        # if self.to_resize:
            # image,mask = self._resize(image,mask)

        # t_image,mask = self._padding(t_image,mask) #to add pad
        
        return image, mask

  