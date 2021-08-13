from torchvision import transforms
from torch.utils.data import Dataset
import os 
from PIL import Image
from natsort import natsorted
import torchvision
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self,number_class = None,img_dir = None,mask_dir = None):
        super().__init__()

        self.number_class = number_class
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.inputImages = natsorted(os.listdir(img_dir))
        self.masks = natsorted(os.listdir(mask_dir))
        self.to_gray = transforms.Grayscale()
        self.to_tensor = transforms.ToTensor()
        
        # if number_class > 1
        self.map = {
            0:0,
            220:0,
            147:1,
            72:2
            #254:1 
        }

    @classmethod
    def preprocess(cls,image):
        tf = torchvision.transforms.ToTensor()
        image = tf(image)
        return image

    def mapping(self,mask):
        for k in self.map:
            mask[mask==k] = self.map[k]
        return mask


    def __getitem__(self, index):
        imgPath = os.path.join(self.img_dir,self.inputImages[index])
        maskPath = os.path.join(self.mask_dir,self.masks[index])
        img = Image.open(imgPath)
        mask = Image.open(maskPath)
        
        mask = self.to_gray(mask)
        
        if self.number_class > 1:
            mask = np.array(mask)
            mask = self.mapping(mask)
            mask = torch.from_numpy(np.array(mask))
            img = self.to_tensor(img)

        else:
            img = self.preprocess(img)
            mask = self.preprocess(mask)
        #assert img.size == mask.size, "Image and mask size should be the same."
        
        return img,mask

    def __len__(self):
        return len(self.inputImages)