from torchvision import transforms
from torch.utils.data import Dataset
import os 
from PIL import Image
from natsort import natsorted
import torchvision


class MyDataset(Dataset):
    
    def __init__(self,imgs_dir=None,masks_dir=None):
        super().__init__()
        
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.inputImages = natsorted(os.listdir(self.imgs_dir))
        self.targetImages = natsorted(os.listdir(self.masks_dir))
        self.to_gray = transforms.Grayscale()
        self.to_tensor = transforms.ToTensor()
        
        self.to_resize = True
        self.resize_ratio = 16
        
        
    def __len__(self):
        return len(self.inputImages)
    
    @classmethod
    def preprocess(cls,image):
        tf = torchvision.transforms.ToTensor()
        image = tf(image)
        return image

    def _resize(self,image,mask):
        #print("Before shapes: ",image.size,mask.size)
        image = image.resize((image.size[0]//self.resize_ratio,image.size[1]//self.resize_ratio))
        mask = mask.resize((mask.size[0]//self.resize_ratio,mask.size[1]//self.resize_ratio))
        #print("After shapes: ",image.size,mask.size)
        return image,mask
    
    def __getitem__(self, index):
        imgPath = os.path.join(self.imgs_dir,self.inputImages[index])
        maskPath = os.path.join(self.masks_dir,self.targetImages[index])
        img = Image.open(imgPath)
        mask = Image.open(maskPath)
        
        if self.resize_ratio:
            img,mask = self._resize(img,mask)
        
        mask = self.to_gray(mask)
        
        assert img.size == mask.size, "Image and mask size should be the same."


        img = self.preprocess(img)
        mask = self.preprocess(mask)    
        
        return [img,mask]
    