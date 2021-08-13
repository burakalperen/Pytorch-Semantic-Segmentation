import numpy as np
import cv2

import torch
from PIL import Image

from unet.unet_model import UNet
from utils.dataset import MyDataset

dataset_args = MyDataset()

def predict_img(net,full_img,device):
    net.eval()
    img = MyDataset.preprocess(full_img)
    img = img.unsqueeze(0)
    img = img.to(device,dtype=torch.float32)
    
    
    
    
    with torch.no_grad():
        output = net(img)
        if net.n_classes == 1:
            output = torch.sigmoid(output) #because we have one class
        else:
            output = torch.softmax(output,dim=1) #for multiple class    
    
    full_mask = output.cpu().squeeze().numpy()    
    
    if dataset_args.to_resize:
        full_mask = cv2.resize(full_mask,(full_mask.shape[1]*dataset_args.resize_ratio,full_mask.shape[0]*dataset_args.resize_ratio))
    
    
    full_mask *= 255  #value of mask pixels may be 254 
    #full_mask = cv2.normalize(full_mask, full_mask, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U) #value of mask pixels may be 255

    return full_mask 

if __name__ == "__main__":
    
    device = torch.device("cuda")
        
    net = UNet(n_channels=3,n_classes=1).to(device)
    
    net.load_state_dict(torch.load("./model_deneme_all_16.pth"))
    
    print("[INFO] Model loaded.")
    
    img = Image.open("./train_data/1_hiphop.png")
    
    if dataset_args.to_resize:
        img = img.resize((img.size[0]//dataset_args.resize_ratio,img.size[1]//dataset_args.resize_ratio))

    
    mask = predict_img(net=net,full_img=img,device=device)
    
            
    cv2.imshow("Mask", mask.astype(np.uint8))
    cv2.waitKey(0)
