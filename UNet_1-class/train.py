# https://github.com/milesial/Pytorch-UNet
# all images must be same shape
# if resize input images, keep aspect-ratio
import sys
import os
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataloader
from tqdm import tqdm

from unet.unet_model import UNet

from utils.dataset import MyDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train_net(device,dir_img,dir_mask,net,epochs=5,batch_size=1,lr=0.001):
    
    dataset = MyDataset(imgs_dir=dir_img,masks_dir=dir_mask)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    #writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')

    optimizer = optim.Adam(net.parameters(),lr=lr, weight_decay= 1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 1, factor = 0.1,verbose=True)    
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    mean_losses = []
    for epoch in range(epochs):
        
        net.train()
        running_loss = []
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (img,mask) in loop:
            img = img.to(device=device,dtype = torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            mask = mask.to(device=device,dtype=mask_type)
            
            mask_pred = net(img)
            
            loss = criterion(mask_pred, mask)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)
            
            loop.set_description(f"Epoch: [{epoch + 1}/{epochs}]")
            loop.set_postfix(batch_loss = loss.item(), mean_loss = mean_loss, lr = optimizer.param_groups[0]["lr"])

        if len(mean_losses) >= 1: 
            if mean_loss < min(mean_losses):
                print("Model saved.")
                torch.save(net.state_dict(), "model.pth") 
            
        mean_losses.append(mean_loss)
        scheduler.step(mean_loss)
    
    
if __name__ == "__main__":
    
    img_dir = "./train_data/images/"
    mask_dir = "./train_data/masks/"

    numberEpoch = 750
    batch_size = 4
    lr = 0.001
    
    device = torch.device("cuda")
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N  
    net = UNet(n_channels=3,n_classes=1,bilinear=True)
    net.to(device=device)
    
    
    try:
        train_net(device=device,
                  dir_img = img_dir, dir_mask = mask_dir,
                  net = net,epochs = numberEpoch,
                  batch_size = batch_size,lr = lr)
        print("[INFO] Training is ended.")
    except KeyboardInterrupt:
        torch.save(net.state_dict(),'INTERRUPTED.pth')
        
        try:
            sys.exit(0)
        except SystemExit:
            os.exit(0)
