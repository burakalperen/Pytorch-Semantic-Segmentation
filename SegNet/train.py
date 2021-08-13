# for multiple class you can choose NLLLoss or CrossEntropyLoss for criterion
# if you'll choose NLLLoss, model(segnet.py) last layer is F.log_softmax()
# if you'll choose CrossEntropyLoss, last layer of model(segnet.py) is just output of conv, so output is raw raw logit 
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from utils.dataset import CustomDataset
from model.segnet import SegNet


def train(device,dir_img,dir_mask,net,number_class,epochs=5,batch_size=1,lr=0.001):

    dataset = CustomDataset(number_class, img_dir = dir_img, mask_dir = dir_mask)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    optimizer = optim.Adam(net.parameters(),lr=lr, weight_decay= 1e-8)
    
    if number_class == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        # Read top of code 
        #criterion = nn.NLLLoss() # last layer of model(segnet.py) should be F.log_sofmax()
        criterion = nn.CrossEntropyLoss() # last layer of model(segnet.py) is output of conv, so output is raw logit

    mean_losses = []
    for epoch in range(epochs):

        net.train()
        running_loss = []
        loop = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_idx, (img,mask) in loop:
            img = img.to(device=device)
            mask_type = torch.float32 if number_class == 1 else torch.long
            mask = mask.to(device=device,dtype=mask_type)

            mask_pred = net(img)

            loss = criterion(mask_pred,mask)

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

if __name__ == "__main__":
    img_dir = "./test_your_network_pseudo_data/images/"
    mask_dir = "./test_your_network_pseudo_data/masks/"

    number_class = 1 # if you have 1 or 2 class, number_class must be 1,  if number_class > 2, number class is must be number of class 
    numberEpoch = 150
    batch_size = 1
    lr = 0.001

    device = torch.device("cuda")

    net = SegNet(number_class=number_class, in_channel=3,out_channel=64)
    net.to(device=device)

    train(device, img_dir, mask_dir, net, number_class, numberEpoch, batch_size, lr)