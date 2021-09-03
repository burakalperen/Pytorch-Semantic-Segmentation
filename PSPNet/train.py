from utils.dataset import CustomDataset
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from network.PSPNet import PSPNet


def train(model,img_path,mask_path,n_classes,batch_size,epochs,lr,use_aux):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_dataset = CustomDataset(img_path,mask_path,n_classes = n_classes)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle = True,num_workers = 8,drop_last = True)
    
    if n_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        #criterion = nn.NLLLoss() # last layer of network(PSPNet.py) should be F.log_sofmax()
        criterion = nn.CrossEntropyLoss() # last layer of network(PSPNet.py) is output of conv, so output is raw logit

    optimizer = optim.Adam(model.parameters(),lr = lr, weight_decay=1e-8)

    model.train()
    mean_losses = []
    print("[INFO] Traning is started.")
    for epoch in range(epochs):
        running_loss = []
        loop = tqdm(enumerate(train_loader), total=len(train_loader)) 

        for idx, (img,mask) in loop:
            img = img.to(device)
            mask_type = torch.float32 if n_classes == 1 else torch.long
            target = mask.to(device,dtype = mask_type)

            # update
            if use_aux:
                mask_prediction,aux = model(img)
                loss1 = criterion(mask_prediction,target)
                loss2 = criterion(aux,target)
                loss = loss1 + 0.4 * loss2 #usually 0.4 is weight of auxillary classifier, but 0.4 is stated in PSPNet paper. 
            
            elif not use_aux:
                mask_prediction = model(img)
                loss = criterion(mask_prediction,target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)

            loop.set_description(f"Epoch: [{epoch + 1} / {epochs}]")
            loop.set_postfix(batch_loss = loss.item(), mean_loss = mean_loss, lr = optimizer.param_groups[0]["lr"])


        if len(mean_losses) >= 1:
            if mean_loss < min(mean_losses):
                print("[INFO] Model saved.")
                torch.save(model.state_dict(),"./checkpoints/deneme_aux_multiclass.pth")
        mean_losses.append(mean_loss)



if __name__ == "__main__":
    
    """ARGS"""
    img_path = "./data/images/"
    mask_path = "./data/masks/"
    n_classes = 3
    lr  = 0.001
    batch_size = 2
    epoch = 150
    use_aux = True #auxiliary loss
    
    # for resnet50, make layers = 50
    # for resnet101, make layers = 101
    # for resnet152, make layers = 152
    model = PSPNet(layers = 50,num_classes=n_classes,training = True,use_aux = use_aux)
    
    train(model,img_path,mask_path,
            n_classes,batch_size,epochs=epoch,
            lr = lr, use_aux = use_aux)

    print("[INFO] Training is ended.")
