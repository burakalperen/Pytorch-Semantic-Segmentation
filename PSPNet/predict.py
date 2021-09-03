import numpy as np
import cv2
from PIL import Image
import torch
from network.PSPNet import PSPNet
import torchvision.transforms as transforms

# yazım hatalarına bak unet 1 class ta predidct normalize ederken birinde uint8 olmuyor onu ayarla
# prediction'ı np.where ile değil self.mapping ile yap

def predict(model,img,device,n_classes):
    to_tensor = transforms.ToTensor()

    model = model.to(device).eval()
    img = to_tensor(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output  = model(img) #aux didn't come from model, while training parameter is False
        if n_classes == 1:
            output = torch.sigmoid(output)
            mask = output.cpu().numpy()
            # mask *= 255
            # mask = mask.astype(np.uint8)
            mask = cv2.normalize(mask, mask, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
            return mask 

        else:
            output = torch.softmax(output,dim = 1)
            mask = torch.argmax(output,1)
            mask = mask.permute(1,2,0) # CHW -> HWC
            mask = mask.cpu().numpy()
            mask = mask.astype("uint8")
            #reverse of label_color map in dataset.py
            label_color_map = {
              0:0,
              1:147,
              2:72  
            }
            for k in label_color_map:
                mask[mask == k] = label_color_map[k]
            return mask



if __name__ == "__main__":
    n_classes = 3
    use_aux = True #if use auxiliary loss


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PSPNet(layers=50,num_classes = n_classes,training=False,pretrained=False,use_aux=use_aux)
    
    if not use_aux:
        model.load_state_dict(torch.load("./checkpoints/deneme_aux_multiclass.pth"))

    elif use_aux:
        #state_dict'den auxiliary olan yerleri silmen lazım
        state_dict = torch.load("./checkpoints/deneme_aux_multiclass.pth")
        update_dict = state_dict.copy()
        # delete keys that consist of "aux"
        for k in state_dict:
            if "aux" in k:
                del update_dict[k]
        #load new state dict without aux weights and biases.
        # With strict=False, load state dict function ignore non-matching keys(state dict keys and weights) 
        model.load_state_dict(update_dict,strict=False)

    img = Image.open("./data/images/2.jpg")

    mask = predict(model,img,device,n_classes = n_classes)

    cv2.imshow("Prediction",mask)
    cv2.waitKey(0)