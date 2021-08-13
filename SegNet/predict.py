#https://discuss.pytorch.org/t/runtimeerror-1only-batches-of-spatial-targets-supported-3d-tensors-but-got-targets-of-size-1-3-96-128/95030
import torch
import cv2
import numpy as np
from PIL import Image

from model.segnet import SegNet
from utils.dataset import CustomDataset

dataset_args = CustomDataset()

def predict(net,full_img,device,number_class):
    net.eval()
    img = dataset_args.preprocess(full_img)
    img = img.unsqueeze(0)
    img = img.to(device,dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if number_class == 1:
            output = torch.sigmoid(output)
            full_mask = output.cpu().squeeze().numpy()
            full_mask *= 255
            #full_mask = cv2.normalize(full_mask, full_mask, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U) #value of mask pixels may be 255
            return full_mask

        elif number_class > 1:
            output = net(img)
            prediction = torch.argmax(output,1)
            prediction = prediction.permute(1,2,0) #CHW -> HWC,(1024,1024,1)
            prediction = prediction.cpu().numpy()
            prediction = prediction.astype("uint8")
            prediction = np.where(prediction==0,0,prediction)
            prediction = np.where(prediction==1,147,prediction)
            prediction = np.where(prediction==2,72,prediction)
            return prediction
            # cv2.imshow("Prediction",prediction)
            # cv2.waitKey(0)

if __name__ == "__main__":
    
    number_class = 1

    
    device = torch.device("cuda")
        

    net = SegNet(number_class = number_class, in_channel = 3, out_channel = 64).to(device)
    
    net.load_state_dict(torch.load("./model.pth"))
    
    print("[INFO] Model loaded.")
    
    img = Image.open("./test_your_network_pseudo_data/images/HipHop_HipHop1_C0_00225.png").convert("RGB")
    
    
    
    mask = predict(net=net,full_img=img,device=device,number_class = number_class)
    
            
    cv2.imshow("Mask", mask.astype(np.uint8))
    cv2.waitKey(0)
   