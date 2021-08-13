import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt 
import torchvision
import cv2
   
from models.model import UNet 
   

def predict_mask(output):
    
    prediction = torch.argmax(output,1)
    prediction = prediction.permute(1,2,0) #CHW -> HWC,(1024,1024,1)
    prediction = prediction.cpu().numpy()

    # num_zeros = (prediction==0).sum()
    # num_ones = (prediction==1).sum()
    # num_twos = (prediction==2).sum()

    prediction = prediction.astype("uint8")
    prediction = np.where(prediction==0,0,prediction)
    prediction = np.where(prediction==1,72,prediction)
    prediction = np.where(prediction==2,247,prediction)
    cv2.imshow("Prediction",prediction)
    cv2.waitKey(0)


def show_probability_map(output):
    
    # slice output channels of prediction, show probability map for each classes
    output = output.cpu()
    #prob = F.softmax(output,1) 
    prob = torch.exp(output) # we're using log_softmax in model, so apply torch.exp to get probabilities
    prob_imgs = torchvision.utils.make_grid(prob.permute(1, 0, 2, 3))
    plt.imshow(prob_imgs.permute(1, 2,0))
    plt.show()

    # MORE THAN ONE BATCH, probability map for each classes
    # prob = F.softmax(output, 1)
    # prob = torch.exp(output)
    # for p in prob:
    #     prob_imgs = torchvision.utils.make_grid(p.unsqueeze(1))
    #     plt.imshow(prob_imgs.permute(1, 2, 0))
    #     plt.show()

def make_square(img):
    img = cv2.copyMakeBorder(
            img,
            top=43,
            bottom=0,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
    return img


if __name__ == "__main__":
    
    to_tensor = transforms.ToTensor()
    to_gray = transforms.Grayscale()

    org_image = Image.open("./DATA/...")

    image = to_gray(org_image)
    image = np.array(image)
    image = make_square(image)
    image = to_tensor(image)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = UNet(in_channels=1,
                out_channels=64,
                n_class=3,
                kernel_size=3,
                padding=1,
                stride=1)

    model = model.to(device)
    model.load_state_dict(torch.load("./model.pth"))
    model.eval()

    with torch.no_grad():
        image = image.unsqueeze(0) #(1,1,1024,1024)
        image = image.to(device)
        print(image.shape)
        output = model(image)  # (1,3,1024,1024), 3 is number of classes, mapping must be number_classes - 1: 0,1,2
    
    predict_mask(output)
    #show_probability_map(output)

    # SIDE NOT
    # x = torch.randn(1, 3, 24, 24)
    # output = F.log_softmax(x, 1) #last process of model
    # print('LogProbs: ', output)
    # print('Probs: ', torch.exp(output))
    # print('Prediction: ', torch.argmax(output, 1))

