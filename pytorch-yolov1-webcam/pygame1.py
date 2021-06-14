import torch
from torchvision import models
 
from newutils import (plot_image_pred,non_max_suppression,plot_image,cellboxes_to_boxes,convert_cellboxes,intersection_over_union)


import torchvision 
from torch.autograd import Variable
from torchvision import transforms
import PIL 


from bbone import resnet50
#from head import Yolov1
import cv2
import argparse
#from nets.nn import resnet50
from PIL import Image 
from torch.utils.data import DataLoader
from functools import partial
import pickle
import pandas as pd
from torchvision import transforms





torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cline = argparse.ArgumentParser(description='YOLO v1 webcam detection demo')

#cline.add_argument('-obj_thold', type=float, default=0.65,
                   #help='threshold for objectness value')
#cline.add_argument('-nms_thold', type=float, default=0.4,
                   #help='threshold for non max supression')
#cline.add_argument('-model_res', type=int, default=448,
                   #help='resolution of the model\'s input')


if __name__ == '__main__':
    args = cline.parse_args()
    with torch.no_grad():
        
        BATCH_SIZE = 64 
        NUM_WORKERS = 2
        PIN_MEMORY = True



        torch.cuda.empty_cache()

        print('\nLOADING MODEL')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet50(split_size=7, num_boxes=2, num_classes=20, pretrained = False).to(device)
        
        model.load_state_dict(torch.load('data/yolov1.pth'),strict=False)
        model.eval()


        print('\nMODEL LOADING COMPLETE')

        classes = pd.read_csv('data/voc.names',header=None)

        
        # torch.cuda.empty_cache()
        # loadweights = "data/yolov1.pth"

        # model = resnet50(split_size=7,num_boxes=2,num_classes=20,pretrained=False).to(device)
    
        CAPTURE_SIZE = (448,448)
        
        data_transforms = transforms.Compose(
                    [
                        transforms.Resize(CAPTURE_SIZE),
                        #transforms.CenterCrop(),
                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0,0.225])
                    ]
                ) 
        def preprocess(image):
            image = PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
            print(image)                             
            image = data_transforms(image)
            image = image.float()
             #image = Variable(image, requires_autograd=True)
            image = image.cuda()
            image = image.unsqueeze(0) #Resnet-50 model seems to only
                               #accepts 4-D Vector Tensor so we need to squeeze another
            return image



        
        while(True):
            #_, image = cap.read()
            print('\nCAPTURE FRAME by FRAME')
            webstream = cv2.VideoCapture(0)
            _, webstream = webstream.read()

            
            x = preprocess(webstream)
            print('\nPREDICTION')
            #webstream = webstream.float()
            x = x.to(device)
            print('shape',x.shape)
            out = model(x)  
            #print(image.shape)
            
            bboxes = cellboxes_to_boxes(out)

            #print("Bounding boxes 0 len ",len(bboxes[0]))
            #print("Bounding boxes0: ",bboxes[0])

            #print("Bounding boxes len:",len(bboxes))
            #print("Bounding boxes: ",bboxes)

            bboxes = non_max_suppression(bboxes[0],iou_threshold=0.5,threshold=0.4,box_format="midpoint")
            print("Bounding boxes: ",bboxes)
            
            x = x.squeeze(0)
            #res = plot_image(webstream,bboxes,classes)
            res = plot_image_pred([x],boxes_pred=[bboxes])


