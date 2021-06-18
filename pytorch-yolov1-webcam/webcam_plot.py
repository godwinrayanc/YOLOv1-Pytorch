


#Import dependencies for the opencv, load model and plotting functions

import torch
from torchvision import models

from newutils import (preprocess,
    plot_image_pred,
    non_max_suppression,
    cellboxes_to_boxes,
    convert_cellboxes,
    intersection_over_union)

from torchvision import transforms
from bbone import resnet50
import cv2
import PIL
from PIL import Image 
from functools import partial
import pickle
import pandas as pd
from torchvision import transforms




#clear gpu memory
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    
    with torch.no_grad():
        
        BATCH_SIZE = 64 
        NUM_WORKERS = 2
        PIN_MEMORY = True


        print('\nLOADING MODEL')

        #define device to gpu and load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = resnet50(split_size=7, num_boxes=2, num_classes=20, pretrained = False).to(device)
        
        #load pretrained weights from data directory
        model.load_state_dict(torch.load('data/yolov1.pth'),strict=False)
        model.eval()


        print('\nMODEL LOADING COMPLETE')

        NUM_FRAMES=200
        for i in range(NUM_FRAMES):
        #while(True):
            
            print('\nCAPTURE FRAME by FRAME')
            webstream = cv2.VideoCapture(0)
            _, webstream = webstream.read()


            #preporcess webcam to pass through CNN
            x = preprocess(webstream)
            
            print('\nPREDICTION')
            print('\nCLOSE WINDOW TO CLICK ANOTHER PICTURE!')
            x = x.to(device)
            out = model(x)  
            
            bboxes = cellboxes_to_boxes(out)

            bboxes = non_max_suppression(bboxes[0],iou_threshold=0.5,threshold=0.4,box_format="midpoint")
            
            #Convert 4D tensor back to 3D tensor for prediction
            x = x.squeeze(0)
            res = plot_image_pred([x],boxes_pred=[bboxes])


