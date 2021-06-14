import torch
from torchvision import models
from newutils import non_max_suppression
from newutils import (create_class_list, 
    plot_image ,
    cellboxes_to_boxes, 
    convert_cellboxes, 
    intersection_over_union,
    non_max_suppression
    )


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

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        
        
        

        while(True):
            #_, image = cap.read()
            print('\nLOADING DATASET STEP 1')
            webstream = cv2.VideoCapture(0)

            _, webstream = webstream.read()
            #webstream = cv2.cvtColor(webstream, cv2.COLOR_BGR2RGB)
            webstream = cv2.resize(webstream,(448,448))
            
            webstream = torch.tensor(webstream)
            print("Format of webstream",webstream.type())
            

            print('\nLOADING DATASET STEP 2')
            web_loader = DataLoader(
                dataset=webstream,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                shuffle=True,
                drop_last=True,
                )
            
            #image_location = 'images/dog.jpg'

            
            # image = torch.tensor(image)
            #print(next(enumerate(web_loader)))
            
            batch_idx,x = next(enumerate(web_loader))
            x = x.to(device)
            #labels = labels.to(device)
            
            #start_time = time.time()

            print('\nPREDICTION')
            
            x = x.unsqueeze(1)

            out = model(x)  
            #print(image.shape)
            
            bboxes = cellboxes_to_boxes(out)
            bboxes = non_max_suppression(bboxes,iou_threshold=0.5,threshold=0.4,box_format="midpoint")
            res = plot_image(image,bboxes,classes)
            
            cv2.imshow('webcam', res)
            k = cv2.waitKey(100)
            if k == 27:                            # Press Esc to quit
            break
            cap.release()
            cv2.destroyAllWindows()
