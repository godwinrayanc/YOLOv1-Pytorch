#Import dependencies
import torch
import torchvision
import torch.nn as nn


def resnet50(split_size, num_boxes, num_classes, pretrained):
    S, B, C = split_size, num_boxes, num_classes

    #Import Resnet50 from pytorch
    model = torchvision.models.resnet50(pretrained = pretrained)
  
    #Enable backprop for all layers
    for param in model.parameters():
        param.requires_grad = True
  
    #Remove FC layers
    modules=list(model.children())[:-2]
    model=nn.Sequential(*modules)

    #Add custom CNN layer
    model.avgpool = nn.AdaptiveAvgPool2d((7,7))
    model = nn.Sequential(model, 
               nn.Conv2d(2048, 1024, kernel_size=(1,1)))

    #Add custom FC layers      
    model.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + B * 5)),)

    return model


def vgg16(split_size, num_boxes, num_classes, pretrained):
    S, B, C = split_size, num_boxes, num_classes
    #Import Resnet50 from pytorch
    model = torchvision.models.vgg16(pretrained = pretrained)

    modules=list(model.children())[:-2]
    model=nn.Sequential(*modules)
    
    #Enable backprop for all layers
    for param in model.parameters():
        param.requires_grad = True

    #Add custom pooling layer
    model.avgpool = nn.AdaptiveAvgPool2d((7,7))

    #Add custom FC layers 
    model.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + B * 5)),)

    return model



