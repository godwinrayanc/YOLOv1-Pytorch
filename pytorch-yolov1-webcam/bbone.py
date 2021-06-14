import torch
import torchvision
import torch.nn as nn

def resnet50(split_size, num_boxes, num_classes, pretrained):
    S, B, C = split_size, num_boxes, num_classes
    model = torchvision.models.resnet50(pretrained = pretrained)
  
    for param in model.parameters():
        param.requires_grad = True

    for (name, module) in model.named_children():
        if name == 'layer4':
            for layer in list(module.children()):
                for param in layer.parameters():
                    param.requires_grad = True
  
    modules=list(model.children())[:-2]
    model=nn.Sequential(*modules)

    model.avgpool = nn.AdaptiveAvgPool2d((7,7))
    model = nn.Sequential(model, 
               nn.Conv2d(2048, 1024, kernel_size=(1,1)))
               #nn.Conv2d(1024, 1024, kernel_size=(3,3), padding=(1,1)),
               #nn.Conv2d(1024, 1024, kernel_size=(3,3), padding=(1,1)),
               #nn.Conv2d(1024, 1024, kernel_size=(3,3), padding=(1,1)))
    model.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + B * 5)),)

    return model