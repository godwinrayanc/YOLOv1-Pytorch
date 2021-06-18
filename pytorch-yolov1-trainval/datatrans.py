#   Required transforms:
#       resize
#       random horizon flip
#       color jitter
#       random affine
#       to tensor

import torch 
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

#from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)

        return img, bboxes

class Resize(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img, bboxes):
        img = TF.resize(img, self.size)
        img = TF.to_tensor(img)
        return img, bboxes

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    
    def forward(self, img, bboxes):
        if random.random() <= self.p:
            img = TF.hflip(img)
            for bbox in bboxes:
                bbox[1] = 1 - bbox[1]
            
        return img, bboxes


class ColorJitter(torch.nn.Module):
    def __init__(self, brightness, saturation):
        super().__init__()
        self.transform = transforms.ColorJitter(
            brightness=brightness,
            saturation=saturation
        )
    
    def forward(self, img, bboxes):
        return self.transform(img), bboxes


class RandomAffine(torch.nn.Module):
    def __init__(self, translate=None, scale=None):
        super().__init__()
        self.translate = translate
        self.scale = scale

    def forward(self, img, bboxes):
        img_size = TF._get_image_size(img)
        if self.translate is not None:
            max_dx = float(self.translate[0])*img_size[0]
            max_dy = float(self.translate[1])*img_size[1]
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)

        else:
            translations = (0,0)
        if self.scale is not None:
            scale = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
        else:
            scale = 1.0
        
        img = TF.affine(
            img, 
            translate=translations, 
            scale=scale,
            angle=0,
            shear=0
        )

        for b, box in enumerate(bboxes):
            x = bboxes[b,1]
            y = bboxes[b,2]
            x_prime = x - 0.5
            y_prime = y - 0.5
            
            x_prime = scale*x_prime + translations[0]/img_size[0]
            y_prime = scale*y_prime + translations[1]/img_size[1]

            bboxes[b,1] = x_prime + 0.5
            bboxes[b,2] = y_prime + 0.5
            bboxes[b,3] = bboxes[b,3]*scale
            bboxes[b,4] = bboxes[b,4]*scale

        # bbox_list = []
        # for col in range(7):
        #     for row in range(7):
        #         if bboxes[col, row, 20] != 0:
        #             bboxes[col, row, 21] += col
        #             bboxes[col, row, 22] += row
        #             bbox_list = [bbox_list, bboxes[col,row,:]]

        # bboxes = torch.zeros_like(bboxes)
        # for bbox in bbox_list:
        #     bbox[21] = scale[0]*bbox[21]+tx
        #     bbox[22] = scale[1]*bbox[22]+ty
        #     col = floor(bbox[21])
        #     row = floor(bbox[22])
        #     bbox[21] += -col
        #     bbox[22] += -row
        #     bboxes[col, row, :] = bbox
        
        return img, bboxes

class ToTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.ToTensor()
    
    def forward(self, img, bboxes):
        if isinstance(img, torch.Tensor):
            return img, bboxes
        else:
            return self.transform(img), bboxes