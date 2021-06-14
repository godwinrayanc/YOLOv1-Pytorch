import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
from pathlib import Path
from collections import deque
import colorsys
import threading
import time
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import PIL
from PIL import Image 


CAPTURE_SIZE = (448,448)
data_transforms = transforms.Compose(
                    [
                    transforms.Resize(CAPTURE_SIZE),
                    transforms.ToTensor()
                    ]
                    )


def preprocess(image):
            
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    image = PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
                                        
    image = data_transforms(image)

    image = image.float()
             
    image = image.cuda()
    image = image.unsqueeze(0) #Resnet-50 model only 
                               #accepts 4-D Vector Tensor, so squeeze another.
    return image


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list
    
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds

def cellboxes_to_boxes(out, S=7):
    
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []


    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)
    print('BOUNDING BOXES FOUND')
    return all_bboxes

LABEL_DICT = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "dining table",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "potted plant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tv/monitor"
}

def plot_image_pred(image, boxes_pred=None, boxes_true=None, figsize=None, nimgs=1):
    """Plots predicted bounding boxes on the image"""
    """now image and boxes need to be list, first degree = number of image"""
    """each item of image/bboxes list: returned from get_batch_bboxes"""

    assert image is not list, "image should be a list of length equal to batch_size"

    if figsize is None:
        figsize=(5*nimgs,5)

    fig = plt.figure(figsize=figsize) 
    for idx_img in range(nimgs):
        ax = fig.add_subplot(1,nimgs,idx_img+1)
    
        im = np.array(image[idx_img].cpu()).transpose((1,2,0))
        height, width, _ = im.shape
        # Display the image
        ax.imshow(im)

        # box[0] is x midpoint, box[2] is width
        # box[1] is y midpoint, box[3] is height

        if boxes_true is not None:
            _boxes_true = boxes_true[idx_img]
            for box in _boxes_true:
                box_label = box[0:2] # class, confidence
                box = box[2:]
                assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
                upper_left_x = box[0] - box[2] / 2
                upper_left_y = box[1] - box[3] / 2
                rect = patches.Rectangle(
                    (upper_left_x * width, upper_left_y * height),
                    box[2] * width,
                    box[3] * height,
                    linewidth=1,
                    edgecolor="g",
                    facecolor="none",
                )
                # Add the patch to the Axes
                ax.add_patch(rect)
                props = dict(boxstyle='square', facecolor='white', color='green', alpha=1)
                ax.text(
                    x=(upper_left_x+0.0175)*width,
                    y=(upper_left_y-0.03)*height,
                    s=LABEL_DICT[box_label[0]],
                    bbox=props,
                    color='w'
                )


        if boxes_pred is not None:
            print('\nPLOTTING ON WEBCAM IMAGE')
            _boxes_pred = boxes_pred[idx_img]
            for box in _boxes_pred:
                box_label = box[0:2] # class, confidence
                box = box[2:] # 0:5
                assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
                upper_left_x = box[0] - box[2] / 2
                upper_left_y = box[1] - box[3] / 2
                rect = patches.Rectangle(
                    (upper_left_x * width, upper_left_y * height),
                    box[2] * width,
                    box[3] * height,
                    linewidth=1,
                    edgecolor="crimson",
                    facecolor="none",
                )
                # Add the patch to the Axes
                ax.add_patch(rect)
                props = dict(boxstyle='square', facecolor='crimson', alpha=1)
                ax.text(
                    x=(upper_left_x+0.0175)*width,
                    y=(upper_left_y-0.03)*height,
                    s=LABEL_DICT[box_label[0]]+f":{box_label[1]:.2f}",
                    bbox=props,
                    color='white'
                )
                print('Object and Confidence in image:',LABEL_DICT[box_label[0]]+f":{box_label[1]:.2f}")
        
    plt.show()


