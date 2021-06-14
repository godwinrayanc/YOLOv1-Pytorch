# YOLO v1 Object Detector implemented in PyTorch
This directory contains PyTorch implementation of YOLOv1 for object detection using Webcam.
__Why this repo?__ The code implements the real-time object detection as in paper YOLOv1. This is helpful for visualizing the performace of the detector on a custom dataset. Pretrained weights (on VOC) are obtained from the notebook and can be downloaded [here](https://drive.google.com/file/d/10BOPY6z9PCRUYvu3csjp6_cweTrZqptk/view?usp=sharing) Currently thr webcam captures an image and then plots the bounding boxes on the image with the class label and confidence score.

Download the weights and place it in the data directory.
## Requirements
- Python 3
- PyTorch
- torchvision
- OpenCV

## Usage
The repo contains one runable script for webcam detection by plots:
- Detection:
```
$ python3 webcam_plot.py
```
The real-time detection for live streaming from webcam is under construction...

## Examples
- <img src="https://github.com/godwinrayanc/YOLOv1-Pytorch/blob/a2042adc5ac0fcb4c0eebf6501bf91694451816f/pytorch-yolov1-webcam/predictions/dining.png" width="448" height="448">

- <img src="https://github.com/godwinrayanc/YOLOv1-Pytorch/blob/c5323790e9af8a6ada7223ca851ada2b9976502b/pytorch-yolov1-webcam/predictions/chair.png" width="448" height="448">

- <img src="https://github.com/godwinrayanc/YOLOv1-Pytorch/blob/c5323790e9af8a6ada7223ca851ada2b9976502b/pytorch-yolov1-webcam/predictions/sofa.png" width="448" height="448">
