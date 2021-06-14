# YOLO v1 Object Detector implemented in PyTorch
This directory contains PyTorch implementation of YOLOv1 for object detection using Webcam.
__Why this repo?__ The code implements the real-time object detection as in paper YOLOv1. This is helpful for visualizing the performace of the detector on a custom dataset. Pretrained weights (on VOC) are obtained from the notebook and can be downloaded [here](https://drive.google.com/file/d/10BOPY6z9PCRUYvu3csjp6_cweTrZqptk/view?usp=sharing)

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

## Examples
- <img src="pytorch-yolov1-webcam/predictions/dining.png" width="448" height="448">

- <img src="https://github.com/mmalotin/pytorch-yolov3/blob/master/predictions/traffic_prediction.jpg?raw=true" width="512" height="512">

- <img src="https://github.com/mmalotin/pytorch-yolov3/blob/master/predictions/dog_prediction.jpg?raw=true" width="612" height="512">
