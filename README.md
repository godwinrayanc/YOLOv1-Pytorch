# You Only Look Once (YOLO V1) with PyTorch

Object Identification involves detecting objects in an image and identifying the classes. Before YOLO(You Only Look Once), object detectors were quite slow as it involves extraction of regions in an image preceding classification. Slower Frames Per Second(FPS) are expensive for object detection in applications such as autonomous driving or fast moving robots. 

Regional Convolutional Neural Network (R-CNN) and its variants such as Fast R-CNN and Faster R-CNN improved accuracy but still managed to reach only 7 FPS [1].  To overcome such difficulties, You Only Look Once (YOLO) was proposed, which can be trained end to end and achieve higher frames/second in real-time. The idea behind YOLO is to look only once in an image rather than looking at multiple regions one by one.

This blog relates to the reproduction of YOLO V1 originally developed by Joseph Redmon *et al*. The convolutional neural network was built with C programming language when the research was published. We aim to implement YOLO V1 using PyTorch to train, test to achieve results as in Table 1 in [2]. Additionally, the results are visualized with webcam images as dataset. The primary objective is to understand the classic architecture of YOLO V1 to dive deeper into Object detection. Additionally, choosing this project will provide hands-on experience with PyTorch and OpenCV framework to execute Object-Detection Projects. All codes for this reproduction is in this [Github Repository](https://github.com/godwinrayanc/YOLOv1-Pytorch).

---

<!-- Object detection is the task to predict bounding boxes of objects in a image and identify its class. Due to the complexity of images, this task has been tricky with traditional techniques. However, over the last two decades, enormous amount of work has been achieved in this field. A few of the most noticeable are Regional Convolutional Neural Network (RCNN), Faster RCNN, and You-Only-Look-Once (YOLO) v1 to v5. Very recently, with Transformers being applied in vision field, the state-of-the-art mean average precision (mAP) were refreshed over and over again, examples of which are DETR and Swin Transformers.

In this blog post, however, we put our eyes back on a classic architecture, Yolo v1, as our touching stone to dig deeper into object detection. We will first briefly introduce Yolo v1 about its formation, loss function and network design. With that being explained, we will present the result of building and training Yolo v1 from scratch, based on which we will close with a conlucion and discussion section.  -->

## 1 What is YOLO?

You Only Look Once: an unified object detection method that was proposed in [2]. YOLO uses a unique problem formation. Instead of proposing many possible regions of interest, YOLO splits an image into SxS grid (called cells) as in Fig. 1. At each cell the network output B bounding boxes. Although implicitly assuming there is only one class of object that is centered in each cell, it reduces the computation complexity greatly. Furthermore, to facilitate the training, bounding box coordinates are represented by their center point's coordinate $(x,y)$, and the width $w$ and height $h$, all of which are relative to the cell's width and height. This means if the image is resized (streched and the resolution changed, not scaled but keeping the resolution the same), these coordinates will remain the same. 

![](https://i.imgur.com/VuUAwKP.png)

*Figure 1. Yolo splits image to cells*

### 1.1 Loss Function
Object detection task includes both Detection (Regression of bounding box coordinates) and Classification, which means the loss of multiple goals need to be well structured and balanced. The loss contains two parts, bounding box prediction loss and classification loss. 

To derive bounding box prediction loss, we first need to define two concepts, "exists" and "responsible". We call an object "exists" in cell $i$ if its centre point falls in cell $i$. The network will predict B(variable for the number of bounding boxes) bounding boxes in each cell, whether or not there exists an object. When there does exist an object in a cell, then out of these B bounding boxes, the one with the highest IoU (intersection over union) with the ground truth is called "responsible". For the bounding box prediction loss, it is only defined on all the responsible bounding boxes. The loss's form is squared error. Furthermore, to balance small and big objects, the object width and height are first square rooted before calculating the square error. This part of the loss is shown in Eq. 1. 
![](https://i.imgur.com/cp4lVS1.png)
*Equation 1: Bounding box prediction loss*


The second part of YOLO's loss function is classification loss. Unconventionally, the classification loss is not defined as cross entropy loss, but mean square error. In the original paper, the author did not specify a reason for this. Our hypothesis is that cross entropy can be infinitely large when prediction is opposite to ground truth. This is fine when the task is merely classification. However, in YOLO, the task also includes regression; therefore, the regression learning might be disrupted by this infinite loss. As a result, the classification is defined to be a mean squred error loss. Besides, since most cells contain no objects, therefore, to balance many cells where no object exists, the author imposed two weight factors, $\lambda_\text{coord}$ and $\lambda_\text{cellnoobj}$ for correspoding terms in the loss funciton. 
This part of loss is also only defined for those cells where objects exist. 
![](https://i.imgur.com/2LlUkaK.png)
*Equation 2: Classification loss*

Lastly, The overall loss function expression is as follows.
![](https://i.imgur.com/CDpBPdE.png)
*Equation 3: Complete loss function*

### 1.2 Network Design
The network architecture of YOLO starts with a CNN back-bone(darknet, [ResNet-50](https://arxiv.org/abs/1512.03385), [VGG-16](https://neurohive.io/en/popular-networks/vgg16/)) after which the feature map is flattened and input to a fully connected layer. In the end, the output features of fully connected layer are reshaped and presented as final output. 
![](https://i.imgur.com/CejDWKx.png)
*Figure 2. Network Architecture*


---

## 2 Reproduction
In this section, we talk about how we implemented YOLOv1 method discussed in the paper. 
### 2.1 Training 
We adopted most of the original settings of the paper, while doing some adaptations. We used code from this [repository](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO) for importing data, creating dataset class, loss functions and some other utility functions like non-maximum suppresion, bounding boxes etc. Using this repository as the base, we added missing aspects to the training script so that we match the method followed by the paper. The following sections below are our additions:

#### 2.1.1 Train/Validation/Test Split:
For finetuning, we use PASCAL VOC dataset. We use all images from 2007-2012 for training, validation and testing. From a total of 43k images, we use 70%/15%/15% split for training/validation/testing. The path to the images and labels are stored in respective csv files for each split.  


#### 2.1.2 Pre-Training:
In the paper, darknet(YOLO architechture) is first trained on the 1000 class ImageNet dataset for a week to achieve 88% accuracy on ImageNet 2012 validation set. After that, the final classification layer is removed and two new fully connected layers are added.This network is then finetuned on the PASCAL VOC datset.

In our case however, pretraining for a week on the Imagenet dataset is unfeasable considering our project timeline. Therefore we decided to use PyTorch's pretrained models vgg16 and ResNet50 as our backbone for further finetuning. After importing the networks, we first removed the last fully connected and average pooling layer.Taking this feature extractor, we added our own adaptive pooling layer followed by two fully connected layers with a dropout and Leaky ReLu activation in between. With a probablity of 0.5 the dropout layer ensures that the network does not overfit. An illustration of the method is pictured below:

![](https://i.imgur.com/bpJud4Y.png)

With this network, we performed the following iterations:
##### 2.1.2.1 Vgg16 as feature extractor/backbone(Frozen weights):
As our feature extractor, we chose Vgg16 network because the same has been used in the paper for comparision with darket architechture. To understand the contribution of pre-training, we first froze all the pretrained weights by setting `requires_grad = False` in `model.parameters()`. 

Our intuition was that, considering vgg16 already has ~15 million parameters in the feature extractor, finetuning on a small dataset(43k images) would overfit the model. Also we were concerned that enabling the feature extactor weights to be modified would make it unlearn everything. Therefore for this iteration, backpropogation is enabled only for the fully connected layers. The training vs validation plot is shown below. 

![](https://i.imgur.com/vGKOI55.png)
*Figure 4. Training vs validation plot for Vgg16+YOLO with frozen feature extractor*

From the plot it can be observed that the model has clearly overfitted. 

##### 2.1.2.2 Vgg16 as feature extractor/backbone(Trainable weights):
For the next iteration we set `requires_grad = False` in `model.parameters()`, thereby enabling backpropogation for all layers. The training vs validation plot for the same is shown below.

![](https://i.imgur.com/7VxDe8L.png)
*Figure 5. Training vs validation plot for Vgg16+YOLO with trainable feature extractor*

From the figure it can be observed that this model has fared better than the one with frozen weights. Our assumption is that by enabling backpropogation to all layers, optimization took place in the whole feature space, allowing the network to to find a better optima. Also the images in Imagenet is of resolution 224x224. In our case, VOC dataset contains images of size 448x448. In restrospect, it makes much more sense to enable training for all layers. However to prevent catastrophic forgetting, we have kept the learning rate at a low value of 2e-5.

##### 2.1.2.3 ResNet50 as feature extractor/backbone(Trainable weights):

Next we tried changing the feature extractor to a more modern architechture like the ResNet50. ResNet50 contains ~23 million parameters in the feature extractor.. Moreover, layers in a Resnet50 also use Batch Normalization which is not used in the paper.  The training vs validation plot for the same is shown below.

![](https://i.imgur.com/mEN61J0.png)
*Figure 6. Training vs validation plot for ResNet50+YOLO with trainable feature extractor*

Using Resnet50 as the backbone, we see a 15% improvement in the validation accuracy.


#### 2.1.3 Data Augumentation:
We also adopted the data augmentations the paper used, which are random scaling and translations of up to 20% of the original image size, and randomly adjustment of exposure and saturation by up to a factor of 1.5. To be noted, when doing random translation, a bounding box label will be removed if the *center point (x,y)* is out of the image, otherwise it is preserved. Below we show what the images look like after the data augmentations. For the convenience of matching the format of data that we use, we wrote our own scripts for data augmentation, which adjusts bounding boxes correctly and matches Yolo's input format. These scripts are provided in this [repository](https://https://github.com/sentient-codebot/object-detection-data-aug).
![](https://i.imgur.com/LImFCCD.png)
*Figure 2. Original Images*
![](https://i.imgur.com/vUsu1gP.png)
*Figure 3. Images after data augmentation*

Besides, we also used random horizontal flip with a flipping probability of 0.5. As for dropout, we used the same settings as the paper, rate=0.5 after the first fully connected layer. 

#### 2.1.4 Learning Rate Scheduling:
Although we followed most of the methods mentioned in the paper our implementation gave us a result of 46% mAP vs 63.4% mAP reported in the paper. The two remaining differences were batch size and learning rate. Due to hardware limitations we had a batch size of 32 vs 64 in the paper.

With regards to learning rate, the paper follows a learning rate scedule. When we tried implementing the same values for our network, the training loss showed Nan values indicating exploding gradients. Therefore, our  learning  rate  schedule  is  as  follows:  For  the  first two epochs we keep a low value of $2*10^{-5}$ to prevent divering due to unstable gradients.Then we increase it to $8*10^{-5}$ and continue training with for 10 epochs, then $4*10^{-5}$ for 5 epochs, and finally $2*10^{-4}$ for remaining epochs.

Below we show a comparision for validation accuracy of ResNet50+YOLO model trained with and without learning rate scheduling. 
![](https://i.imgur.com/YWe6vlI.png)

From the plot, we can observe that at epoch 30, the model with rate scheduling has already reached an accuracy of 47% mAP as opposed to 42% mAP without. We could not continue further due to time constraints.

The above plot was obtained for a conservating rate sceduling. In the future we can try experimenting with higher initial learning rates, as given in the paper.



## 3 Results
### 3.1 Test Results
For test time, we use no transforms on the input images except resizing it to 448x448. After around 30 epochs, we achieved a training mAP of 55% and test mAP of 47%.

Here are a few images along with the ground truth and predicted bounding boxes. 
![](https://i.imgur.com/NlGEFDz.png)
*Figure 4. Test images and predicted bounding boxes and classes*
![](https://i.imgur.com/dLwqGMD.png)
*Figure 5. Transformed test images and predicted bounding boxes and classes*
As shown above, the trained model is robust against exposure and saturation distortion, mild scaling, and mild translation. Even when part of the object is out of the image, the model can still have a fair guess of boudary of the whole object. 

As for runtime speed, the complete inference process (starting from inputting the image to the network to finishing all the post processing like non-maximal suppression) runs at 120 fps on Nvidia V100, and at 11 fps on Nvidia GeForce 1660super. 


| Model | mAP | FPS | GPU |
| -------- | -------- | -------- | -------- |
| YOLO[2]   | 63.4%   | 45    | Titan-X    |
| YOLO+Resnet   |  47%    | 120     |  v100   |

---
### 3.2 WebCam performance
Another interesting way to visualize results in real-time is by connecting the object detection model to a webcam. We implemented an custom object detector using OpenCV and PyTorch. The webcam stream is preprocessed in order to accomodate changes in resolution, RGB channels and data format(3D to 4D tensor) for the model. The weights obtained after training with VOC dataset are loaded to the ResNet-50 model.

Images are captured using the webcam and passed throught the model for predicting bounding boxes. The predicted bounding boxes are plotted over the webcam image along with its class label and confidence score. Figure 6 shows the predicted bounding boxes of chair, dining table and person on the webcam data.


![](https://i.imgur.com/Esjzxv6.png)

*Figure 6. Predicted bounding boxes and classes in webcam images*

---

## 4 Conclusion
Through our reproduction, we have proved YOLO to be a solid object detection method, despite a small gap between achived test accuray and the accuray in paper. The main advantage of YOLO is fast and reasobably quick. This point is proved by running YOLO on our local machine to detect object in PASCAL images as well as web camera images. Even with our cheap graphics card, we still achieved reasonable detection results in a short time. 

However, there still remain a few issues. The biggest one is the proposed learning schedule is completely not applicable. Despite the author proposed to use $10^{-2}$ for the majority of ephocs, we noticed unstable gradients with learning rates larger than $10^{-4}$. The second issue is the confusing description about data augmentation in the paper. The author did not specify what to do when objects are moved out of image when doing random translation. The third one lies in YOLO itself. YOLO's way of formulating object detection problem limits that it can only detect B objects for one cell at most. 

With YOLO being an interesting and innovative method, we expect adaptions can be made on it to further reveal its potential. What's more, due to its straightforward architecture, some changes can be easily made, such as adding BatchNorm, and replacing with a better backend like ResNet. We also expect the ideas of YOLO can inspire people to propose even better algorithms (maybe they already did :). 


Contribution:

Chinmay Polya Ramesh - Training

Nan Lin - Data Augmentation and Test Results

Godwin Rayan Chandran - Webcam Results and Poster Presentation

## References
1. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detectionwith region proposal networks.arXiv preprint arXiv:1506.01497, 2015.
2. Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You Only Look Once: Unified, Real-Time Object Detection. In2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),pages 779–788, Las Vegas, NV, USA, June 2016. IEEE 
