<div align="center">
<h1> Traffic Sign Classification :vertical_traffic_light: </h1>
<i>Using Deep-learning to develop a model that accurately classifies over 43 types of traffic signs.</i>

<p align="center">
<img src="https://github.com/raofida75/Image-Classification-Application/blob/main/images/all-signs.png" width="700"/>
</p> </div>

## Problem Statement
Our road infrastructure would be incomplete without traffic signs. These traffic signs are simple for people to recognise; yet, detecting traffic signs remains a difficult task for computer systems. In this project, I will develop a traffic sign classifier that can recognise and understand traffic signs and it can be used in autonomous vehicles.

## Requirements

- Python
- Pandas
- NumPy 
- TensorFlow 
- Matplotlib
- seaborn
- sklearn

## Dataset
Dataset used: [German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
Place the training images in the `/data/Training` folder and testing images in `/data/Testing` folder.

This dataset has more than **50,000 images** of **43 classes**.

## Data Preparation

- crop out the background
- resizing the images to 32 x 32. 
- save the resulting pickle file

## Methodology
#### I have performed image classification using three different techniques i.e

1. <i>Neural network from scratch</i>

Neural Network was developed using numPy by implementing functions for all the required components such as forward propogation, cost function, backward propogation etc.  Adam optimizer was utilized; plus, regularization and drop-out was used to reduce overfitting. Moreover, PCA was employed to speed up the training of the model. The model gave **87%** accuracy on the test dataset.

[Link to the notebook](https://github.com/raofida75/Image-Classification-Application/blob/main/notebooks/1.%20classifier-from-scratch.ipynb)

2. <i>Convolutional Neural Network</i>

I have used tensorflow to develop complex CNN architectures such as Inception, Resnet, etc. to further improve the performance of the previous model.
With Convolutional Neural Network, I was able to achieve <b>96%</b> accuracy on the test dataset. This model will be used to create a streamlit-based web application.

[Link to the notebook](https://github.com/raofida75/Image-Classification-Application/blob/main/notebooks/2.%20classifier-using-cnn.ipynb)

3. <i>Semi Supervised GAN</i>

Semi-supervised learning is a method of training a classifier that employs both labelled and unlabeled data. This classifier utilises a small amount of labelled data and a large amount of unlabeled data. We will train a Deep CNN using these data sources to develop a model capable of assigning a desired label to a new datapoint. 

This model was able to achieve <i><b>87%</b></i> accuracy with just <b>20%</b> labelled data.

[Link to the notebook](https://github.com/raofida75/Image-Classification-Application/blob/main/notebooks/3.%20classifier-using-semisupervised-gan.ipynb)


## [Application](https://github.com/raofida75/Image-Classification-Application/blob/main/app.py)
Lastly, built an application using streamlit where user uploads an image and the model classifies the traffic sign.
<p align="center">
<img src="https://github.com/raofida75/Image-Classification-Application/blob/main/images/APP.png" width="750"/>
</p>

              
