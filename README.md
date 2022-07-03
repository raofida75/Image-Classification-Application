# Traffic Sign Classification

In this project, I used Python, NumPy, and TensorFlow to classify images of traffic signs.

Dataset used: [German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). This dataset has more than 50,000 images of 43 classes.

## Data Preparation
Download the dataset from kaggle. I have used pickle to save the training dataset and testing dataset after loading the images and resizing them to 32 x 32. Now I will used these pickled file to perform image classification using three different techniques i.e
1 - Neural network from scratch
2 - Various CNN architectures such as ResNet and Inception
3 - Semi Supervised GAN

## [Neural Network from scratch](https://github.com/raofida75/Image-Classification-Application/blob/main/1.%20NN%20from%20scratch/Traffic_Sign_Classifier_NumPy.ipynb)

Created a Neural Network from scratch using numPy with an accuracy of 87%. Adam Optimizer was also implemented from scratch. Overfitting was reduced by using regularization and drop-out techniques. PCA was used to reduce image features, which accelerated training speed. 

## Implementing various [CNN architecture such as Inception and ResNet] (https://github.com/raofida75/Image-Classification-Application/blob/main/2.%20CNN/Traffic_Sign_Classifier_Keras.ipynb)
The performance of the model was further improved by employing advanced CNN architectures with Keras such as Inception, Resnet, etc. 

## [Semi Supervised Classification with GAN](https://github.com/raofida75/Image-Classification-Application/blob/main/3.%20Semi-supervised%20GAN/Semi_Supervised_Classification_with_GAN.ipynb) 
In addition, I developed an advanced classifier using semi-supervised GAN. It is useful if we do not have the entire dataset labelled. This model was able to achieve 87% accuracy with just 20% labelled data.

## [Application](https://github.com/raofida75/Image-Classification-Application/blob/main/app.py)
Lastly, built an application using stream-lit where user uploads an image and the model classifies the traffic sign.
<p align="center">
<img src="https://github.com/raofida75/Image-Classification-Application/blob/main/APP.png" width="750"/>
</p>
