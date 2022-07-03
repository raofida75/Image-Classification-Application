# Traffic Sign Classification

In this project, I used Python, NumPy, and TensorFlow to classify images of traffic signs.

Dataset used: [German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). This dataset has more than 50,000 images of 43 classes.

## Data Preparation
Download the dataset from kaggle. I have used pickle to save the training dataset and testing dataset after loading the images and resizing them to 32 x 32. Now I will used these pickled file to perform image classification using three different techniques i.e
1 - Neural network from scratch
2 - Various CNN architectures such as ResNet and Inception
3 - Semi Supervised GAN

## [Neural Network from scratch](https://github.com/raofida75/Image-Classification-Application/blob/main/1.%20NN%20from%20scratch/Traffic_Sign_Classifier_NumPy.ipynb)
Following helper functions were created to successfully build a Neural Network from scratch

1 - Initializing of Parameters: Use random initialization for the weight matrices, while zero initialization for the biases.

2 - Forward Propagation: The input data is fed forward through the network. Each hidden layer accepts input data, processes it according to the activation function, and then passes it on to the next layer.

3 - Cost Function: A neural network's cost function is the sum of errors in each layer. This is accomplished by first locating the error at each layer and       then adding the individual errors to obtain the total error.

4 - Backward Propagation: Back propagation is used to calculate the gradient of the loss function with respect to the parameters.

5 - Updating Parameters: Lastly, update the parameters using the gradients obtained from back propagation.

Created a Neural Network from scratch using numPy with an accuracy of 87%. Adam Optimizer was also implemented from scratch. Overfitting was reduced by using regularization and drop-out techniques. PCA was used to reduce image features without any significant loss in the quality of the images, which accelerated training speed. 

## Implementing various [CNN architecture such as Inception and ResNet](https://github.com/raofida75/Image-Classification-Application/blob/main/2.%20CNN/Traffic_Sign_Classifier_Keras.ipynb)
The performance of the model was further improved by employing advanced CNN architectures with Keras such as Inception, Resnet, etc. 

## [Semi Supervised Classification with GAN](https://github.com/raofida75/Image-Classification-Application/blob/main/3.%20Semi-supervised%20GAN/Semi_Supervised_Classification_with_GAN.ipynb) 
In addition, I developed an advanced classifier using semi-supervised GAN. It is useful if we do not have the entire dataset labelled. This model was able to achieve 87% accuracy with just 20% labelled data.

## [Application](https://github.com/raofida75/Image-Classification-Application/blob/main/app.py)
Lastly, built an application using stream-lit where user uploads an image and the model classifies the traffic sign.
<p align="center">
<img src="https://github.com/raofida75/Image-Classification-Application/blob/main/APP.png" width="750"/>
</p>
